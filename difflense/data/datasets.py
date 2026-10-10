"""Cache access and torch datasets.

Normalisation used everywhere (data, sampler, tagger):
    n(E) = log1p(E / (s_p * a))
s_p = mean non-zero HR cell energy of plane p (train split), a = cell area relative to a native cell.
Dividing by the area makes LR, HR and finer-than-native cells comparable (energy density), which is
what lets one SR model act at every stage of the ladder.

Flow-matching variables of a child cell (paper section 3.2: the LR energy is copied to each child):
    c  = n(E_parent / K) on the child grid = parent energy density        (condition)
    x1 = n(E_child) - c                                                   (target: correction to the copy)
"""
import json
import math
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from ..geometry import Geometry, train_transitions
from ..physics.ops import measure_lr, pool_to, uniform_upsample
from .graphs import dense_to_nodes, neighbour_index, neighbour_offsets, split_parents
from .prepare import cache_file


def normalise(E, scale, area):
    return torch.log1p(E / (scale * area))


def denormalise(n, scale, area):
    return scale * area * torch.expm1(n.clamp_min(0))


class _Lazy:
    def __init__(self, fn, *arrays):
        self.fn, self.arrays = fn, arrays

    def __len__(self):
        return min(len(a) for a in self.arrays)

    def __getitem__(self, i):
        single = isinstance(i, (int, np.integer))
        parts = [np.asarray(a[[i]] if single else a[i], dtype=np.float32) for a in self.arrays]
        out = self.fn(*parts)
        return out[0] if single else out


class CacheStore:
    """kinds:
    hr            truth on the native grid           hr@UxV   truth sum-pooled to grid UxV
    lr            the measurement on its own grid    lr_up    measurement spread uniformly onto the native grid
    sr            SR output on the native grid       sr@UxV   SR output at an intermediate stage
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = cfg["paths"]["cache"]
        with open(os.path.join(self.dir, "meta.json")) as f:
            self.meta = json.load(f)
        if list(self.meta["lr_factor"]) != list(cfg["ladder"]["lr_factor"]):
            raise ValueError(f"cache was built with lr_factor {self.meta['lr_factor']}, config says "
                             f"{cfg['ladder']['lr_factor']}: rerun prepare with --force")
        self.geom = Geometry(cfg)
        self.scales = np.asarray(self.meta["scales"], dtype=np.float32)
        self.sigma = np.asarray(self.meta["noise_sigma"], dtype=np.float32)
        self.lr_shape = tuple(self.meta["lr_shape"])
        self.gen_dir = cfg["paths"].get("generated") or os.path.join(cfg["paths"]["out"], "generated")

    def n(self, split):
        return self.meta["counts"][split]

    def target(self, split):
        return np.load(cache_file(self.cfg, split, "target"))

    def sr_path(self, split, shape):
        return os.path.join(self.gen_dir, f"{split}_sr_{shape[0]}x{shape[1]}.npy")

    def has(self, split, kind):
        if kind.startswith("sr"):
            return os.path.exists(self.sr_path(split, self.kind_shape(kind)))
        return True

    def kind_shape(self, kind):
        m = re.match(r".*@(\d+)x(\d+)$", kind)
        if m:
            return int(m.group(1)), int(m.group(2))
        return self.lr_shape if kind == "lr" else self.geom.native

    def area(self, kind):
        return self.geom.area(self.kind_shape(kind))

    def array(self, split, kind):
        base = kind.split("@")[0]
        shape = self.kind_shape(kind)
        hr = np.load(cache_file(self.cfg, split, "hr"), mmap_mode="r")
        if base == "hr":
            return hr if shape == self.geom.native else _Lazy(lambda x: pool_to(x, shape), hr)
        lr = np.load(cache_file(self.cfg, split, "lr"), mmap_mode="r")
        if base == "lr":
            return lr
        if base == "lr_up":
            f = (self.geom.native[0] // self.lr_shape[0], self.geom.native[1] // self.lr_shape[1])
            return _Lazy(lambda x: uniform_upsample(x, f), lr)
        if base == "sr":
            return np.load(self.sr_path(split, shape), mmap_mode="r")
        raise ValueError(f"unknown input kind {kind}")


# ---------------------------------------------------------------- SR graphs
class Featurizer:
    """Builds model inputs for a set of child cells: condition, position features, neighbours."""

    def __init__(self, geom, scales, window):
        self.geom = geom
        self.scales = torch.as_tensor(scales, dtype=torch.float32)
        self.offsets = neighbour_offsets(window, geom.plane_offsets)
        self.native_area = float(geom.native[0] * geom.native[1])

    def area(self, ch):
        return self.native_area / (ch["nu"] * ch["nv"]).float()

    def __call__(self, ch, E_parent, K):
        s = self.scales.to(ch["pl"].device)[ch["pl"]]
        a = self.area(ch)
        ch["scale"], ch["area"] = s, a
        ch["c"] = normalise(E_parent / K, s, a)
        ch["pos"] = self.geom.pos_features(ch["iu"], ch["iv"], ch["nu"], ch["nv"])
        ch["nbr"] = neighbour_index(ch, self.offsets, self.geom.P, self.geom.circular_u)
        return ch


class TransitionDataset(Dataset):
    """One event, one (parent grid -> child grid) transition. Train: random transition, fresh noise on the
    measured grid, log-normal jitter on clean parents. val/test: transition i % T, the stored measurement."""

    def __init__(self, store, split, train, max_n=None):
        self.store, self.split, self.train = store, split, train
        self.cfg = store.cfg
        self.trans = train_transitions(store.geom, self.cfg)
        n = store.n(split)
        self.n = n if max_n is None else min(n, max_n)
        self._hr = self._lr = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._hr is None:
            self._hr = np.load(cache_file(self.cfg, self.split, "hr"), mmap_mode="r")
            self._lr = np.load(cache_file(self.cfg, self.split, "lr"), mmap_mode="r")
        st, g = self.store, self.store.geom
        if self.train:
            t = self.trans[np.random.randint(len(self.trans))]
        else:
            t = self.trans[i % len(self.trans)]
        pshape, cshape, f = t
        hr = np.asarray(self._hr[i], dtype=np.float32)
        child = torch.from_numpy(np.ascontiguousarray(pool_to(hr, cshape)))
        noisy = pshape == st.lr_shape and float(st.sigma.max()) > 0
        if pshape == st.lr_shape:
            if self.train and noisy:
                lf = (g.native[0] // pshape[0], g.native[1] // pshape[1])
                parent = measure_lr(hr[None], lf, st.sigma, self.cfg["noise"]["topo"], g.circular_u)[0]
            else:
                parent = torch.from_numpy(np.asarray(self._lr[i], dtype=np.float32))
        else:
            parent = torch.from_numpy(pool_to(hr, pshape))
            aug = self.cfg["noise"].get("cond_aug", 0.0)
            if self.train and aug > 0:
                parent = parent * torch.exp(torch.randn_like(parent) * aug)
        return {"parent": parent, "child": child, "factor": f, "noisy": int(noisy)}


class Collate:
    def __init__(self, feat):
        self.feat = feat

    def __call__(self, items):
        parts, flags, p_off = [], [], 0
        for b, it in enumerate(items):
            par = dense_to_nodes(it["parent"][None])
            par["ev"].fill_(b)
            ch = split_parents(par, it["factor"])
            K = it["factor"][0] * it["factor"][1]
            ch["E_true"] = it["child"][ch["pl"], ch["iu"], ch["iv"]]
            ch["E_parent"] = par["E"][ch["parent"]]
            ch["K"] = torch.full_like(ch["E_true"], float(K))
            ch["parent"] = ch["parent"] + p_off
            p_off += len(par["E"])
            parts.append(ch)
            flags.append(it["noisy"])
        batch = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        batch = self.feat(batch, batch["E_parent"], batch["K"])
        batch["x1"] = normalise(batch["E_true"], batch["scale"], batch["area"]) - batch["c"]
        batch["flag"] = torch.tensor(flags, dtype=torch.long)
        batch["B"] = len(items)
        batch["n_parents"] = p_off
        return batch


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ------------------------------------------------------------------ tagger
class TaggerDataset(Dataset):
    """(normalised image, per-plane log total energy, target) for any input kind.
    Augmentations respect the physics: jets are reflected in eta and phi (images are centred on the jet
    axis); showers are rotated about the beam axis and reflected in angle."""

    def __init__(self, store, split, kind, y, augment=False, max_n=None):
        self.store, self.split, self.kind = store, split, kind
        self.y = y.astype(np.float32)
        self.area = store.area(kind)
        self.augment = augment
        self.kind_geom = store.geom.kind
        self.arr = None
        n = len(y)
        self.n = n if max_n is None else min(n, max_n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = self.store.array(self.split, self.kind)
        E = torch.from_numpy(np.asarray(self.arr[i], dtype=np.float32))
        s = torch.as_tensor(self.store.scales).reshape(-1, 1, 1)
        x = torch.log1p(E / (s * self.area))
        g = torch.log1p(E.sum((1, 2)) / s.reshape(-1))
        if self.augment:
            if self.kind_geom == "jet":
                if np.random.rand() < 0.5:
                    x = x.flip(-2)
                if np.random.rand() < 0.5:
                    x = x.flip(-1)
            else:
                x = torch.roll(x, np.random.randint(x.shape[-2]), dims=-2)
                if np.random.rand() < 0.5:
                    x = x.flip(-2)
        return x, g, torch.tensor(self.y[i])


def regression_target(y, stats=None):
    """calo: standardised log E_inc (stats from the train split)."""
    ly = np.log(y)
    stats = stats or (float(ly.mean()), float(ly.std()))
    return (ly - stats[0]) / stats[1], stats
