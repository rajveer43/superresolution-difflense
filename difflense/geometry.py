"""Detector geometry. The only module that knows what a plane and a grid axis mean.

Data are stacks of 2-D planes (P, U, V):
  jet    : CMS quark/gluon images. Planes = tracks / ECAL / HCAL, axes = (eta, phi), native 128 x 128
           (125 x 125 zero-padded so 2x pooling divides evenly). Pixel = 0.0174 (ECAL crystal).
  shower : CaloChallenge DS2. Planes = 45 longitudinal layers, axes = (angle, radius), native 16 x 9.
           Super-resolution happens in the transverse plane and keeps the layers, as the paper does
           in (eta, phi). The angular axis is periodic.

A cell on a grid (nu, nv) covers native cells [iu*nU/nu, (iu+1)*nU/nu) - also for grids finer than
native, where this is a fraction of a native cell. All physical coordinates follow from that, so
LR, SR, HR and the extrapolated grids are measured with the same code.
"""
import math

import numpy as np
import torch

CALO_R_EDGES = [0.0, 4.65, 9.3, 13.95, 18.6, 27.9, 37.2, 46.5, 55.8, 65.1]  # mm, binning_dataset_2.xml


class Geometry:
    def __init__(self, cfg):
        d = cfg["data"]
        self.kind = d["kind"]
        pl = d["planes"]
        self.planes = [f"layer{i}" for i in range(pl)] if isinstance(pl, int) else list(pl)
        self.P = len(self.planes)
        self.native = tuple(int(x) for x in d["hr_shape"])
        self.circular_u = bool(d.get("circular_u", False))
        # neighbouring planes a node attends to at the same (u, v)
        self.plane_offsets = list(d.get("plane_offsets", [-1, 1]))
        if self.kind == "jet":
            self.pixel = float(d.get("pixel_size", 0.0174))
            self.center = float(d.get("center_index", 63.0))
        else:
            self.r_edges_native = np.asarray(d.get("r_edges", CALO_R_EDGES), dtype=np.float64)
            self.layer_groups = d.get("layer_groups", [[0, 15], [15, 30], [30, 45]])
        self._rc = {}

    # ---------------------------------------------------------------- grids
    def area(self, shape):
        """Cell area relative to a native cell (4 for a 2x pooled grid, 1/4 for 2x finer)."""
        return (self.native[0] * self.native[1]) / (shape[0] * shape[1])

    def grid_name(self, shape):
        return f"{shape[0]}x{shape[1]}"

    def is_supervised(self, shape):
        return shape[0] <= self.native[0] and shape[1] <= self.native[1]

    # ------------------------------------------------------------ positions
    def r_edges(self, nv):
        """Radial bin edges (mm) on a grid with nv radial cells: merged or evenly subdivided native bins."""
        R = len(self.r_edges_native) - 1
        e = self.r_edges_native
        if nv <= R:
            if R % nv:
                raise ValueError(f"radial grid {nv} does not divide {R}")
            return e[:: R // nv]
        if nv % R:
            raise ValueError(f"radial grid {nv} is not a multiple of {R}")
        k = nv // R
        return np.concatenate([np.linspace(a, b, k + 1)[:-1] for a, b in zip(e[:-1], e[1:])] + [e[-1:]])

    def r_centers(self, nv):
        if nv not in self._rc:
            e = self.r_edges(nv)
            self._rc[nv] = 0.5 * (e[1:] + e[:-1])
        return self._rc[nv]

    def native_center(self, i, n, N):
        """Centre of cell i on an n-cell axis, in native index units (native cell k has centre k)."""
        return (i + 0.5) * (N / n) - 0.5

    def physical(self, pl, iu, iv, nu, nv):
        """numpy -> dict of physical cell coordinates.
        jet: eta, phi;  shower: layer, angle, r, x, y."""
        iu = np.asarray(iu, dtype=np.float64)
        iv = np.asarray(iv, dtype=np.float64)
        nu = np.broadcast_to(np.asarray(nu), iu.shape)
        nv = np.broadcast_to(np.asarray(nv), iv.shape)
        if self.kind == "jet":
            return {"eta": (self.native_center(iu, nu, self.native[0]) - self.center) * self.pixel,
                    "phi": (self.native_center(iv, nv, self.native[1]) - self.center) * self.pixel}
        a = 2 * math.pi * (iu + 0.5) / nu
        r = np.empty_like(iv)
        for n in np.unique(nv):
            m = nv == n
            r[m] = self.r_centers(int(n))[iv[m].astype(int)]
        return {"layer": np.asarray(pl, dtype=np.float64), "angle": a, "r": r,
                "x": r * np.cos(a), "y": r * np.sin(a)}

    def pos_dim(self):
        return 3

    def pos_features(self, iu, iv, nu, nv):
        """Scale-free node position features (torch): the same physical point gives the same value
        on every grid, so one model can act at every stage, including the extrapolated ones."""
        iu, iv, nu, nv = iu.float(), iv.float(), nu.float(), nv.float()
        if self.kind == "jet":
            fu = (iu + 0.5) / nu * 2 - 1
            fv = (iv + 0.5) / nv * 2 - 1
            return torch.stack([fu, fv, torch.sqrt(fu ** 2 + fv ** 2)], -1)
        a = 2 * math.pi * (iu + 0.5) / nu
        r = torch.empty_like(iv)
        for n in torch.unique(nv).tolist():
            m = nv == n
            rc = torch.as_tensor(self.r_centers(int(n)), dtype=torch.float32, device=iv.device)
            r[m] = rc[iv[m].long()] / float(self.r_edges_native[-1])
        return torch.stack([torch.sin(a), torch.cos(a), r], -1)


def grid_ladder(geom, cfg):
    """[(grid shape, supervised?)] from the measured LR grid to the last stage."""
    lad = cfg["ladder"]
    shape = tuple(n // f for n, f in zip(geom.native, lad["lr_factor"]))
    if any(n * f != N for n, f, N in zip(shape, lad["lr_factor"], geom.native)):
        raise ValueError(f"lr_factor {lad['lr_factor']} does not divide native grid {geom.native}")
    out = [shape]
    for f in lad["stages"]:
        shape = (shape[0] * f[0], shape[1] * f[1])
        out.append(shape)
    if geom.native not in out:
        raise ValueError(f"ladder {out} never reaches the native grid {geom.native}; no stage can be checked")
    return out


def train_transitions(geom, cfg):
    """(parent shape, child shape, factor) pairs with truth on both sides."""
    lad = cfg["ladder"]
    if lad.get("train_transitions"):
        pairs = [(tuple(p), tuple(c)) for p, c in lad["train_transitions"]]
    else:
        grids = [g for g in grid_ladder(geom, cfg) if geom.is_supervised(g)]
        pairs = list(zip(grids[:-1], grids[1:]))
        # one coarser transition below the measured LR, when the factor divides: more scales to learn from
        f = lad["stages"][0]
        coarser = (grids[0][0] // f[0], grids[0][1] // f[1])
        if coarser[0] * f[0] == grids[0][0] and coarser[1] * f[1] == grids[0][1] and min(coarser) >= 2:
            pairs.insert(0, (coarser, grids[0]))
    out = []
    for p, c in pairs:
        if c[0] % p[0] or c[1] % p[1] or not geom.is_supervised(c):
            raise ValueError(f"bad training transition {p} -> {c}")
        out.append((p, c, (c[0] // p[0], c[1] // p[1])))
    return out
