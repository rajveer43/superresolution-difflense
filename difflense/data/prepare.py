"""Raw data -> cache of float32 memmaps (linear energies), streamed so the full datasets never sit in RAM.

    {cache}/meta.json
    {cache}/{split}_hr.npy       (N, P, U, V)  ground truth on the native grid (qg zero-padded to 128)
    {cache}/{split}_lr.npy       (N, P, u, v)  the measurement: HR sum-pooled by ladder.lr_factor, plus
                                               noise and topo-clustering, seeded per chunk (reproducible)
    {cache}/{split}_target.npy   (N,)          qg: 1 = quark, 0 = gluon;  calo: E_inc [MeV]

Per-plane scales (mean non-zero HR cell energy) and noise levels come from the train split only.
Sources:
  qg        ML4Sci CMS jet images: parquet (column X_jets, y), .h5/.hdf5, .npz or .pt with X_jets / y
  calo      CaloChallenge DS2: dataset_2_1.hdf5 (train + val) and dataset_2_2.hdf5 (test)
  synthetic small jet- or shower-like events, for tests and smoke runs
data.import_cache points at a cache of the older `superres` package and converts it (its split is kept).
"""
import glob
import json
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ..geometry import Geometry
from ..physics.ops import measure_lr, noise_sigma

SPLITS = ("train", "val", "test")
IMG_KEYS = ("X_jets", "X", "images", "jet_images", "x", "data")
LABEL_KEYS = ("y", "Y", "label", "labels", "target")
CHUNK = 1024


# ------------------------------------------------------------------ readers
def _nchw(X, P):
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] != P and X.shape[-1] == P:
        X = np.moveaxis(X, -1, 1)
    return np.ascontiguousarray(X)


def _qg_files(raw):
    exts = ("*.parquet", "*.h5", "*.hdf5", "*.npz", "*.pt")
    return sorted(f for e in exts for f in glob.glob(os.path.join(raw, "**", e), recursive=True))


def _load_small(path):
    if path.endswith(".npz"):
        z = np.load(path)
        obj = {k: z[k] for k in z.files}
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        obj = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in obj.items()}
    xk = next(k for k in IMG_KEYS if k in obj)
    yk = next(k for k in LABEL_KEYS if k in obj)
    return obj[xk], np.asarray(obj[yk], dtype=np.float64).reshape(len(obj[xk]), -1)[:, -1]


def _qg_count(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as h:
            return h[next(k for k in IMG_KEYS if k in h)].shape[0]
    return len(_load_small(path)[0])


def _iter_qg_file(path, P):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        xc = next(c for c in IMG_KEYS if c in cols)
        yc = next(c for c in LABEL_KEYS if c in cols)
        for b in pf.iter_batches(batch_size=CHUNK, columns=[xc, yc]):
            col = b.column(xc)
            shape = np.asarray(col[0].as_py()).shape
            flat = col
            while pa.types.is_list(flat.type) or pa.types.is_large_list(flat.type) \
                    or pa.types.is_fixed_size_list(flat.type):
                flat = flat.flatten()
            X = flat.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(col), *shape)
            yield _nchw(X, P), np.asarray(b.column(yc).to_pylist(), dtype=np.float64).reshape(len(col), -1)[:, -1]
    elif ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as h:
            xk = next(k for k in IMG_KEYS if k in h)
            yk = next(k for k in LABEL_KEYS if k in h)
            for s in range(0, h[xk].shape[0], CHUNK):
                yield _nchw(h[xk][s:s + CHUNK], P), np.asarray(h[yk][s:s + CHUNK], dtype=np.float64).reshape(-1)
    else:
        X, y = _load_small(path)
        for s in range(0, len(X), CHUNK):
            yield _nchw(X[s:s + CHUNK], P), y[s:s + CHUNK]


class Source:
    """n rows, iter() -> (X (k,P,U,V) float32 >= 0 on the native grid, y (k,)) in row order.
    fixed: None (random split) | int (rows >= fixed are the test split) | [n_train, n_val, n_test]."""

    def __init__(self, n, it, fixed=None):
        self.n, self.iter, self.fixed = n, it, fixed


def qg_source(cfg):
    d = cfg["data"]
    files = _qg_files(cfg["paths"]["raw"])
    if not files:
        raise FileNotFoundError(
            f"no jet files under {cfg['paths']['raw']}. Put the ML4Sci quark/gluon parquet/h5 files there "
            "(or set data.import_cache to an existing `superres` cache).")
    P = Geometry(cfg).P
    pad = [(0, 0), (0, 0)] + [tuple(p) for p in d.get("pad", [[1, 2], [1, 2]])]
    total = sum(_qg_count(f) for f in files)
    n = min(total, d.get("max_events") or total)
    # sample rows across all files: the class files are often separate
    keep = np.zeros(total, dtype=bool)
    keep[np.random.default_rng(cfg["seed"]).choice(total, n, replace=False)] = True

    def it():
        r = 0
        for path in files:
            print(f"  reading {path}")
            for X, y in _iter_qg_file(path, P):
                m = keep[r:r + len(X)]
                r += len(X)
                if m.any():
                    X = np.nan_to_num(X[m], nan=0.0, posinf=0.0, neginf=0.0)
                    yield np.pad(np.maximum(X, 0.0), pad).astype(np.float32), y[m]
    return Source(n, it)


def calo_source(cfg):
    import h5py
    d = cfg["data"]
    paths = [os.path.join(cfg["paths"]["raw"], f) for f in d["files"]]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing. Download CaloChallenge DS2 (Zenodo record 6366271).")
    caps = (d.get("max_events"), d.get("test_max"))
    ns = []
    for p, cap in zip(paths, caps):
        with h5py.File(p, "r") as h:
            ns.append(min(cap or h["showers"].shape[0], h["showers"].shape[0]))

    def it():
        for p, n in zip(paths, ns):
            with h5py.File(p, "r") as h:
                for s in range(0, n, CHUNK):
                    e = min(n, s + CHUNK)
                    S = h["showers"][s:e].astype(np.float32).reshape(e - s, *d["native_shape"])
                    yield np.maximum(S, 0.0), h["incident_energies"][s:e].astype(np.float64).reshape(-1)
    # dataset_2_2 is the CaloChallenge evaluation file: it is the test split
    return Source(sum(ns), it, fixed=ns[0])


def synthetic_source(cfg):
    """Jet-like (2 classes differing in width / multiplicity) or shower-like (energy-dependent) events."""
    d = cfg["data"]
    n = d.get("max_events") or 2000
    P = Geometry(cfg).P
    U, V = d["hr_shape"]

    def it():
        rng = np.random.default_rng(cfg["seed"])
        for s in range(0, n, CHUNK):
            k = min(CHUNK, n - s)
            X = np.zeros((k, P, U, V), dtype=np.float32)
            if d["kind"] == "jet":
                y = rng.integers(0, 2, k).astype(np.float64)
                uu, vv = np.meshgrid(np.arange(U), np.arange(V), indexing="ij")
                for i in range(k):
                    width = 0.8 if y[i] else 1.6
                    for _ in range(rng.poisson(3 if y[i] else 6) + 1):
                        cu, cv = rng.normal(U / 2, U / 14, 2)
                        e = rng.exponential(5.0)
                        for p in range(P):
                            w = width * (1 + p)
                            blob = np.exp(-((uu - cu) ** 2 + (vv - cv) ** 2) / (2 * w ** 2))
                            blob *= e * (0.5 if p == 0 else 1.0) / blob.sum()
                            X[i, p] += blob * (rng.random(blob.shape) < (0.15 if p == 0 else 1.0))
                X[X < d["hit_threshold"]] = 0
            else:
                y = np.exp(rng.uniform(np.log(1e3), np.log(1e5), k))
                layers = np.arange(P)[:, None, None]
                r = np.arange(V)[None, None, :]
                for i in range(k):
                    depth = 0.25 * P + 0.05 * P * np.log10(y[i] / 1e3) + rng.normal(0, 1)
                    prof = np.exp(-0.5 * ((layers - depth) / (0.12 * P)) ** 2)
                    lat = np.exp(-r / (1.0 + 0.05 * layers)) * (1 + 0.3 * rng.random((P, U, 1)))
                    sh = prof * lat * rng.gamma(4.0, 0.25, (P, U, V))
                    X[i] = sh / sh.sum() * y[i] * 0.95
                X[X < d["hit_threshold"]] = 0
            yield X, y
    return Source(n, it)


def import_source(cfg, path):
    """Arrays of the older `superres` package: qg (N,3,128,128) as is, calo (N,1,48,16,9) -> (N,45,16,9)."""
    d = cfg["data"]
    arrs = [(np.load(os.path.join(path, f"{s}_hr.npy"), mmap_mode="r"),
             np.load(os.path.join(path, f"{s}_target.npy"))) for s in SPLITS]

    def it():
        for h, y in arrs:
            for s in range(0, len(h), CHUNK):
                X = np.asarray(h[s:s + CHUNK], dtype=np.float32)
                if X.ndim == 5:
                    X = X[:, 0, : d["native_shape"][0]]
                yield X, y[s:s + CHUNK]
    return Source(sum(len(a) for a, _ in arrs), it, fixed=[len(a) for a, _ in arrs])


# -------------------------------------------------------------------- cache
def cache_file(cfg, split, what):
    return os.path.join(cfg["paths"]["cache"], f"{split}_{what}.npy")


def _split_ids(src, cfg, rng):
    """split id (0/1/2) and position inside the split for every row (rows shuffled within a split)."""
    N = src.n
    fr = np.asarray(cfg["data"]["split"], dtype=float) / sum(cfg["data"]["split"])
    sid = np.full(N, 2)
    if isinstance(src.fixed, list):
        b = np.cumsum([0] + src.fixed)
        for i in range(3):
            sid[b[i]:b[i + 1]] = i
    elif src.fixed is not None:
        n_tv = src.fixed
        perm = rng.permutation(n_tv)
        n_val = int(round(fr[1] / (fr[0] + fr[1]) * n_tv))
        sid[:n_tv] = 0
        sid[perm[:n_val]] = 1
    else:
        perm = rng.permutation(N)
        n_tr, n_va = int(fr[0] * N), int(fr[1] * N)
        sid[perm[:n_tr]] = 0
        sid[perm[n_tr:n_tr + n_va]] = 1
    pos = np.empty(N, dtype=np.int64)
    for i in range(3):
        rows = np.where(sid == i)[0]
        pos[rows] = rng.permutation(len(rows))
    return sid, pos


def prepare(cfg, force=False):
    cdir = cfg["paths"]["cache"]
    meta_path = os.path.join(cdir, "meta.json")
    if os.path.exists(meta_path) and not force:
        print(f"cache exists: {meta_path} (use --force to rebuild)")
        with open(meta_path) as f:
            return json.load(f)
    d = cfg["data"]
    geom = Geometry(cfg)
    rng = np.random.default_rng(cfg["seed"])
    if d.get("import_cache"):
        src = import_source(cfg, d["import_cache"])
    else:
        src = {"qg": qg_source, "calo": calo_source, "synthetic": synthetic_source}[d["source"]](cfg)
    sid, pos = _split_ids(src, cfg, rng)
    counts = {s: int((sid == i).sum()) for i, s in enumerate(SPLITS)}
    os.makedirs(cdir, exist_ok=True)

    # pass 1: HR + targets
    hr = {s: open_memmap(cache_file(cfg, s, "hr"), "w+", np.float32, (counts[s], geom.P, *geom.native))
          for s in SPLITS}
    tgt = {s: np.zeros(counts[s]) for s in SPLITS}
    r = 0
    for X, y in src.iter():
        if tuple(X.shape[1:]) != (geom.P, *geom.native):
            raise ValueError(f"data shape {X.shape[1:]} != planes x native grid {(geom.P, *geom.native)}")
        k = len(X)
        for i, s in enumerate(SPLITS):
            m = sid[r:r + k] == i
            if m.any():
                p = pos[r:r + k][m]
                hr[s][p] = X[m]
                tgt[s][p] = y[m]
        r += k
        print(f"\r  wrote {r}/{src.n}", end="")
    print()
    if r != src.n:
        raise RuntimeError(f"expected {src.n} rows, read {r}")
    for s in SPLITS:
        hr[s].flush()
        np.save(cache_file(cfg, s, "target"), tgt[s])

    # train-only statistics
    tr = np.asarray(hr["train"][: min(5000, counts["train"])])
    scales = []
    for p in range(geom.P):
        v = tr[:, p][tr[:, p] > 0]
        scales.append(float(v.mean()) if v.size else 1.0)
    sigma = noise_sigma(cfg, scales)

    # pass 2: the measured LR (fixed noise realisation per split, so every stage sees the same measurement)
    f = tuple(cfg["ladder"]["lr_factor"])
    lr_shape = (geom.native[0] // f[0], geom.native[1] // f[1])
    for si, s in enumerate(SPLITS):
        lr = open_memmap(cache_file(cfg, s, "lr"), "w+", np.float32, (counts[s], geom.P, *lr_shape))
        for a in range(0, counts[s], CHUNK):
            g = torch.Generator().manual_seed(int(cfg["seed"]) * 7919 + si * 1_000_003 + a)
            block = np.asarray(hr[s][a:a + CHUNK])
            lr[a:a + len(block)] = measure_lr(block, f, sigma, cfg["noise"]["topo"], geom.circular_u, g).numpy()
        lr.flush()
    labels = np.concatenate(list(tgt.values()))
    meta = {"dataset": cfg["dataset"], "kind": d["kind"], "planes": geom.planes, "native": list(geom.native),
            "lr_factor": list(f), "lr_shape": list(lr_shape), "counts": counts, "scales": scales,
            "noise_sigma": sigma.tolist(), "noise": cfg["noise"], "hit_threshold": d["hit_threshold"],
            "seed": cfg["seed"],
            "label_counts": ({str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
                             if d["kind"] == "jet" else None)}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"cache written to {cdir}: {counts}")
    return meta
