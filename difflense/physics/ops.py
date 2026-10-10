"""Energy-conserving resolution changes and the LR measurement model. numpy or torch, (..., U, V).

sum_pool         merge fu x fv cells: what a coarser detector records; total energy unchanged.
uniform_upsample spread each coarse cell evenly over its fine cells (energy-preserving baseline).
measure_lr       sum-pool + Gaussian electronic noise + 4-2-0 topological clustering, keeping only
                 clustered cells with positive energy (paper section 2).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _torch(x):
    return isinstance(x, torch.Tensor)


def sum_pool(x, f):
    fu, fv = int(f[0]), int(f[1])
    if fu == 1 and fv == 1:
        return x
    *lead, U, V = x.shape
    if U % fu or V % fv:
        raise ValueError(f"grid {U}x{V} not divisible by {fu}x{fv}")
    y = x.reshape(*lead, U // fu, fu, V // fv, fv)
    return y.sum(dim=(-3, -1)) if _torch(y) else y.sum(axis=(-3, -1))


def repeat_upsample(x, f):
    fu, fv = int(f[0]), int(f[1])
    if _torch(x):
        return x.repeat_interleave(fu, dim=-2).repeat_interleave(fv, dim=-1)
    return np.repeat(np.repeat(x, fu, axis=-2), fv, axis=-1)


def uniform_upsample(x, f):
    return repeat_upsample(x, f) / (int(f[0]) * int(f[1]))


def factor_between(coarse, fine):
    return (fine[0] // coarse[0], fine[1] // coarse[1])


def pool_to(x, shape):
    """Sum-pool a (..., U, V) array to grid `shape`."""
    return sum_pool(x, factor_between(shape, x.shape[-2:]))


def _dilate(mask, circular_u):
    """8-neighbour dilation of a boolean (N, P, U, V) torch mask, periodic along U if circular_u."""
    m = mask.float()
    N, P, U, V = m.shape
    m = m.reshape(N * P, 1, U, V)
    if circular_u:
        m = torch.cat([m[:, :, -1:], m, m[:, :, :1]], 2)
        m = F.pad(m, (1, 1, 0, 0))
    else:
        m = F.pad(m, (1, 1, 1, 1))
    return (F.max_pool2d(m, 3, 1) > 0).reshape(N, P, U, V)


def topo_cluster(E, sigma, thresholds=(4.0, 2.0, 0.0), circular_u=False, max_iter=50):
    """ATLAS-style topological clustering on (N, P, U, V) energies, in the transverse plane of each plane.
    Seeds |E| > t_seed*sigma, grow through cells with |E| > t_grow*sigma, then add boundary neighbours
    with |E| > t_cell*sigma. Returns the boolean mask of clustered cells."""
    t_seed, t_grow, t_cell = thresholds
    s = sigma.reshape(1, -1, 1, 1)
    a = E.abs()
    m = a > t_seed * s
    grow = a > t_grow * s
    for _ in range(max_iter):
        new = _dilate(m, circular_u) & grow | m
        if torch.equal(new, m):
            break
        m = new
    return _dilate(m, circular_u) & (a > t_cell * s) | m


def measure_lr(hr, factor, sigma, thresholds, circular_u=False, generator=None):
    """HR (N, P, U, V) -> noisy, clustered LR on the grid pooled by `factor` (torch, float32).
    sigma: (P,) noise per LR cell; a plane with sigma 0 is only pooled (e.g. tracks)."""
    hr = torch.as_tensor(hr, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    lr = sum_pool(hr, factor)
    if float(sigma.max()) <= 0:
        return lr
    noise = torch.randn(lr.shape, generator=generator) * sigma.reshape(1, -1, 1, 1)
    noisy = lr + noise
    keep = topo_cluster(noisy, sigma.clamp_min(1e-12), thresholds, circular_u)
    noiseless = (sigma <= 0).reshape(1, -1, 1, 1)
    out = torch.where(keep & (noisy > 0), noisy, torch.zeros_like(noisy))
    return torch.where(noiseless, lr, out)


def noise_sigma(cfg, scales):
    """Per-plane noise sigma per LR cell (data units)."""
    n = cfg["noise"]
    if not n.get("enabled", True):
        return np.zeros(len(scales), dtype=np.float32)
    rel = n["rel_sigma"]
    rel = np.full(len(scales), float(rel)) if np.isscalar(rel) else np.asarray(rel, dtype=np.float64)
    if len(rel) != len(scales):
        raise ValueError(f"noise.rel_sigma has {len(rel)} entries, data has {len(scales)} planes")
    return (rel * np.asarray(scales)).astype(np.float32)
