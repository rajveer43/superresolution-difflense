"""Physics observables from sparse cells (any grid, any mix of grids), vectorised over events.

cells: dict of numpy arrays ev, pl, iu, iv, nu, nv, E (one entry per non-empty cell). Coordinates come
from Geometry.physical, so LR, SR, HR and extrapolated grids are measured identically.

Jets (each cell a massless constituent with pT = cell value, as in the CMS E2E images):
    sum_<plane>, sum_all, jet_pt, jet_mass, girth, dR_std, C2, C3, D2 (energy correlators, beta = 1, on the
    ecf_top_k hottest cells, paper figure 7); grid-dependent: ptD, n_hits
Showers:
    e_total, e_ratio = E_dep / E_inc, depth_centroid / width (layers), r_centroid / width (mm), x / y
    centroid (mm), frac_early / mid / late, C2, C3, D2 (transverse distance / R_max); grid-dependent: n_hits
Grid-dependent observables change with the cell size even for a perfect reconstruction, so they are only
compared between images on the same grid.
"""
import numpy as np

GRID_DEPENDENT = ("n_hits", "ptD")


def dense_to_cells(img, offset=0):
    """(N, P, U, V) numpy -> cells dict (events numbered from `offset`)."""
    img = np.asarray(img)
    N, P, U, V = img.shape
    ev, pl, iu, iv = np.nonzero(img > 0)
    return {"ev": ev + offset, "pl": pl, "iu": iu, "iv": iv, "E": img[ev, pl, iu, iv].astype(np.float64),
            "nu": np.full(len(ev), U), "nv": np.full(len(ev), V)}


def torch_cells(parts):
    """Partition from the progressive sampler (list of torch node dicts) -> one numpy cells dict."""
    keys = ("ev", "pl", "iu", "iv", "nu", "nv", "E")
    return {k: np.concatenate([p[k].detach().cpu().numpy() for p in parts]).astype(
        np.float64 if k == "E" else np.int64) for k in keys}


def _bc(ev, w, n):
    return np.bincount(ev, weights=w, minlength=n)


def _top_k(ev, E, coords, n, k):
    """Per event, the k most energetic cells: z (n, k) normalised, coords (n, k, d); padding has z = 0."""
    order = np.lexsort((-E, ev))
    ev_s = ev[order]
    start = np.searchsorted(ev_s, np.arange(n))
    rank = np.arange(len(ev_s)) - start[ev_s]
    m = rank < k
    sel = order[m]
    z = np.zeros((n, k))
    X = np.zeros((n, k, coords.shape[1]))
    z[ev[sel], rank[m]] = E[sel]
    X[ev[sel], rank[m]] = coords[sel]
    z /= np.maximum(z.sum(1, keepdims=True), 1e-30)
    return z, X


def energy_correlators(z, X, scale=1.0, chunk=256):
    """C2 = e3 e1^3 / e2^2 ... (Larkoski, Salam, Thaler): with e1 = 1,
    C2 = e3 / e2^2, C3 = e4 e2 / e3^2, D2 = e3 / e2^3. beta = 1."""
    n = len(z)
    e2, e3, e4 = np.zeros(n), np.zeros(n), np.zeros(n)
    for s in range(0, n, chunk):
        zz, xx = z[s:s + chunk], X[s:s + chunk]
        th = np.sqrt(((xx[:, :, None] - xx[:, None]) ** 2).sum(-1)) / scale
        e2[s:s + chunk] = 0.5 * np.einsum("bi,bj,bij->b", zz, zz, th)
        e3[s:s + chunk] = np.einsum("bi,bj,bk,bij,bik,bjk->b", zz, zz, zz, th, th, th, optimize=True) / 6
        e4[s:s + chunk] = np.einsum("bi,bj,bk,bl,bij,bik,bil,bjk,bjl,bkl->b", zz, zz, zz, zz, th, th, th, th,
                                    th, th, optimize=True) / 24
    safe = lambda a: np.maximum(a, 1e-30)
    return {"C2": e3 / safe(e2 ** 2), "C3": e4 * e2 / safe(e3 ** 2), "D2": e3 / safe(e2 ** 3)}


def jet_observables(cells, n, geom, thr, ecf_k=32):
    ev, E = cells["ev"], cells["E"]
    c = geom.physical(cells["pl"], cells["iu"], cells["iv"], cells["nu"], cells["nv"])
    eta, phi = c["eta"], c["phi"]
    out = {}
    per_plane = np.bincount(ev * geom.P + cells["pl"], weights=E, minlength=n * geom.P).reshape(n, geom.P)
    for p, name in enumerate(geom.planes):
        out[f"sum_{name}"] = per_plane[:, p]
    tot = _bc(ev, E, n)
    safe = np.maximum(tot, 1e-30)
    px, py = _bc(ev, E * np.cos(phi), n), _bc(ev, E * np.sin(phi), n)
    pz, e = _bc(ev, E * np.sinh(eta), n), _bc(ev, E * np.cosh(eta), n)
    out["sum_all"] = tot
    out["jet_pt"] = np.hypot(px, py)
    out["jet_mass"] = np.sqrt(np.clip(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0, None))
    ce, cp = _bc(ev, E * eta, n) / safe, _bc(ev, E * phi, n) / safe
    dR = np.sqrt((eta - ce[ev]) ** 2 + (phi - cp[ev]) ** 2)
    out["girth"] = _bc(ev, E * dR, n) / safe
    out["dR_std"] = np.sqrt(np.clip(_bc(ev, E * dR ** 2, n) / safe - out["girth"] ** 2, 0, None))
    out["ptD"] = np.sqrt(_bc(ev, E ** 2, n)) / safe
    area = (geom.native[0] * geom.native[1]) / (cells["nu"] * cells["nv"])
    out["n_hits"] = _bc(ev, (E > thr[cells["pl"]] * area).astype(float), n)  # threshold scales with cell area
    z, X = _top_k(ev, E, np.stack([eta, phi], 1), n, ecf_k)
    out.update(energy_correlators(z, X))
    return out


def shower_observables(cells, n, geom, thr, e_inc, ecf_k=32):
    ev, E = cells["ev"], cells["E"]
    c = geom.physical(cells["pl"], cells["iu"], cells["iv"], cells["nu"], cells["nv"])
    tot = _bc(ev, E, n)
    safe = np.maximum(tot, 1e-30)
    L = c["layer"]
    out = {"e_total": tot, "e_ratio": tot / e_inc}
    d = _bc(ev, E * L, n) / safe
    out["depth_centroid"] = d
    out["depth_width"] = np.sqrt(np.clip(_bc(ev, E * L ** 2, n) / safe - d ** 2, 0, None))
    rc = _bc(ev, E * c["r"], n) / safe
    out["r_centroid"] = rc
    out["r_width"] = np.sqrt(np.clip(_bc(ev, E * c["r"] ** 2, n) / safe - rc ** 2, 0, None))
    out["x_centroid"] = _bc(ev, E * c["x"], n) / safe
    out["y_centroid"] = _bc(ev, E * c["y"], n) / safe
    for name, (a, b) in zip(("frac_early", "frac_mid", "frac_late"), geom.layer_groups):
        m = (L >= a) & (L < b)
        out[name] = _bc(ev[m], E[m], n) / safe
    area = (geom.native[0] * geom.native[1]) / (cells["nu"] * cells["nv"])
    out["n_hits"] = _bc(ev, (E > thr[cells["pl"]] * area).astype(float), n)
    z, X = _top_k(ev, E, np.stack([c["x"], c["y"]], 1), n, ecf_k)
    out.update(energy_correlators(z, X, scale=float(geom.r_edges_native[-1])))
    return out


def observables(cells, n, geom, thr, target=None, ecf_k=32):
    thr = np.asarray(thr, dtype=np.float64)
    if geom.kind == "jet":
        return jet_observables(cells, n, geom, thr, ecf_k)
    return shower_observables(cells, n, geom, thr, target, ecf_k)


def radial_power_spectrum(img, cells_per_native=1.0, n_bins=None):
    """Radially averaged power spectrum of a 2-D energy image (energy-normalised): structure per spatial
    frequency. Used to check that extrapolated grids add no artefacts (e.g. checkerboard at the cell pitch).
    Returns (frequency in cycles per native cell, power); cells_per_native = grid size / native size."""
    img = np.asarray(img, dtype=np.float64)
    img = img / max(img.sum(), 1e-30)
    U, V = img.shape
    F = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    ku = np.fft.fftshift(np.fft.fftfreq(U))
    kv = np.fft.fftshift(np.fft.fftfreq(V))
    K = np.sqrt(ku[:, None] ** 2 + kv[None] ** 2)
    n_bins = n_bins or min(U, V) // 2
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(K.ravel(), edges) - 1, 0, n_bins - 1)
    p = np.bincount(idx, weights=F.ravel(), minlength=n_bins) / np.maximum(np.bincount(idx, minlength=n_bins), 1)
    return 0.5 * (edges[1:] + edges[:-1]) * cells_per_native, p
