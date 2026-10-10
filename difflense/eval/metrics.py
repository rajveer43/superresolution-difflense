"""Metrics. No PSNR / SSIM: image-similarity scores reward blurring and ignore energy, which is the
quantity that matters. Every metric here answers a physics question.

Distribution level   W1 / sigma_truth, KS statistic, mean shift / sigma        (is the physics right?)
Event level          median bias and IQR/1.349 resolution of SR vs truth       (is each event right?)
Energy               event energy residual (paper fig. 6), energy closure       (is energy conserved?)
Cell level           hit precision / recall / F1, energy-weighted log error, cell-spectrum W1
Geometry             per-event energy mover's distance (Sinkhorn), in Delta R or mm
Uncertainty          ensemble pull (error / ensemble spread): calibrated if its width is ~1
Downstream           tagger AUC and 1/eps_B at eps_S, regressor response bias / resolution, bootstrap errors
"""
import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve

SIG_EFFS = (0.3, 0.5, 0.7)


def robust_sigma(x):
    x = np.asarray(x)
    if len(x) == 0:
        return float("nan")
    q75, q25 = np.percentile(x, [75, 25])
    return float((q75 - q25) / 1.349)


def compare(ref, other, relative=False):
    """ref, other: the same events' observable from truth and from a reconstruction."""
    a, b = np.asarray(ref, np.float64), np.asarray(other, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    sd = a.std() + 1e-12
    out = {"w1_over_sigma": float(wasserstein_distance(a, b) / sd), "ks": float(ks_2samp(a, b).statistic),
           "mean_shift_over_sigma": float((b.mean() - a.mean()) / sd),
           "pearson_r": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")}
    if relative:
        m = np.abs(a) > 1e-12
        r = b[m] / a[m] - 1
        out.update(event_bias=float(np.median(r)), event_resolution=robust_sigma(r), residual="relative")
    else:
        r = (b - a) / sd
        out.update(event_bias=float(np.median(r)), event_resolution=robust_sigma(r), residual="per sigma_truth")
    return out


RELATIVE = ("sum_", "e_total", "e_ratio", "jet_pt", "jet_mass")


def is_relative(name):
    return name.startswith(RELATIVE)


def cell_metrics(pred, true, thr, scales, area=1.0):
    """pred, true: (N, P, U, V) on the same grid. thr, scales: (P,)."""
    t = np.asarray(thr).reshape(1, -1, 1, 1) * area
    s = np.asarray(scales).reshape(1, -1, 1, 1) * area
    hp, ht = pred > t, true > t
    tp = float((hp & ht).sum())
    prec = tp / max(float(hp.sum()), 1)
    rec = tp / max(float(ht.sum()), 1)
    union = hp | ht
    lp, lt = np.log1p(pred / s), np.log1p(true / s)
    w = (pred + true)[union]
    wl1 = float((np.abs(lp - lt)[union] * w).sum() / max(w.sum(), 1e-30))
    vp, vt = pred[hp], true[ht]
    spec = float(wasserstein_distance(np.log10(vp), np.log10(vt))) if len(vp) and len(vt) else float("nan")
    return {"hit_precision": prec, "hit_recall": rec, "hit_f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "energy_weighted_log_error": wl1, "cell_spectrum_w1_log10": spec,
            "occupancy_ratio": float(hp.sum()) / max(float(ht.sum()), 1)}


def energy_residuals(pred_tot, true_tot):
    ok = true_tot > 0
    r = pred_tot[ok] / true_tot[ok] - 1
    return {"bias_median": float(np.median(r)), "bias_mean": float(r.mean()), "resolution_iqr": robust_sigma(r),
            "rms": float(r.std())}, r


def sinkhorn_emd(cells_a, cells_b, n, coords_fn, k=64, eps_frac=0.02, iters=300):
    """Per-event energy mover's distance between two energy distributions, normalised to unit energy
    (shape only; the energy scale is measured by energy_residuals). Top-k cells per event, log-domain
    Sinkhorn on CPU. coords_fn(cells) -> (n_cells, d) physical coordinates. Returns (n,) distances."""
    from ..physics.observables import _top_k
    za, Xa = _top_k(cells_a["ev"], cells_a["E"], coords_fn(cells_a), n, k)
    zb, Xb = _top_k(cells_b["ev"], cells_b["E"], coords_fn(cells_b), n, k)
    za, zb, Xa, Xb = (torch.as_tensor(v, dtype=torch.float64) for v in (za, zb, Xa, Xb))
    C = torch.cdist(Xa, Xb)                                          # (n, k, k)
    eps = eps_frac * C.flatten(1).median(1).values.clamp_min(1e-9)[:, None, None]
    la = torch.log(za.clamp_min(1e-300))
    lb = torch.log(zb.clamp_min(1e-300))
    f = torch.zeros_like(za)
    g = torch.zeros_like(zb)
    for _ in range(iters):
        f = -eps[:, :, 0] * torch.logsumexp((g[:, None] - C) / eps + lb[:, None], 2)
        g = -eps[:, :, 0] * torch.logsumexp((f[:, :, None] - C) / eps + la[:, :, None], 1)
    plan = torch.exp((f[:, :, None] + g[:, None] - C) / eps + la[:, :, None] + lb[:, None])
    d = (plan * C).sum((1, 2))
    empty = (za.sum(1) == 0) | (zb.sum(1) == 0)
    d[empty] = float("nan")
    return d.numpy()


def ensemble_pull(members_tot, true_tot):
    """members_tot: (N, M) per-event totals of each ensemble member. pull = (mean - truth) / std."""
    m, s = members_tot.mean(1), members_tot.std(1, ddof=1) if members_tot.shape[1] > 1 else None
    if s is None:
        return {}
    ok = (true_tot > 0) & (s > 0)
    pull = (m[ok] - true_tot[ok]) / s[ok]
    return {"pull_median": float(np.median(pull)), "pull_width_iqr": robust_sigma(pull),
            "frac_within_1sigma": float((np.abs(pull) < 1).mean())}


def rejection_at(y, score, eff):
    fpr, tpr, _ = roc_curve(y, score)
    return 1.0 / max(float(np.interp(eff, tpr, fpr)), 1e-6)


def classification_metrics(y, score, n_boot=200, seed=0):
    y = np.asarray(y).astype(int)
    out = {"auc": float(roc_auc_score(y, score)), "accuracy": float(((score > 0) == y).mean())}
    for e in SIG_EFFS:
        out[f"rej_at_eff{int(e * 100)}"] = rejection_at(y, score, e)
    rng = np.random.default_rng(seed)
    aucs, rejs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], score[i]))
            rejs.append(rejection_at(y[i], score[i], 0.5))
    out["auc_err"] = float(np.std(aucs)) if aucs else float("nan")
    out["rej_at_eff50_err"] = float(np.std(rejs)) if rejs else float("nan")
    return out


def regression_metrics(e_true, e_pred, n_bins=8, n_boot=200, seed=0):
    R = e_pred / e_true
    med = float(np.median(R))
    out = {"response_bias": med - 1, "resolution": robust_sigma(R) / med}
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_boot):
        i = rng.integers(0, len(R), len(R))
        res.append(robust_sigma(R[i]) / np.median(R[i]))
    out["resolution_err"] = float(np.std(res))
    edges = np.logspace(np.log10(e_true.min()), np.log10(e_true.max()) + 1e-9, n_bins + 1)
    out["binned"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() >= 20:
            mr = float(np.median(R[m]))
            out["binned"].append({"e_lo": float(lo), "e_hi": float(hi), "n": int(m.sum()), "bias": mr - 1,
                                  "resolution": robust_sigma(R[m]) / mr})
    return out
