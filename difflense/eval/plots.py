"""Figures (png + pdf). Colours follow one fixed categorical order so a method keeps its colour in
every figure: truth = ink (step line), SR = blue, LR = orange, uniform = aqua, extra = yellow/magenta.
Energy maps: one-hue sequential scale on log energy. Residuals: diverging scale centred on zero."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

INK = "#0b0b0b"
MUTED = "#52514e"
COLORS = {"truth": INK, "hr": INK, "sr": "#2a78d6", "lr": "#eb6834", "lr_up": "#eb6834",
          "uniform": "#1baf7a"}
EXTRA = ["#eda100", "#e87ba4", "#008300", "#4a3aa7"]
LABELS = {"hr": "HR truth", "truth": "HR truth", "sr": "SR (predicted)", "lr": "LR (measured)",
          "lr_up": "LR (measured)", "uniform": "LR spread uniformly"}
SEQ = "Blues"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "legend.frameon": False, "lines.linewidth": 2,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})


def color(name, i=0):
    return COLORS.get(name.split("@")[0], EXTRA[i % len(EXTRA)])


def save(fig, directory, name):
    os.makedirs(directory, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(directory, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _show(ax, img, vmin, vmax, title):
    img = np.where(img > 0, img, np.nan)
    im = ax.imshow(img.T, origin="lower", cmap=SEQ, norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return im


def event_ladder(lr, srs, hrs, planes, directory, name, axis_labels=("u", "v")):
    """One event: rows = planes; columns = measured LR, SR at each truth-checked grid, truth, SR - truth."""
    grids = list(srs)
    ncol = 1 + len(grids) + 2
    fig, axes = plt.subplots(len(planes), ncol, figsize=(1.9 * ncol, 1.9 * len(planes)), squeeze=False)
    for p, pname in enumerate(planes):
        vals = np.concatenate([a[p][a[p] > 0] for a in [lr] + list(srs.values()) + list(hrs.values())] + [[1e-6]])
        vmin, vmax = np.percentile(vals, 1), vals.max()
        im = _show(axes[p, 0], lr[p], vmin, vmax, f"LR measured {lr.shape[1]}x{lr.shape[2]}")
        for j, g in enumerate(grids):
            _show(axes[p, 1 + j], srs[g][p], vmin, vmax, f"SR {g[0]}x{g[1]}")
        g = grids[-1]
        _show(axes[p, ncol - 2], hrs[g][p], vmin, vmax, f"truth {g[0]}x{g[1]}")
        d = srs[g][p] - hrs[g][p]
        lim = max(np.abs(d).max(), 1e-9)
        ax = axes[p, ncol - 1]
        imd = ax.imshow(d.T, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
        ax.set_title("SR - truth", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        axes[p, 0].set_ylabel(pname)
        fig.colorbar(im, ax=axes[p, ncol - 2], fraction=0.046, pad=0.02, label="energy")
        fig.colorbar(imd, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    save(fig, directory, name)


def histograms(obs, names, directory, name, bins=50, logy=False):
    """obs: {method: {observable: values}}; truth drawn as a step line, others as thin lines."""
    names = [n for n in names if all(n in o for o in obs.values())]
    nc = 4
    nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.4 * nr), squeeze=False)
    for ax, n in zip(axes.ravel(), names):
        ref = obs.get("hr", next(iter(obs.values())))[n]
        lo, hi = np.nanpercentile(np.concatenate([o[n] for o in obs.values()]), [0.5, 99.5])
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
        for i, (m, o) in enumerate(obs.items()):
            ax.hist(o[n], bins=edges, histtype="step", density=True, color=color(m, i),
                    lw=2.4 if m == "hr" else 1.6, label=LABELS.get(m.split("@")[0], m))
        ax.set_title(n)
        if logy:
            ax.set_yscale("log")
        del ref
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, directory, name)


def residual_hist(res, directory, name, xlabel="(E_pred - E_true) / E_true"):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    allv = np.concatenate(list(res.values()))
    lo, hi = np.percentile(allv, [1, 99])
    edges = np.linspace(lo, hi, 60)
    for i, (m, r) in enumerate(res.items()):
        ax.hist(r, bins=edges, histtype="step", density=True, color=color(m, i),
                label=f"{LABELS.get(m, m)}: median {np.median(r):+.3f}, "
                      f"σ {np.subtract(*np.percentile(r, [75, 25])) / 1.349:.3f}")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    save(fig, directory, name)


def ensemble_curve(sizes, widths, directory, name):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(sizes, widths, "o-", color=COLORS["sr"], ms=5)
    ax.set_xlabel("ensemble size")
    ax.set_ylabel("σ of event energy residual")
    ax.set_xscale("log")
    save(fig, directory, name)


def curves(x, ys, xlabel, ylabel, directory, name, logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, (lab, y) in enumerate(ys.items()):
        ax.plot(x if not isinstance(x, dict) else x[lab], y, "o-", ms=4, color=EXTRA[i % len(EXTRA)]
                if lab not in COLORS else COLORS[lab], label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    if len(ys) > 1:
        ax.legend(fontsize=7)
    save(fig, directory, name)


def zoom_ladder(images, directory, name, plane_name, zoom_frac=0.25):
    """images: [(grid, 2-D array)] of one plane from the measurement to the last grid.
    Top row: full image; bottom row: the same central physical window (zoom_frac of the image)."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.2), squeeze=False)
    vals = np.concatenate([im[im > 0] / (im.shape[0] * im.shape[1]) ** -1 for _, im in images] + [[1e-9]])
    for j, (g, im) in enumerate(images):
        dens = im * (im.shape[0] * im.shape[1])  # energy density: grids are comparable
        vmin, vmax = np.percentile(vals, 2), vals.max()
        _show(axes[0, j], dens, vmin, vmax, f"{g[0]}x{g[1]}")
        U, V = im.shape
        hu, hv = int(U * zoom_frac / 2), int(V * zoom_frac / 2)
        cu, cv = np.unravel_index(np.argmax(im), im.shape)
        cu, cv = np.clip(cu, hu, U - hu), np.clip(cv, hv, V - hv)
        _show(axes[1, j], dens[cu - hu:cu + hu, cv - hv:cv + hv], vmin, vmax, "zoom (same physical window)")
    axes[0, 0].set_ylabel(plane_name)
    fig.tight_layout()
    save(fig, directory, name)


def roc(curves_, directory, name):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, (lab, (tpr, fpr, auc)) in enumerate(curves_.items()):
        ax.plot(tpr, 1 / np.maximum(fpr, 1e-4), color=color(lab, i), label=f"{lab}: AUC {auc:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("signal efficiency ε_S (quark)")
    ax.set_ylabel("background rejection 1/ε_B")
    ax.legend(fontsize=7)
    save(fig, directory, name)


def training_curves(history_csv, directory, name, cols):
    import pandas as pd
    if not os.path.exists(history_csv):
        return
    h = pd.read_csv(history_csv)
    cols = [c for c in cols if c in h]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 2.6), squeeze=False)
    for ax, c in zip(axes[0], cols):
        ax.plot(h["epoch"], h[c], "o-", ms=3, color=COLORS["sr"])
        ax.set_title(c)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    save(fig, directory, name)
