"""Downsample -> super-resolve -> compare with the HR truth, on every truth-checked grid of the ladder.

Methods compared with the truth:
    lr        the measurement (noisy, clustered, coarse) on its own grid
    uniform   the measurement spread evenly onto the finer grid (energy-preserving, no learning)
    sr        the model output
Outputs ({out}/evaluation/sr/): metrics.json, observables_<grid>.csv, figures/.
"""
import csv
import os

import numpy as np
import torch

from ..data.datasets import CacheStore
from ..physics.observables import GRID_DEPENDENT, dense_to_cells, observables
from ..physics.ops import pool_to, uniform_upsample
from ..sr.progressive import Progressive, load_sr_model, thresholds
from ..train.common import get_device
from . import plots
from .metrics import (cell_metrics, compare, energy_residuals, ensemble_pull, is_relative, robust_sigma,
                      sinkhorn_emd)


def _coords_fn(geom):
    def f(c):
        p = geom.physical(c["pl"], c["iu"], c["iv"], c["nu"], c["nv"])
        return np.stack([p["eta"], p["phi"]], 1) if geom.kind == "jet" else np.stack([p["x"], p["y"]], 1)
    return f


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate_sr(cfg):
    store = CacheStore(cfg)
    geom = store.geom
    ev_cfg = cfg["eval"]
    out = os.path.join(cfg["paths"]["out"], "evaluation", "sr")
    fig = os.path.join(out, "figures")
    os.makedirs(fig, exist_ok=True)
    native = geom.native
    sr_nat = store.array("test", "sr")
    n = min(len(sr_nat), store.n("test"), ev_cfg["max_events"] or 10 ** 9)
    hr = np.asarray(store.array("test", "hr")[:n], dtype=np.float32)
    lr = np.asarray(store.array("test", "lr")[:n], dtype=np.float32)
    target = store.target("test")[:n]
    thr = thresholds(cfg, geom.P).numpy()
    from ..geometry import grid_ladder
    grids = [g for g in grid_ladder(geom, cfg)[1:] if geom.is_supervised(g)]
    res = {"n_events": n, "grids": [list(g) for g in grids]}
    k = ev_cfg["ecf_top_k"]

    # ---- 1. event energy (paper fig. 6a), total and per plane
    sr = np.asarray(sr_nat[:n], dtype=np.float32)
    e_true, e_lr, e_sr = hr.sum((2, 3)), lr.sum((2, 3)), sr.sum((2, 3))
    res["energy"] = {}
    r_all = {}
    for name, e in (("lr", e_lr), ("sr", e_sr)):
        m, r = energy_residuals(e.sum(1), e_true.sum(1))
        res["energy"][name] = {"total": m, **{p: energy_residuals(e[:, i], e_true[:, i])[0]
                                              for i, p in enumerate(geom.planes)}}
        r_all[name] = r
    plots.residual_hist(r_all, fig, "energy_residual_total")

    # ---- 2. ensemble calibration of stage 1 (spread vs actual error of the event energy)
    mem_path = os.path.join(store.gen_dir, "test_stage1_members.npy")
    if os.path.exists(mem_path):
        mem = np.load(mem_path, mmap_mode="r")[:n].sum(2)
        res["ensemble_calibration"] = ensemble_pull(np.asarray(mem, dtype=np.float32), e_true.sum(1))

    # ---- 3. observables + cell metrics + EMD on each truth-checked grid
    res["grids_eval"] = {}
    for g in grids:
        key = f"{g[0]}x{g[1]}"
        hr_g = pool_to(hr, g)
        sr_g = sr if g == native else np.asarray(store.array("test", f"sr@{g[0]}x{g[1]}")[:n], dtype=np.float32)
        f_up = (g[0] // lr.shape[2], g[1] // lr.shape[3])
        uni_g = uniform_upsample(lr, f_up)
        cells = {"hr": dense_to_cells(hr_g), "sr": dense_to_cells(sr_g), "uniform": dense_to_cells(uni_g)}
        obs = {m: observables(c, n, geom, thr, target, k) for m, c in cells.items()}
        obs_lr = observables(dense_to_cells(lr), n, geom, thr, target, k)
        rows, comp = [], {}
        for o in obs["hr"]:
            for m, src in (("sr", obs["sr"]), ("uniform", obs["uniform"]), ("lr", obs_lr)):
                if m == "lr" and o.startswith(GRID_DEPENDENT):
                    continue  # cell-size dependent: LR is on another grid
                c = compare(obs["hr"][o], src[o], relative=is_relative(o))
                comp.setdefault(o, {})[m] = c
                rows.append({"grid": key, "observable": o, "method": m, **c})
        _write_csv(os.path.join(out, f"observables_{key}.csv"), rows)
        area = geom.area(g)
        cm = {m: cell_metrics(a, hr_g, thr, store.scales, area) for m, a in (("sr", sr_g), ("uniform", uni_g))}
        emd = {m: sinkhorn_emd(cells[m], cells["hr"], n, _coords_fn(geom), ev_cfg["emd_top_k"])
               for m in ("sr", "uniform")}
        res["grids_eval"][key] = {
            "observables": comp, "cells": cm,
            "emd": {m: {"median": float(np.nanmedian(v)), "mean": float(np.nanmean(v))} for m, v in emd.items()}}
        names = [o for o in obs["hr"] if not o.startswith(GRID_DEPENDENT)]
        plots.histograms({"hr": obs["hr"], "lr": obs_lr, "uniform": obs["uniform"], "sr": obs["sr"]}, names, fig,
                         f"observables_{key}", ev_cfg["hist_bins"])
        plots.histograms({"hr": obs["hr"], "lr": obs_lr, "sr": obs["sr"]}, ["C2", "C3", "D2"], fig,
                         f"energy_correlators_{key}", ev_cfg["hist_bins"])

    # ---- 4. example events across the ladder
    for i in range(min(ev_cfg["n_examples"], n)):
        srs = {g: (sr[i] if g == native else np.asarray(store.array("test", f"sr@{g[0]}x{g[1]}")[i]))
               for g in grids}
        hrs = {g: pool_to(hr[i], g) for g in grids}
        plots.event_ladder(lr[i], srs, hrs, geom.planes, fig, f"event_{i}")

    # ---- 5. ensemble-size study (paper fig. 6b): stage-1 energy resolution vs ensemble size
    sizes = ev_cfg.get("ensemble_study") or []
    if sizes:
        device = get_device(cfg)
        model = load_sr_model(cfg, store, device)
        prog = Progressive(cfg, model, store, device)
        m_ev = min(ev_cfg["ensemble_study_events"], n)
        x = torch.from_numpy(lr[:m_ev].astype(np.float32))
        widths = []
        for M in sizes:
            tots = []
            for a in range(0, m_ev, cfg["sampling"]["batch_size"]):
                gen = torch.Generator(device=device).manual_seed(a + 17 * M)
                _, info, parts = prog.run(x[a:a + cfg["sampling"]["batch_size"]], last_grid=prog.ladder[1],
                                          ensemble=M, generator=gen)
                tots.append(info[0]["members_sum"].numpy().sum(2).mean(1))
            r = np.concatenate(tots) / np.maximum(e_true[:m_ev].sum(1), 1e-30) - 1
            widths.append(robust_sigma(r))
        res["ensemble_study"] = {"sizes": sizes, "resolution": widths}
        plots.ensemble_curve(sizes, widths, fig, "ensemble_size")

    plots.training_curves(os.path.join(cfg["paths"]["out"], "sr_model", "history.csv"), fig, "sr_training",
                          ["train_loss", "val_loss", "phys_sr_res", "lr"])
    from ..config import save_config
    save_config(res, os.path.join(out, "metrics.json"))
    _print_summary(res)
    return res


def _print_summary(res):
    print("\nevent energy residual (total):")
    for m, v in res["energy"].items():
        t = v["total"]
        print(f"  {m:8s} bias {t['bias_median']:+.4f}   resolution {t['resolution_iqr']:.4f}")
    for g, v in res["grids_eval"].items():
        print(f"grid {g}:")
        for m, c in v["cells"].items():
            print(f"  {m:8s} hit F1 {c['hit_f1']:.3f}  E-weighted log err {c['energy_weighted_log_error']:.3f}  "
                  f"EMD {v['emd'][m]['median']:.4f}")
