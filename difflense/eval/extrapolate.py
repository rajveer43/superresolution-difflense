"""Progressive reconstruction beyond the native grid, up to the last grid of the ladder (2048 x 2048 for jets).

There is no truth finer than the native grid (CMS: one ECAL crystal), so these stages cannot be scored
against HR. They are checked on what must hold for any physical refinement:
  * energy closure      the partition at every grid sums exactly to the stage-1 (denoised) energy, per
                        event and per plane (refining a cell can only move energy inside it);
  * observable drift    physical, grid-independent observables (jet mass, girth, C2/D2, shower centroids and
                        widths) must not drift as cells get finer: reported vs the native-grid SR and vs truth;
  * spectra             radially averaged power spectrum per grid, in cycles per native cell: a physical
                        refinement continues the native spectrum smoothly, artefacts show up as peaks at the
                        cell pitch;
  * zoom panels         the same physical window at every scale.
How well the model's self-similar split extrapolates is measured separately and honestly with
configs/*_extrapolation_check.yaml: the last truth-checked transition is held out of training and scored
against truth (see README).
Outputs ({out}/evaluation/extrapolation/): metrics.json, drift.csv, events/*.npz (sparse multi-grid
partitions; difflense.sr.progressive.render turns one into a dense image), figures/.
"""
import csv
import os

import numpy as np
import torch

from ..config import save_config
from ..data.datasets import CacheStore
from ..geometry import grid_ladder
from ..physics.observables import GRID_DEPENDENT, dense_to_cells, observables, radial_power_spectrum, torch_cells
from ..sr.progressive import Progressive, load_sr_model, render, thresholds
from ..train.common import get_device, seed_everything
from . import plots
from .metrics import compare, is_relative


def _plane_view(parts, p):
    """Partition restricted to plane p, re-labelled as plane 0 (for single-plane rendering)."""
    out = []
    for q in parts:
        m = q["pl"] == p
        d = {k: v[m] for k, v in q.items()}
        d["pl"] = torch.zeros_like(d["pl"])
        out.append(d)
    return out


def extrapolate(cfg):
    seed_everything(cfg["seed"])
    device = get_device(cfg)
    store = CacheStore(cfg)
    geom = store.geom
    r = cfg["refine"]
    out = os.path.join(cfg["paths"]["out"], "evaluation", "extrapolation")
    fig = os.path.join(out, "figures")
    os.makedirs(os.path.join(out, "events"), exist_ok=True)
    model = load_sr_model(cfg, store, device)
    prog = Progressive(cfg, model, store, device)
    ladder = grid_ladder(geom, cfg)
    beyond = [g for g in ladder if not geom.is_supervised(g)]
    if not beyond:
        print("ladder ends at the native grid: nothing to extrapolate")
        return {}
    n = min(r["events"], store.n("test"))
    lr = np.load(os.path.join(cfg["paths"]["cache"], "test_lr.npy"), mmap_mode="r")
    hr = np.load(os.path.join(cfg["paths"]["cache"], "test_hr.npy"), mmap_mode="r")
    target = store.target("test")
    thr = thresholds(cfg, geom.P).numpy()
    k = cfg["eval"]["ecf_top_k"]
    zoom_plane = cfg["eval"].get("zoom_plane")

    obs = {g: [] for g in ladder[1:]}
    obs_truth = []
    closure, stage_info, spectra = [], {g: [] for g in ladder[1:]}, {g: [] for g in [geom.native] + beyond}
    for i in range(n):
        x = torch.from_numpy(np.asarray(lr[i:i + 1], dtype=np.float32))
        snaps = {}

        def on_stage(grid, parts, st, snaps=snaps):
            cells = torch_cells(parts)
            cells["ev"][:] = 0
            obs[grid].append(observables(cells, 1, geom, thr, target[i:i + 1], k))
            per_plane = np.bincount(cells["pl"], weights=cells["E"], minlength=geom.P)
            snaps[grid] = (per_plane, [{kk: v.clone() for kk, v in q.items()} for q in parts]
                           if (grid == geom.native or not geom.is_supervised(grid)) else None)
            stage_info[grid].append({kk: st[kk] for kk in ("seconds", "nodes", "active_after", "leaves")})

        gen = torch.Generator(device=device).manual_seed(int(cfg["seed"]) + i)
        prog.run(x, refine=True, on_stage=on_stage, generator=gen)
        obs_truth.append(observables(dense_to_cells(np.asarray(hr[i:i + 1])), 1, geom, thr, target[i:i + 1], k))

        # closure: every grid after stage 1 must hold the stage-1 energy, plane by plane
        e1 = snaps[ladder[1]][0]
        closure.append(max(float(np.abs(snaps[g][0] - e1).max() / max(e1.sum(), 1e-30)) for g in ladder[1:]))

        p = zoom_plane if zoom_plane is not None else int(np.argmax(e1))
        imgs = [(ladder[0], np.asarray(lr[i, p]))]
        for g in [geom.native] + beyond:
            parts = _plane_view(snaps[g][1], p)
            img = render(parts, 1, 1, g)[0, 0].cpu().numpy()
            spectra[g].append(radial_power_spectrum(img, cells_per_native=g[0] / geom.native[0]))
            imgs.append((g, img))
        if i < r["save_events"]:
            cells = torch_cells(snaps[beyond[-1]][1])
            np.savez_compressed(os.path.join(out, "events", f"event_{i}.npz"), **cells,
                                truth_native=np.asarray(hr[i]), measured=np.asarray(lr[i]))
        if i < cfg["eval"]["n_examples"]:
            plots.zoom_ladder(imgs, fig, f"zoom_event_{i}_plane_{geom.planes[p]}", geom.planes[p])
        print(f"\r  event {i + 1}/{n}", end="")
    print()

    # ---- aggregate
    stack = lambda lst: {kk: np.concatenate([d[kk] for d in lst]) for kk in lst[0]}
    O = {g: stack(v) for g, v in obs.items()}
    T = stack(obs_truth)
    names = [o for o in T if not o.startswith(GRID_DEPENDENT)]
    rows, drift = [], {}
    for g in [geom.native] + beyond:
        key = f"{g[0]}x{g[1]}"
        for o in names:
            vs_native = compare(O[geom.native][o], O[g][o], relative=is_relative(o))
            vs_truth = compare(T[o], O[g][o], relative=is_relative(o))
            drift.setdefault(o, {})[key] = {"vs_native_sr": vs_native, "vs_truth": vs_truth}
            rows.append({"grid": key, "observable": o, "drift_bias_vs_native_sr": vs_native["event_bias"],
                         "drift_res_vs_native_sr": vs_native["event_resolution"],
                         "bias_vs_truth": vs_truth["event_bias"], "res_vs_truth": vs_truth["event_resolution"],
                         "w1_over_sigma_vs_truth": vs_truth["w1_over_sigma"]})
    with open(os.path.join(out, "drift.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"n_events": n, "grids": [list(g) for g in ladder],
           "closure_max_rel": float(np.max(closure)), "closure_median_rel": float(np.median(closure)),
           "stages": {f"{g[0]}x{g[1]}": {kk: float(np.mean([s[kk] for s in v])) for kk in v[0]}
                      for g, v in stage_info.items() if v},
           "drift": drift}
    save_config(res, os.path.join(out, "metrics.json"))

    xs = [g[0] for g in [geom.native] + beyond]
    for o in ["jet_mass", "girth", "C2", "D2"] if geom.kind == "jet" else ["r_width", "depth_width", "C2", "D2"]:
        if o in drift:
            plots.curves(xs, {"bias vs native SR": [drift[o][f"{g[0]}x{g[1]}"]["vs_native_sr"]["event_bias"]
                                                    for g in [geom.native] + beyond],
                              "bias vs truth": [drift[o][f"{g[0]}x{g[1]}"]["vs_truth"]["event_bias"]
                                                for g in [geom.native] + beyond]},
                         "grid size (cells per axis)", f"{o}: median event residual", fig, f"drift_{o}", logx=True)
    spec_f = {f"{g[0]}x{g[1]}": spectra[g][0][0] for g in spectra if spectra[g]}
    spec_p = {f"{g[0]}x{g[1]}": np.mean([s[1] for s in spectra[g]], 0) for g in spectra if spectra[g]}
    plots.curves(spec_f, spec_p, "spatial frequency (cycles per native cell)", "power (energy-normalised)", fig,
                 "power_spectra", logx=True, logy=True)
    print(f"closure: max relative energy change across grids {res['closure_max_rel']:.2e}")
    for g, s in res["stages"].items():
        print(f"  {g}: {s['nodes']:.0f} nodes, {s['seconds']:.2f}s per event")
    return res
