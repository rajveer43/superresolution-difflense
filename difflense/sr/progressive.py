"""Progressive super-resolution: measured LR -> stage 1 -> ... -> native grid -> ... -> 2048 x 2048.

Stage 1 starts from the noisy measurement: it denoises and splits, and is not forced to the measured
energies (the measurement is biased by noise and clustering, the paper's point). Every later stage starts
from the previous output, which is clean, so its children are projected onto their parent energies:
energy is conserved exactly from stage 1 to the last grid.

Grids up to the native one are refined everywhere (they are compared with the truth). Beyond native there
is no truth and the node count grows 4x per stage, so refinement is adaptive: the most energetic cells are
split first, up to refine.max_nodes_per_event; the rest stay as leaves of their own grid. Leaves + active
cells always partition the image, and `render` turns any partition into a dense image on any grid
(a leaf spreads its energy evenly over the fine cells it covers, which keeps the energy exact).
"""
import time

import numpy as np
import torch

from ..data.datasets import Featurizer
from ..data.graphs import dense_to_nodes, nodes_to_dense, select
from ..geometry import grid_ladder
from ..models.flow import sr_stage

CORE = ("ev", "pl", "iu", "iv", "nu", "nv", "E")


def thresholds(cfg, P):
    t = cfg["data"]["hit_threshold"]
    return torch.as_tensor(np.full(P, float(t)) if np.isscalar(t) else np.asarray(t), dtype=torch.float32)


def refine_select(nodes, budget_parents, min_density, scales, native_area):
    """Split the most energetic cells of each event first; keep the rest as leaves."""
    if len(nodes["E"]) == 0:
        return nodes, nodes
    area = native_area / (nodes["nu"] * nodes["nv"]).float()
    dens = nodes["E"] / (scales.to(nodes["E"].device)[nodes["pl"]] * area)
    o1 = torch.argsort(nodes["E"], descending=True)
    o2 = torch.argsort(nodes["ev"][o1], stable=True)
    order = o1[o2]
    ev_sorted = nodes["ev"][order]
    start = torch.searchsorted(ev_sorted, ev_sorted, right=False)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order), device=order.device) - start
    keep = (dens > min_density) & (rank < budget_parents)
    return select(nodes, keep), select(nodes, ~keep)


def render(parts, B, P, shape):
    """Dense (B, P, *shape) image from any list of node dicts on grids coarser than or equal to `shape`."""
    dev = parts[0]["E"].device if parts else "cpu"
    out = torch.zeros(B, P, *shape, device=dev)
    for nodes in parts:
        if len(nodes["E"]) == 0:
            continue
        grids = torch.unique(torch.stack([nodes["nu"], nodes["nv"]], 1), dim=0)
        for nu, nv in grids.tolist():
            m = (nodes["nu"] == nu) & (nodes["nv"] == nv)
            fu, fv = shape[0] // nu, shape[1] // nv
            if fu * nu != shape[0] or fv * nv != shape[1]:
                raise ValueError(f"cannot render grid {nu}x{nv} onto {shape}")
            sub = select(nodes, m)
            img = nodes_to_dense(sub, B, P, (nu, nv))
            out += img.repeat_interleave(fu, -2).repeat_interleave(fv, -1) / (fu * fv)
    return out


class Progressive:
    def __init__(self, cfg, model, store, device):
        self.cfg, self.model, self.store, self.device = cfg, model.eval(), store, device
        self.geom = store.geom
        self.ladder = grid_ladder(self.geom, cfg)
        self.factors = [tuple(f) for f in cfg["ladder"]["stages"]]
        self.feat = Featurizer(self.geom, store.scales, cfg["sr_model"]["window"])
        self.thr = thresholds(cfg, self.geom.P)
        self.noisy = float(store.sigma.max()) > 0
        self.scales = torch.as_tensor(store.scales)

    @torch.no_grad()
    def run(self, lr, last_grid=None, ensemble=None, steps=None, refine=False, on_stage=None, generator=None):
        """lr: (B, P, u, v) measurement. Runs every stage up to `last_grid` (default: the whole ladder).
        Returns {grid: dense (B,P,grid) image} for grids <= native, and per-stage info.
        on_stage(grid, parts, info) is called after every stage with the full partition (leaves + active)."""
        s, r = self.cfg["sampling"], self.cfg["refine"]
        B, P = lr.shape[0], self.geom.P
        nodes = dense_to_nodes(lr.to(self.device))
        leaves, dense, info = [], {}, []
        for k, f in enumerate(self.factors):
            grid = self.ladder[k + 1]
            supervised = self.geom.is_supervised(grid)
            first = k == 0
            parents = nodes
            if not supervised and refine:
                K = f[0] * f[1]
                parents, frozen = refine_select(nodes, r["max_nodes_per_event"] // K, r["min_density"],
                                                self.scales, self.feat.native_area)
                leaves.append(frozen)
            ens = ensemble or (s["ensemble"] if supervised else r["ensemble"])
            stp = steps or (s["steps"] if supervised else r["steps"])
            t0 = time.time()
            ch = sr_stage(self.model, self.feat, parents, f, first and self.noisy, B, self.cfg, ens, stp,
                          s["solver"], self.thr, project=not (first and self.noisy), generator=generator)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            st = {"grid": grid, "supervised": supervised, "seconds": time.time() - t0,
                  "nodes": int(len(ch["E"])), "members_sum": ch["E_members_sum"].cpu()}
            nodes = select({k_: ch[k_] for k_ in CORE}, ch["E"] > 0)
            st["active_after"] = int(len(nodes["E"]))
            st["leaves"] = int(sum(len(l_["E"]) for l_ in leaves))
            info.append(st)
            if supervised:
                dense[grid] = nodes_to_dense(nodes, B, P, grid)
            if on_stage is not None:
                on_stage(grid, leaves + [nodes], st)
            if last_grid is not None and grid == tuple(last_grid):
                break
        return dense, info, leaves + [nodes]


def load_sr_model(cfg, store, device, which="best"):
    import os
    from ..models.sr_transformer import build_sr_model
    feat = Featurizer(store.geom, store.scales, cfg["sr_model"]["window"])
    model = build_sr_model(cfg, store.geom, len(feat.offsets)).to(device)
    path = os.path.join(cfg["paths"]["out"], "sr_model", f"{which}.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    return model.eval()
