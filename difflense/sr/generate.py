"""Run the truth-checked part of the ladder (measured LR -> ... -> native grid) on whole splits.

Writes, under {out}/generated/:
    {split}_sr_{U}x{V}.npy           (N, P, U, V) SR output (ensemble mean) on every grid up to native
    {split}_stage1_members.npy       (N, M, P) per-event, per-plane totals of each stage-1 ensemble member
    {split}_progress.json            resume point (generation of a large train split takes a while)
Each batch uses its own seeded generator, so a resumed run gives the same numbers as an uninterrupted one.
"""
import json
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ..data.datasets import CacheStore
from ..data.prepare import cache_file
from ..train.common import get_device, seed_everything
from .progressive import Progressive, load_sr_model


def generate(cfg, splits=None):
    seed_everything(cfg["seed"])
    device = get_device(cfg)
    store = CacheStore(cfg)
    geom = store.geom
    model = load_sr_model(cfg, store, device)
    prog = Progressive(cfg, model, store, device)
    grids = [g for g in prog.ladder[1:] if geom.is_supervised(g)]
    s = cfg["sampling"]
    os.makedirs(store.gen_dir, exist_ok=True)
    dtype = np.dtype(s["dtype"])
    for split in splits or s["splits"]:
        cap = (s.get("max_per_split") or {}).get(split)
        n = min(store.n(split), cap or store.n(split))
        prog_path = os.path.join(store.gen_dir, f"{split}_progress.json")
        done = json.load(open(prog_path))["done"] if os.path.exists(prog_path) else 0
        mode = "r+" if done else "w+"
        if done >= n:
            print(f"{split}: already generated ({n} events)")
            continue
        outs = {g: open_memmap(store.sr_path(split, g), mode, dtype, (n, geom.P, *g)) for g in grids}
        mem = open_memmap(os.path.join(store.gen_dir, f"{split}_stage1_members.npy"), mode, np.float32,
                          (n, s["ensemble"], geom.P))
        lr = np.load(cache_file(cfg, split, "lr"), mmap_mode="r")
        bs = s["batch_size"]
        print(f"{split}: generating {n} events on grids {grids} (from {done})")
        for a in range(done, n, bs):
            b = min(n, a + bs)
            gen = torch.Generator(device=device).manual_seed(int(cfg["seed"]) * 100003 + a)
            x = torch.from_numpy(np.asarray(lr[a:b], dtype=np.float32))
            dense, info, _ = prog.run(x, last_grid=geom.native, generator=gen)
            for g in grids:
                outs[g][a:b] = dense[g].cpu().numpy().astype(dtype)
            mem[a:b] = info[0]["members_sum"].numpy()
            for o in list(outs.values()) + [mem]:
                o.flush()
            with open(prog_path, "w") as f:
                json.dump({"done": b, "n": n}, f)
            print(f"\r  {b}/{n}", end="")
        print()
