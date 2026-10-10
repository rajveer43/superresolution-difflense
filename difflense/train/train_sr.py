"""Train the graph SR flow-matching model on every truth-checked transition of the ladder at once.

One set of weights serves every stage: inputs are scale-free (energy densities, positions as fractions of
the image, child offsets in parent widths), so the model is pushed to learn how a cell's energy splits
given its neighbourhood, independent of the absolute cell size. That is what the stages beyond the native
grid rely on.

Best practices: AdamW without decay on norms/biases/embeddings, warmup + cosine schedule, gradient
clipping, EMA weights for validation and sampling, bf16 autocast on CUDA, early stopping on the
(seeded, so comparable across epochs) EMA validation loss, a physics validation metric every few epochs,
full resume from last.pt.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import save_config
from ..data.datasets import CacheStore, Collate, Featurizer, TransitionDataset, to_device
from ..geometry import train_transitions
from ..models.flow import cfm_loss
from ..models.sr_transformer import build_sr_model
from ..sr.progressive import Progressive
from .common import (EMA, EarlyStopping, RunLogger, autocast, count_params, get_device, param_groups,
                     save_checkpoint, seed_everything, warmup_cosine, worker_init)


def _loader(ds, cfg, bs, shuffle, collate, device):
    nw = cfg["num_workers"]
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=nw,
                      worker_init_fn=worker_init, pin_memory=device.type == "cuda",
                      persistent_workers=nw > 0, drop_last=shuffle)


@torch.no_grad()
def val_loss(model, dl, device, ac, sigma_min, seed):
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        tot, n = 0.0, 0
        for g in dl:
            g = to_device(g, device)
            with ac():
                tot += float(cfm_loss(model, g, sigma_min)) * g["B"]
            n += g["B"]
    return tot / max(n, 1)


@torch.no_grad()
def physics_val(cfg, model, store, device, n):
    """Event energy residual (SR - truth) / truth on the native grid, from a short sampling run."""
    lr = torch.from_numpy(np.asarray(np.load(os.path.join(cfg["paths"]["cache"], "val_lr.npy"),
                                             mmap_mode="r")[:n], dtype=np.float32))
    hr = np.asarray(np.load(os.path.join(cfg["paths"]["cache"], "val_hr.npy"), mmap_mode="r")[:n])
    prog = Progressive(cfg, model, store, device)
    out = []
    for s in range(0, n, 32):
        dense, _, _ = prog.run(lr[s:s + 32], last_grid=store.geom.native, ensemble=2, steps=10,
                               generator=torch.Generator(device=device).manual_seed(s))
        out.append(dense[store.geom.native].sum((1, 2, 3)).cpu().numpy())
    sr = np.concatenate(out)
    true = hr.sum((1, 2, 3))
    lr_tot = lr.numpy().sum((1, 2, 3))
    ok = true > 0
    r_sr, r_lr = sr[ok] / true[ok] - 1, lr_tot[ok] / true[ok] - 1
    iqr = lambda x: float(np.subtract(*np.percentile(x, [75, 25])) / 1.349)
    return {"phys_sr_bias": float(np.median(r_sr)), "phys_sr_res": iqr(r_sr),
            "phys_lr_bias": float(np.median(r_lr)), "phys_lr_res": iqr(r_lr)}


def train_sr(cfg):
    seed_everything(cfg["seed"])
    device = get_device(cfg)
    store = CacheStore(cfg)
    geom = store.geom
    tc = cfg["sr_train"]
    run_dir = os.path.join(cfg["paths"]["out"], "sr_model")
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.json"))
    print("transitions:", [(p, c) for p, c, _ in train_transitions(geom, cfg)])

    feat = Featurizer(geom, store.scales, cfg["sr_model"]["window"])
    col = Collate(feat)
    dl_tr = _loader(TransitionDataset(store, "train", True, tc["max_train"]), cfg, tc["batch_size"], True, col, device)
    dl_va = _loader(TransitionDataset(store, "val", False, tc["max_val"]), cfg, tc["batch_size"], False, col, device)

    model = build_sr_model(cfg, geom, len(feat.offsets)).to(device)
    print(f"SR model: {count_params(model) / 1e6:.2f}M parameters, device {device}")
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"], betas=(0.9, 0.99))
    sched = warmup_cosine(opt, tc["epochs"] * len(dl_tr), tc["warmup_frac"])
    ema = EMA(model, tc["ema_decay"])
    stopper = EarlyStopping(tc["patience"], tc["min_delta"])
    ac, scaler = autocast(cfg, device)
    start = 0
    last = os.path.join(run_dir, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        stopper.load_state_dict(ck["stopper"])
        start = ck["epoch"] + 1
        print(f"resumed from epoch {start}")
        if stopper.stop or start >= tc["epochs"]:
            print("training already finished")
            return
    log = RunLogger(run_dir, cfg, f"{cfg['run_name']}_sr")

    for epoch in range(start, tc["epochs"]):
        model.train()
        tot, n, gn = 0.0, 0, 0.0
        for g in dl_tr:
            g = to_device(g, device)
            with ac():
                loss = cfm_loss(model, g, tc["sigma_min"])
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            gn += float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]))
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            ema.update(model)
            tot += float(loss.detach())
            n += 1
        vl = val_loss(ema.model, dl_va, device, ac, tc["sigma_min"], cfg["seed"])
        row = {"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": vl, "grad_norm": gn / max(n, 1),
               "lr": sched.get_last_lr()[0]}
        if tc["physics_val_every"] and (epoch + 1) % tc["physics_val_every"] == 0:
            row.update(physics_val(cfg, ema.model, store, device,
                                   min(tc["physics_val_events"], store.n("val"))))
        best = stopper.step(vl)
        row["best"] = int(best)
        log.log(row)
        state = {"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "stopper": stopper.state_dict(), "epoch": epoch, "val_loss": vl}
        save_checkpoint(last, **state)
        if best:
            save_checkpoint(os.path.join(run_dir, "best.pt"), ema=ema.model.state_dict(), epoch=epoch, val_loss=vl)
        if stopper.stop:
            print(f"early stop at epoch {epoch} (best val loss {stopper.best:.5f})")
            break
    log.close()
