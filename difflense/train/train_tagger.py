"""ResNet-18 downstream taggers: does super-resolution help the analysis?

qg   : quark vs gluon classification -> ROC AUC, background rejection 1/eps_B at eps_S = 30/50/70 %
calo : incident-energy regression    -> response bias and resolution (overall and per energy bin)

One network per input kind (hr | lr | lr_up | sr | sr@UxV) and seed. Every input uses the same train / val /
test events (the smallest available set), so differences come from the input alone. The HR-trained network
is also applied unchanged to SR and to the uniformly upsampled LR ("transfer"): a drop there means SR is not
yet distributed like real HR images, even if a dedicated network can still use it.
Best practices: AdamW, warmup + cosine, bf16 autocast on CUDA, physics-preserving augmentations, early
stopping on validation AUC / loss with the best checkpoint restored, several seeds, bootstrap errors.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from ..config import save_config
from ..data.datasets import CacheStore, TaggerDataset, regression_target
from ..eval import plots
from ..eval.metrics import classification_metrics, regression_metrics
from ..models.resnet import ResNet18Tagger
from .common import (EarlyStopping, RunLogger, autocast, count_params, get_device, param_groups,
                     save_checkpoint, seed_everything, warmup_cosine, worker_init)


def _available(store, kind, split):
    if not store.has(split, kind):
        return 0
    return len(store.array(split, kind))


@torch.no_grad()
def predict(model, dl, device, ac):
    model.eval()
    out = []
    for x, g, _ in dl:
        with ac():
            out.append(model(x.to(device), g.to(device)).float().cpu())
    return torch.cat(out).numpy()


def _loss(task, pred, y, smoothing):
    if task == "classification":
        y = y * (1 - smoothing) + 0.5 * smoothing
        return F.binary_cross_entropy_with_logits(pred, y)
    return F.smooth_l1_loss(pred, y, beta=0.5)


def train_one(cfg, store, kind, seed, n, targets, task, device, run_dir):
    tc = cfg["tagger"]
    seed_everything(seed)
    nw = cfg["num_workers"]
    mk = lambda split, aug, shuffle: DataLoader(
        TaggerDataset(store, split, kind, targets[split][:n[split]], augment=aug), batch_size=tc["batch_size"],
        shuffle=shuffle, num_workers=nw, worker_init_fn=worker_init, pin_memory=device.type == "cuda",
        persistent_workers=nw > 0, drop_last=shuffle)
    dl_tr, dl_va = mk("train", tc["augment"], True), mk("val", False, False)
    model = ResNet18Tagger(store.geom.P, store.geom.P, 1, tc["dropout"]).to(device)
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"])
    sched = warmup_cosine(opt, tc["epochs"] * len(dl_tr), tc["warmup_frac"])
    ac, scaler = autocast(cfg, device)
    mode = "max" if task == "classification" else "min"
    stopper = EarlyStopping(tc["patience"], 1e-4, mode)
    log = RunLogger(run_dir, cfg, os.path.basename(run_dir))
    print(f"  {kind} seed {seed}: {count_params(model) / 1e6:.1f}M params, {n['train']} train events")
    for epoch in range(tc["epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for x, g, y in dl_tr:
            x, g, y = x.to(device), g.to(device), y.to(device)
            with ac():
                loss = _loss(task, model(x, g).float(), y, tc["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        pv = predict(model, dl_va, device, ac)
        yv = targets["val"][:n["val"]]
        val_loss = float(_loss(task, torch.from_numpy(pv), torch.from_numpy(yv.astype(np.float32)), 0.0))
        row = {"epoch": epoch, "train_loss": tot / max(nb, 1), "val_loss": val_loss}
        if task == "classification":
            row["val_auc"] = float(roc_auc_score(yv, pv))
        best = stopper.step(row.get("val_auc", val_loss))
        row["best"] = int(best)
        log.log(row)
        if best:
            save_checkpoint(os.path.join(run_dir, "best.pt"), model=model.state_dict(), epoch=epoch)
        if stopper.stop:
            break
    log.close()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device)["model"])
    plots.training_curves(os.path.join(run_dir, "history.csv"), run_dir, "training_curves",
                          ["train_loss", "val_loss", "val_auc"])
    return model, ac


def score(cfg, store, model, ac, kind, n_test, targets_raw, task, stats, device):
    dl = DataLoader(TaggerDataset(store, "test", kind, np.zeros(n_test)), batch_size=cfg["tagger"]["batch_size"],
                    num_workers=cfg["num_workers"])
    p = predict(model, dl, device, ac)
    y = targets_raw[:n_test]
    if task == "classification":
        return classification_metrics(y, p, cfg["tagger"]["n_boot"]), p
    e_pred = np.exp(p * stats[1] + stats[0])
    return regression_metrics(y, e_pred, n_boot=cfg["tagger"]["n_boot"]), e_pred


def train_taggers(cfg):
    device = get_device(cfg)
    store = CacheStore(cfg)
    tc = cfg["tagger"]
    task = "classification" if store.geom.kind == "jet" else "regression"
    root = os.path.join(cfg["paths"]["out"], "taggers")
    os.makedirs(root, exist_ok=True)
    kinds = [k for k in tc["inputs"] if all(_available(store, k, s) for s in ("train", "val", "test"))]
    skipped = sorted(set(tc["inputs"]) - set(kinds))
    if skipped:
        print(f"skipping inputs without generated data: {skipped} (run the generate stage)")
    n = {s: min(_available(store, k, s) for k in kinds) for s in ("train", "val", "test")}
    if tc["max_train"]:
        n["train"] = min(n["train"], tc["max_train"])
    raw = {s: store.target(s) for s in ("train", "val", "test")}
    if task == "classification":
        targets, stats = {s: (raw[s] == 1).astype(np.float32) for s in raw}, None
    else:
        _, stats = regression_target(raw["train"][:n["train"]])
        targets = {s: regression_target(raw[s], stats)[0].astype(np.float32) for s in raw}
    print(f"taggers ({task}) on {kinds}, events {n}")
    results = {}
    for kind in kinds:
        for seed in tc["seeds"]:
            name = f"{kind.replace('@', '-')}_seed{seed}"
            run_dir = os.path.join(root, name)
            mpath = os.path.join(run_dir, "metrics.json")
            if os.path.exists(mpath):
                results[name] = json.load(open(mpath))
                continue
            os.makedirs(run_dir, exist_ok=True)
            model, ac = train_one(cfg, store, kind, seed, n, targets, task, device, run_dir)
            m, pred = score(cfg, store, model, ac, kind, n["test"], raw["test"], task, stats, device)
            m = {"input": kind, "seed": seed, "evaluated_on": kind, "n": n, **m}
            np.savez(os.path.join(run_dir, "test_predictions.npz"), pred=pred, target=raw["test"][:n["test"]])
            if kind == "hr":  # transfer: the HR network applied to other inputs on the HR grid
                m["transfer"] = {}
                for t in tc["transfer"]:
                    if t in kinds and store.kind_shape(t) == store.geom.native:
                        m["transfer"][t] = score(cfg, store, model, ac, t, n["test"], raw["test"], task, stats,
                                                 device)[0]
            save_config(m, mpath)
            results[name] = m
    summary = summarise(results, kinds, task)
    save_config(summary, os.path.join(root, "summary.json"))
    if task == "classification":
        curves_ = {}
        for kind in kinds:
            f = os.path.join(root, f"{kind.replace('@', '-')}_seed{tc['seeds'][0]}", "test_predictions.npz")
            z = np.load(f)
            fpr, tpr, _ = roc_curve(z["target"] == 1, z["pred"])
            curves_[kind] = (tpr, fpr, roc_auc_score(z["target"] == 1, z["pred"]))
        plots.roc(curves_, os.path.join(root, "figures"), "roc")
    _print(summary, task)
    return summary


def summarise(results, kinds, task):
    keys = ["auc", "rej_at_eff30", "rej_at_eff50", "rej_at_eff70"] if task == "classification" else \
        ["response_bias", "resolution"]
    out = {}
    for kind in kinds:
        rs = [r for r in results.values() if r["input"] == kind]
        out[kind] = {k: {"mean": float(np.mean([r[k] for r in rs])), "std_over_seeds": float(np.std([r[k] for r in rs])),
                         "n_seeds": len(rs)} for k in keys}
        tr = [r["transfer"] for r in rs if r.get("transfer")]
        for t in (tr[0] if tr else {}):
            out[f"hr-tagger-on-{t}"] = {k: {"mean": float(np.mean([x[t][k] for x in tr])),
                                            "std_over_seeds": float(np.std([x[t][k] for x in tr])),
                                            "n_seeds": len(tr)} for k in keys}
    return out


def _print(summary, task):
    print("\ndownstream ResNet-18 (mean ± std over seeds):")
    for kind, v in summary.items():
        print("  " + f"{kind:22s}" + "  ".join(f"{k} {m['mean']:.4f}±{m['std_over_seeds']:.4f}" for k, m in v.items()))
