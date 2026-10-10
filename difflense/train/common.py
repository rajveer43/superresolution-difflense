"""Training utilities shared by the SR model and the taggers."""
import contextlib
import copy
import csv
import math
import os
import random
import time

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def get_device(cfg):
    want = cfg.get("device", "auto")
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(cfg, device):
    """bf16 autocast on CUDA when supported (no loss scaling needed), fp16 + GradScaler otherwise."""
    if not cfg.get("amp", True) or device.type != "cuda":
        return contextlib.nullcontext, None
    if torch.cuda.is_bf16_supported():
        return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None
    return (lambda: torch.autocast("cuda", dtype=torch.float16)), torch.amp.GradScaler("cuda")


class EMA:
    """Exponential moving average of the weights; evaluation and sampling use the EMA model."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def warmup_cosine(optimizer, total_steps, warmup_frac=0.03, final_frac=0.01):
    warm = max(1, int(warmup_frac * total_steps))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total_steps - warm))
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def param_groups(model, weight_decay):
    """No weight decay on biases, norms, embeddings and attention biases."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "emb" in n or n.endswith("bias") or ".plane." in n or ".flag." in n
         else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0, mode="min"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = math.inf if mode == "min" else -math.inf
        self.bad = 0

    def step(self, value):
        """True when `value` is a new best."""
        better = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if better:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return better

    @property
    def stop(self):
        return self.bad >= self.patience

    def state_dict(self):
        return {"best": self.best, "bad": self.bad}

    def load_state_dict(self, s):
        self.best, self.bad = s["best"], s["bad"]


class RunLogger:
    """history.csv (one row per epoch) + optional Weights & Biases."""

    def __init__(self, run_dir, cfg, name):
        self.path = os.path.join(run_dir, "history.csv")
        self.wb = None
        if cfg.get("wandb", {}).get("enabled"):
            import wandb
            self.wb = wandb.init(project=cfg["wandb"]["project"], name=name, config=cfg, reinit=True)
        self.t0 = time.time()

    def log(self, row):
        row = {**row, "elapsed_s": round(time.time() - self.t0, 1)}
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        if self.wb:
            self.wb.log(row)
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    def close(self):
        if self.wb:
            self.wb.finish()


def save_checkpoint(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
