"""Configuration: generic defaults, merged with a dataset YAML and with --set overrides.

Everything is a plain nested dict so it serialises to JSON next to every result.
Dotted overrides: --set sr_train.lr=5e-4 ladder.stages='[[2,2],[2,2]]'
"""
import copy
import json
import os

import yaml

BASE = {
    "run_name": "default",
    "seed": 42,
    "device": "auto",            # auto | cuda | mps | cpu
    "amp": True,                 # bf16/fp16 autocast on CUDA only
    "num_workers": 2,
    "paths": {
        "raw": "data/raw/{dataset}",
        "cache": "data/cache/{dataset}",
        "out": "runs/{dataset}/{run_name}",
        "generated": None,       # SR outputs of the generate stage; default {out}/generated
    },
    "data": {
        # kind: jet (CMS quark/gluon images) | shower (CaloChallenge DS2) ; set by the dataset YAML
        "kind": "jet",
        "source": "qg",          # qg | calo | synthetic
        "max_events": None,
        "split": [0.8, 0.1, 0.1],
        "import_cache": None,    # path to a cache written by the older `superres` package (reused as is)
    },
    # The measured LR image is HR sum-pooled by lr_factor (+ noise). SR then runs stage by stage, each stage
    # splitting every cell by its factor. Stages whose output grid is <= the native grid are checked against
    # truth; later ones have no truth and are checked by physics consistency only.
    "ladder": {
        "lr_factor": [4, 4],
        "stages": [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        # transitions the SR model is trained on, as (parent grid, child grid); null = every transition
        # inside the native grid, including ones coarser than the measured LR (more scales -> better
        # self-similarity for the extrapolated stages)
        "train_transitions": None,
    },
    # Electronic noise and topological clustering on the measured LR cells (paper section 2).
    "noise": {
        "enabled": True,
        "rel_sigma": 0.1,        # per plane (list) or one value; sigma = rel_sigma * mean non-zero HR cell energy
        "topo": [4.0, 2.0, 0.0], # seed / grow / boundary thresholds in units of sigma (ATLAS-like 4-2-0)
        "cond_aug": 0.1,         # log-normal jitter of clean parents during training (cascade exposure bias)
    },
    "sr_model": {
        "dim": 256,
        "depth": 4,
        "heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "window": 2,             # in-plane neighbourhood radius (cells) of the local attention
        "attn_chunk": 65536,     # nodes per attention chunk at inference (memory bound)
    },
    "sr_train": {
        "batch_size": 25,        # events (paper table 2)
        "lr": 1e-3,
        "weight_decay": 0.01,
        "warmup_frac": 0.03,
        "epochs": 100,
        "patience": 10,          # early stopping on EMA validation loss
        "min_delta": 1e-4,
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        "max_train": None,
        "max_val": 3000,
        "physics_val_every": 5,  # epochs; short sampling run on val to log the energy resolution
        "physics_val_events": 200,
        "sigma_min": 1e-4,
    },
    "sampling": {
        "steps": 25,
        "solver": "midpoint",    # euler | midpoint | dopri5 (torchdiffeq)
        "ensemble": 10,          # paper section 3.3
        "batch_size": 64,
        "splits": ["train", "val", "test"],
        "max_per_split": {"train": 20000, "val": None, "test": None},
        "dtype": "float32",
    },
    # Stages beyond the native grid: adaptive refinement so 2048 x 2048 stays tractable.
    "refine": {
        "events": 32,            # test events pushed to the final grid
        "ensemble": 4,
        "steps": 10,
        "max_nodes_per_event": 400000,
        "min_density": 0.0,      # split a cell only if E / (scale * area) exceeds this (0 = all non-zero)
        "save_events": 8,        # sparse multi-level outputs written to disk
    },
    "tagger": {
        "arch": "resnet18",
        "inputs": ["hr", "lr", "lr_up", "sr"],
        "transfer": ["sr", "lr_up"],  # also score the HR-trained tagger on these (same grid as HR)
        "seeds": [42, 43, 44],
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "epochs": 30,
        "patience": 6,
        "warmup_frac": 0.03,
        "dropout": 0.2,
        "label_smoothing": 0.0,
        "augment": True,
        "max_train": None,
        "n_boot": 200,
    },
    "eval": {
        "max_events": 5000,
        "ecf_top_k": 32,
        "emd_top_k": 64,
        "ensemble_study": [1, 2, 5, 10, 20],
        "ensemble_study_events": 300,
        "n_examples": 4,
        "hist_bins": 50,
    },
    "wandb": {"enabled": False, "project": "difflense"},
}


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_value(s):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(BASE)
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        k, _, v = item.partition("=")
        set_dotted(cfg, k.strip(), parse_value(v))
    if "dataset" not in cfg:
        raise ValueError("config needs a top-level 'dataset' name")
    for k, v in cfg["paths"].items():
        if v is None:
            continue
        cfg["paths"][k] = v.format(dataset=cfg["dataset"], run_name=cfg["run_name"])
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
