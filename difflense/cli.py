"""python -m difflense <stage> [<stage> ...] -c configs/qg.yaml [--set key=value ...] [--force]

stages (in order; `all` runs every one):
  download      fetch the raw data (CERNBox for qg, Zenodo for calo); skips complete files
  prepare       raw data -> cache (HR truth, the noisy LR measurement, targets)
  train_sr      flow-matching graph SR model on every truth-checked transition
  generate      SR on whole splits up to the native grid (inputs of the taggers and of eval_sr)
  eval_sr       SR vs truth on every truth-checked grid + paper figures 6 and 7
  extrapolate   native grid -> 2048 x 2048 with physics consistency checks
  train_tagger  ResNet-18 on hr / lr / lr_up / sr, several seeds
  report        one markdown page with every table
"""
import argparse
import json
import os

from .config import load_config, save_config

STAGES = ["download", "prepare", "train_sr", "generate", "eval_sr", "extrapolate", "train_tagger", "report"]


def report(cfg):
    out = cfg["paths"]["out"]
    lines = [f"# {cfg['dataset']} / {cfg['run_name']}", ""]

    def load(*p):
        f = os.path.join(out, *p)
        return json.load(open(f)) if os.path.exists(f) else None

    sr = load("evaluation", "sr", "metrics.json")
    if sr:
        lines += ["## Super-resolution vs truth", "", f"{sr['n_events']} test events.", "",
                  "### Event energy residual (E_reco - E_true) / E_true", "",
                  "| input | median bias | resolution (IQR/1.349) |", "|---|---|---|"]
        for m, v in sr["energy"].items():
            lines.append(f"| {m} | {v['total']['bias_median']:+.4f} | {v['total']['resolution_iqr']:.4f} |")
        if sr.get("ensemble_calibration"):
            c = sr["ensemble_calibration"]
            lines += ["", f"Ensemble calibration (pull of event energy): median {c['pull_median']:+.2f}, "
                          f"width {c['pull_width_iqr']:.2f} (1 = calibrated), "
                          f"{100 * c['frac_within_1sigma']:.0f}% within 1σ."]
        for g, v in sr["grids_eval"].items():
            lines += ["", f"### Grid {g}", "", "| method | hit F1 | E-weighted log error | cell spectrum W1 | EMD (median) |",
                      "|---|---|---|---|---|"]
            for m, c in v["cells"].items():
                lines.append(f"| {m} | {c['hit_f1']:.3f} | {c['energy_weighted_log_error']:.3f} | "
                             f"{c['cell_spectrum_w1_log10']:.3f} | {v['emd'][m]['median']:.4f} |")
            lines += ["", "| observable | method | W1/σ | KS | event bias | event resolution |", "|---|---|---|---|---|---|"]
            for o, ms in v["observables"].items():
                for m, c in ms.items():
                    lines.append(f"| {o} | {m} | {c['w1_over_sigma']:.3f} | {c['ks']:.3f} | {c['event_bias']:+.3f} | "
                                 f"{c['event_resolution']:.3f} |")
        if sr.get("ensemble_study"):
            e = sr["ensemble_study"]
            lines += ["", "Energy resolution vs ensemble size: " +
                      ", ".join(f"{s}: {w:.4f}" for s, w in zip(e["sizes"], e["resolution"]))]
    ex = load("evaluation", "extrapolation", "metrics.json")
    if ex:
        lines += ["", "## Beyond the native grid (no truth: consistency checks)", "",
                  f"{ex['n_events']} events. Energy closure: max relative change {ex['closure_max_rel']:.2e}.", "",
                  "| grid | nodes / event | seconds / event |", "|---|---|---|"]
        for g, s in ex["stages"].items():
            lines.append(f"| {g} | {s['nodes']:.0f} | {s['seconds']:.2f} |")
        lines += ["", "Median per-event drift of grid-independent observables relative to the native-grid SR:", "",
                  "| observable | " + " | ".join(next(iter(ex["drift"].values())).keys()) + " |",
                  "|---|" + "---|" * len(next(iter(ex["drift"].values())))]
        for o, d in ex["drift"].items():
            lines.append(f"| {o} | " + " | ".join(f"{v['vs_native_sr']['event_bias']:+.4f}" for v in d.values()) + " |")
    tg = load("taggers", "summary.json")
    if tg:
        lines += ["", "## Downstream ResNet-18", "", "| input | " + " | ".join(next(iter(tg.values())).keys()) + " |",
                  "|---|" + "---|" * len(next(iter(tg.values())))]
        for k, v in tg.items():
            lines.append(f"| {k} | " + " | ".join(f"{m['mean']:.4f} ± {m['std_over_seeds']:.4f}" for m in v.values())
                         + " |")
    path = os.path.join(out, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report: {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="difflense", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true", help="prepare: rebuild the cache")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    os.makedirs(cfg["paths"]["out"], exist_ok=True)
    save_config(cfg, os.path.join(cfg["paths"]["out"], "config.json"))
    stages = STAGES if "all" in a.stages else [s for s in STAGES if s in a.stages]
    for st in stages:
        print(f"\n===== {st} =====")
        if st == "download":
            from .data.download import download
            download(cfg)
        elif st == "prepare":
            from .data.prepare import prepare
            prepare(cfg, force=a.force)
        elif st == "train_sr":
            from .train.train_sr import train_sr
            train_sr(cfg)
        elif st == "generate":
            from .sr.generate import generate
            generate(cfg)
        elif st == "eval_sr":
            from .eval.evaluate_sr import evaluate_sr
            evaluate_sr(cfg)
        elif st == "extrapolate":
            from .eval.extrapolate import extrapolate
            extrapolate(cfg)
        elif st == "train_tagger":
            from .train.train_tagger import train_taggers
            train_taggers(cfg)
        elif st == "report":
            report(cfg)


if __name__ == "__main__":
    main()
