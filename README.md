# DiffLense: progressive graph super-resolution for calorimeter data

Super-resolution of calorimeter images with conditional flow matching on graphs, following
*Denoising Graph Super-Resolution towards Improved Collider Event Reconstruction* (Kakati, Dreyer, Gross, 2024),
applied to two datasets:

| Dataset | Content | Native grid | Measured LR | Ladder |
|---|---|---|---|---|
| `qg` | CMS Open Data quark/gluon jet images (ML4Sci): tracks, ECAL, HCAL | 3 x 125 x 125, padded to 128 x 128 | 32 x 32 | 32 → 64 → **128** → 256 → 512 → 1024 → **2048** |
| `calo` | CaloChallenge 2022 Dataset 2, electron showers | 45 layers x 16 angle x 9 radius | 4 x 3 per layer | 4x3 → 8x9 → **16x9** → 32x18 → … → **2048x1152** |

The high-resolution (HR) truth is downsampled to a measured low-resolution (LR) image. That image is super-resolved step by step and compared with the truth using physics metrics. A ResNet-18 trained on the result checks whether it helps a downstream task. PSNR and SSIM are deliberately not used.

## Method

**Measurement model (paper section 2).** The LR image is the HR image sum-pooled: a coarser detector would record exactly this, and total energy is conserved. Gaussian electronic noise is then added per cell, and an ATLAS-like 4-2-0 topological clustering keeps only clustered cells with positive energy. The tracks channel carries no calorimeter noise. Noise levels are relative to the mean non-zero cell energy of each plane, computed from the train split only.

**Graph super-resolution (paper section 3).** Every non-empty cell is a graph node. One stage splits each cell into `fu x fv` children. Each child receives the parent energy as its condition, and a transformer predicts the flow-matching vector field per node:

- Target: `x1 = log1p(E_child / (s·a)) - log1p(E_parent / (K·s·a))`, a correction to the copied parent energy. Here `s` is the plane scale, `a` the cell area relative to a native cell, and `K = fu·fv` the number of children.
- Objective: conditional flow matching with the OT path and σ_min = 1e-4 (Lipman et al.). Sampling uses 25 midpoint steps by default; `dopri5` via torchdiffeq is an option.
- Ensemble sampling (paper section 3.3): the same input is sampled 10 times and averaged in energy space. The ensemble spread is kept for calibration checks.

**Transformer.** It is DiT-style, with adaLN-zero conditioning on time and on whether the parent is a noisy measurement, and QK-normalised attention. Attention is **local**: the (2w+1)² neighbouring cells, the same cell in the other planes, a learned bias per relative offset, and a per-event pooled global context. The paper uses full attention on small graphs. At 2048 x 2048 a graph has 10⁵–10⁶ nodes, so attention has to be local; the paper's outlook points the same way.

**One model for every stage.** All inputs are scale-free: energy densities, positions as fractions of the image, and child offsets in parent widths. The model is trained on every transition that has truth on both sides, including one coarser than the measurement, so it learns how energy splits largely independently of cell size.

**Progressive inference.**
1. Stage 1 starts from the noisy measurement. It denoises as well as splits, and is not tied to the measured energy.
2. Every later stage starts from clean model output. Its children are projected onto their parent's energy, so the energy of stage 1 is conserved exactly up to 2048 x 2048.
3. Beyond the native grid, refinement is adaptive. The most energetic cells are split first, within `refine.max_nodes_per_event`; the rest stay as exact leaves of their own grid.

### What the 2048 x 2048 output can and cannot claim

No truth exists below one ECAL crystal (CMS) or one CaloChallenge voxel. The stages beyond the native grid therefore **cannot be scored against truth**, and any structure they add is learned from coarser scales, not measured. They are checked on what any physical refinement must satisfy:

- energy closure;
- stability of grid-independent observables as the grid gets finer;
- smooth power spectra with no artefacts at the cell pitch;
- zoom panels of the same physical window at every scale.

The `*_extrapolation_check.yaml` configs estimate the error honestly. They hold the last transition that has truth out of training and score it against truth.

## Metrics (no PSNR / SSIM)

| Question | Metric |
|---|---|
| Is energy right? | Event energy residual (E_SR − E_true)/E_true: median bias and IQR/1.349 resolution, total and per plane (paper fig. 6a) |
| Is the physics distribution right? | W1/σ_truth, KS statistic, mean shift for jet mass, pT, girth, ΔR spread, energy correlators C2/C3/D2 (paper fig. 7); shower depth/radial centroids and widths, layer fractions, E_dep/E_inc |
| Is each event right? | Per-event bias and resolution of every observable, Pearson r |
| Are the right cells hit? | Hit precision / recall / F1, energy-weighted log error, cell-energy spectrum W1 |
| Is energy in the right place? | Per-event energy mover's distance (Sinkhorn), in ΔR or mm |
| Is the ensemble honest? | Pull of event energy (width 1 = calibrated); resolution vs ensemble size (paper fig. 6b) |
| Does it help an analysis? | ResNet-18 on HR / LR / LR-upsampled / SR. qg: ROC AUC and 1/ε_B at ε_S = 30/50/70 %. calo: energy-response bias and resolution. Several seeds plus bootstrap errors; the HR tagger is also applied to SR to expose domain shift |

Baselines: the measured LR, and the LR spread uniformly onto the finer grid (no learning, energy-preserving).

## Training practice

- **Optimisation:** AdamW with no decay on norms, biases or embeddings; warmup then cosine schedule; gradient clipping.
- **Stability and speed:** EMA weights for validation and sampling; bf16 autocast on CUDA.
- **Early stopping:** on the EMA validation loss (seeded so epochs are comparable), with the best checkpoint kept and full resume from `last.pt`.
- **Monitoring:** a physics validation metric, the event-energy resolution after the full ladder, is logged every few epochs.
- **Data handling:** fresh noise every epoch on train; a fixed, seeded measurement for val and test; statistics from the train split only.
- **Tagger:** physics-preserving augmentations (η/φ reflections for jets, rotations about the beam axis for showers); early stopping on validation AUC or loss; the same events for every input; several seeds.
- **Long jobs:** resumable, seeded generation and per-epoch `history.csv` (optionally W&B).

## Usage

### Colab (recommended)

Upload [`notebooks/difflense_colab.ipynb`](notebooks/difflense_colab.ipynb) to Colab, or open it from GitHub. Choose a GPU runtime and add a `GITHUB_TOKEN` Colab secret while the repo is private. Then run all cells: first with `SMOKE = True`, then with `SMOKE = False`.
- The notebook downloads the data once to Drive and runs every stage.
- Results go to `MyDrive/GSoC_DiffLense/experiments/<dataset>/<date>_<run-tag>/` (logs, models, metrics, figures, report) plus a zip in `exports/`.
- Every stage resumes after a disconnect.

### Command line

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m difflense download -c configs/qg.yaml     # CERNBox share (3 parquet files, 3.7 GB); calo: Zenodo
# or put the ML4Sci quark/gluon parquet/h5 files in data/raw/qg/,
#       CaloChallenge dataset_2_1.hdf5 / dataset_2_2.hdf5 in data/raw/calo/
#       (or reuse a cache of the older `superres` package: --set data.import_cache=/path/to/cache/qg)

python -m difflense all -c configs/qg.yaml          # download → prepare → train_sr → generate → eval_sr → extrapolate → train_tagger → report
python -m difflense all -c configs/calo.yaml
python -m difflense train_sr generate eval_sr report -c configs/qg_extrapolation_check.yaml

# single stages and overrides
python -m difflense train_sr -c configs/qg.yaml --set sr_train.epochs=50 sr_model.dim=192
python -m difflense extrapolate -c configs/qg.yaml --set refine.events=64

# tests and a tiny synthetic end-to-end run (CPU)
pytest -q tests
python -m difflense all -c configs/smoke_qg.yaml
```

Outputs go to `runs/<dataset>/<run_name>/`: `sr_model/`, `generated/`, `evaluation/sr/`, `evaluation/extrapolation/`, `taggers/` and `report.md`. Every figure is written as png and pdf.

## Layout

```
difflense/
  config.py            defaults + YAML + --set overrides
  geometry.py          planes, grids, physical coordinates, the ladder
  physics/ops.py       sum-pooling, uniform upsampling, noise + topo-clustering
  physics/observables.py  jet / shower observables, energy correlators, power spectrum
  data/prepare.py      streaming raw → cache, train-only statistics
  data/graphs.py       sparse cells, splitting, hashed neighbour search
  data/datasets.py     SR transition dataset + collate, tagger dataset
  models/sr_transformer.py  graph DiT with local attention
  models/flow.py       flow-matching loss, ODE sampler, energy projection
  models/resnet.py     ResNet-18 tagger / regressor
  sr/progressive.py    stage-by-stage inference, adaptive refinement, rendering
  sr/generate.py       resumable generation on whole splits
  train/               SR and tagger training, shared utilities
  eval/                SR evaluation, extrapolation checks, metrics, figures
configs/               qg, calo, held-out-scale checks, synthetic smoke runs
tests/                 unit tests of the physics invariants
```

## Status

Unit tests pass. Every stage has run end to end on synthetic jet and shower data (`configs/smoke_*.yaml`). The download stage has been checked against the CERNBox share and the parquet schema. Full training on the real datasets is next.

## Reference

N. Kakati, E. Dreyer, E. Gross, *Denoising Graph Super-Resolution towards Improved Collider Event Reconstruction*, 2024. Original code: https://github.com/nilotpal09/SuperResolutionHEP
