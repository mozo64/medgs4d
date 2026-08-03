# MedGS4D Technical Guide

MedGS4D trains a phase-conditioned deformation model on top of a frozen canonical MedGS reconstruction. The standard workflow is:

1. discover and extract one 4D-CT patient,
2. inspect and prepare a study as reusable HU volumes,
3. train and evaluate a static canonical MedGS model,
4. select an exact canonical checkpoint,
5. train the dynamic deformation MLP,
6. compare dynamic checkpoints and inspect reconstruction errors,
7. diagnose the learned deformation field,
8. browse completed runs from a notebook.

The upstream MedGS repository remains separate and is supplied with `--medgs-repo`.

## 1. Repository structure

```text
medgs4d/
├── medgs4d/
│   ├── canonical.py
│   ├── config.py
│   ├── data.py
│   ├── deformation.py
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── reporting.py
│   ├── results.py
│   ├── runs.py
│   ├── splits.py
│   ├── training.py
│   └── visualization.py
├── scripts/
│   ├── data.py
│   ├── train_canonical.py
│   ├── evaluate_canonical.py
│   ├── train_medgs4d.py
│   ├── evaluate_medgs4d.py
│   ├── evaluate_checkpoints.py
│   ├── diagnose_deformation.py
│   └── visualize_medgs4d.py
├── notebooks/
│   └── results_browser.ipynb
├── tests/
└── pyproject.toml
```

## 2. Environment

Example WORF setup:

```bash
cd /home/jovyan/shared/mtm_medgs_stack/repo/medgs4d

PYTHON=/home/jovyan/shared/mtm_medgs_stack/envs/medgs-worf/bin/python
MEDGS_REPO=/home/jovyan/shared/mtm_medgs_stack/repo/MedGS

$PYTHON -m pip install -e .

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`CUDA_VISIBLE_DEVICES=1` exposes the second physical GPU as logical `cuda:0` inside PyTorch. The CLI still uses `--device cuda`.

Basic checks:

```bash
$PYTHON -m compileall -q medgs4d scripts
$PYTHON -m pytest -q -m "not worf"
```

For the complete current CLI:

```bash
$PYTHON scripts/<script>.py --help
```

## 3. End-to-end example

### 3.1 List available patients

```bash
ARCHIVES_DIR=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/raw/series_zips

$PYTHON scripts/data.py list-patients \
  --archives-dir "$ARCHIVES_DIR"
```

The command can also inspect already extracted patients:

```bash
$PYTHON scripts/data.py list-patients \
  --dicom-dir /home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/raw/dicom_by_series
```

### 3.2 Extract one patient

```bash
DICOM_DIR=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/raw/dicom_by_series
PATIENT_ID=113_HM10395

$PYTHON scripts/data.py extract \
  --archives-dir "$ARCHIVES_DIR" \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --workers 4
```

Optional persistent extraction log:

```bash
$PYTHON scripts/data.py extract \
  --archives-dir "$ARCHIVES_DIR" \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --workers 4 \
  --log extraction_patient113.csv
```

Output layout:

```text
<DICOM_DIR>/
└── 113_HM10395/
    └── <SeriesInstanceUID>/
        ├── DICOM files
        └── .unpack_complete.json
```

`.unpack_complete.json` records the source archive and completion metadata. It allows repeated extraction calls to skip unchanged, complete series.

### 3.3 List and inspect studies

```bash
$PYTHON scripts/data.py list-studies \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID"
```

Select a study:

```bash
STUDY_UID=1.3.6.1.4.1.14519.5.2.1.6834.5010.717414683086426123380260040917
```

Inspect respiratory phases and geometry:

```bash
$PYTHON scripts/data.py inspect \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID"
```

Confirm that the selected respiratory series have compatible dimensions, slice coordinates, and spacing before preparation.

### 3.4 Prepare reusable 4D-CT volumes

```bash
PREPARED_ROOT=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/prepared
STUDY_NAME=patient113_19991117

$PYTHON scripts/data.py prepare \
  --dicom-dir "$DICOM_DIR" \
  --prepared-root "$PREPARED_ROOT" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID" \
  --study-name "$STUDY_NAME"
```

Prepared study:

```text
<PREPARED_ROOT>/patient113_19991117/
├── manifest.json
├── phase_summary.csv
├── phase_slice_manifest.csv
└── volumes/
    ├── raw/
    │   ├── phase_00.npy
    │   ├── phase_10.npy
    │   └── ...
    └── denoised/
        ├── phase_00.npy
        ├── phase_10.npy
        └── ...
```

Artifacts:

| File | Meaning |
|---|---|
| `manifest.json` | Prepared-study contract: patient and study identifiers, available phases, volume shape, HU window, denoising sigma, and artifact paths. |
| `phase_summary.csv` | One row per respiratory phase: dimensions, slice count, coordinate range, median spacing, and volume paths. |
| `phase_slice_manifest.csv` | One row per phase and slice: source DICOM path, slice index, coordinate, geometry metadata, and prepared volume paths. |
| `volumes/raw/phase_*.npy` | Reconstructed `float32` CT volumes in Hounsfield units. |
| `volumes/denoised/phase_*.npy` | The same HU volumes after the configured 3-D Gaussian filter. |

Default preparation parameters are:

```text
HU window:       [-1000, 400]
Denoise sigma:   [0.20, 0.40, 0.40]
```

The HU window is used for image conversion. It does not replace the HU values stored in the NumPy volumes.

```bash
STUDY_DIR="$PREPARED_ROOT/$STUDY_NAME"

cat "$STUDY_DIR/manifest.json"
head "$STUDY_DIR/phase_summary.csv"
head "$STUDY_DIR/phase_slice_manifest.csv"
```

### 3.5 Train the canonical MedGS model

```bash
CANONICAL_ROOT=/home/jovyan/shared/mtm_medgs_stack/results/canonical
CANONICAL_RUN_NAME=phase00_smoke_iter1000

$PYTHON scripts/train_canonical.py \
  --data-dir "$STUDY_DIR" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$CANONICAL_ROOT" \
  --run-name "$CANONICAL_RUN_NAME" \
  --canonical-phase 0 \
  --representation raw \
  --iterations 1000 \
  --poly-degree 2 \
  --batch-size 3 \
  --camera mirror \
  --log-every 100 \
  --force
```

The script performs four operations:

1. converts the selected canonical HU volume to the upstream MedGS image layout,
2. launches upstream MedGS training,
3. saves the canonical checkpoint,
4. evaluates every canonical slice and generates CSV, JSON, and PDF artifacts.

Canonical run layout:

```text
<CANONICAL_ROOT>/patient113_19991117/phase00_smoke_iter1000/
├── config.json
├── canonical_run.json
├── frame_manifest.csv
├── canonical_training_history.csv
├── canonical_training_history.pdf
├── dataset/
│   ├── original/
│   └── mirror/
├── model/
│   ├── cfg_args
│   ├── chkpnt1000.pth
│   ├── point_cloud/
│   └── other upstream MedGS outputs
├── tools/
│   └── train_with_history.py
└── evaluation/
    ├── per_slice.csv
    ├── overall.json
    ├── canonical_metrics.pdf
    ├── history.csv
    └── history.pdf
```

Important artifacts:

| File | Meaning |
|---|---|
| `config.json` | Canonical run configuration: phase, representation, target iterations, polynomial degree, batch size, camera mode, and seed. |
| `canonical_run.json` | Resolved paths and provenance, including the selected study, upstream MedGS repository, final checkpoint, and resume metadata. |
| `frame_manifest.csv` | Mapping from canonical slice indices to source volume rows and generated PNG files. |
| `dataset/original/*.png` | Windowed canonical slices used by MedGS. |
| `dataset/mirror/*.png` | Mirrored images required by the selected upstream camera pipeline. |
| `model/chkpntN.pth` | Restartable upstream MedGS checkpoint. The payload also stores its true iteration. |
| `canonical_training_history.csv` | Minibatch training metrics recorded every `--log-every` iterations. |
| `canonical_training_history.pdf` | Training metrics versus iteration. |
| `evaluation/per_slice.csv` | Full canonical evaluation for every slice: `L1`, `PSNR`, and `SSIM`. |
| `evaluation/overall.json` | Mean, standard deviation, minimum, and maximum of the full-slice metrics. |
| `evaluation/canonical_metrics.pdf` | Full-slice `PSNR`, `SSIM`, and `L1` versus slice index for the current checkpoint. |
| `evaluation/history.csv` | One full-slice evaluation row per completed canonical checkpoint. |
| `evaluation/history.pdf` | Mean full-slice metrics versus canonical checkpoint iteration. |

`canonical_training_history.csv` contains:

```text
Iteration, TotalLoss, L1, InterpolationL1, SSIM, SSIMLoss,
SigmaLoss, PSNR, EMALoss, EMAPSNR, GaussianCount,
IterationTimeMs, ElapsedSeconds
```

#### Resume canonical training

Only the target iteration may change. Resume always uses the newest available `chkpnt*.pth` in the canonical run.

Example: extend the same run from 1000 to 5000 iterations:

```bash
$PYTHON scripts/train_canonical.py \
  --data-dir "$STUDY_DIR" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$CANONICAL_ROOT" \
  --run-name "$CANONICAL_RUN_NAME" \
  --canonical-phase 0 \
  --representation raw \
  --iterations 5000 \
  --poly-degree 2 \
  --batch-size 3 \
  --camera mirror \
  --log-every 100 \
  --resume
```

A later extension can preserve another checkpoint:

```bash
$PYTHON scripts/train_canonical.py \
  --data-dir "$STUDY_DIR" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$CANONICAL_ROOT" \
  --run-name "$CANONICAL_RUN_NAME" \
  --canonical-phase 0 \
  --representation raw \
  --iterations 6000 \
  --poly-degree 2 \
  --batch-size 3 \
  --camera mirror \
  --log-every 100 \
  --resume
```

Both `chkpnt5000.pth` and `chkpnt6000.pth` can then remain available for comparison or downstream use.

#### Re-evaluate a canonical run

```bash
CANONICAL_RUN="$CANONICAL_ROOT/$STUDY_NAME/$CANONICAL_RUN_NAME"

$PYTHON scripts/evaluate_canonical.py \
  --run-dir "$CANONICAL_RUN" \
  --medgs-repo "$MEDGS_REPO" \
  --target-representation raw \
  --device cuda
```

### 3.6 Train MedGS4D from an exact canonical checkpoint

The canonical model is frozen. The dynamic optimizer updates only the deformation MLP, which predicts phase-conditioned residuals for canonical Gaussian `x`, `z`, and `m`.

Select the canonical checkpoint explicitly:

```bash
CANONICAL_RUN="$CANONICAL_ROOT/$STUDY_NAME/$CANONICAL_RUN_NAME"
CANONICAL_CHECKPOINT="$CANONICAL_RUN/model/chkpnt5000.pth"

DYNAMIC_ROOT=/home/jovyan/shared/mtm_medgs_stack/results/medgs4d
DYNAMIC_RUN_NAME=phase00_canon5000_full_smoke_iter1000
```

Start a smoke run:

```bash
$PYTHON scripts/train_medgs4d.py \
  --data-dir "$STUDY_DIR" \
  --canonical-model "$CANONICAL_RUN" \
  --canonical-checkpoint "$CANONICAL_CHECKPOINT" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$DYNAMIC_ROOT" \
  --run-name "$DYNAMIC_RUN_NAME" \
  --canonical-phase 0 \
  --target-representation raw \
  --split-mode full \
  --iterations 1000 \
  --learning-rate 5e-4 \
  --checkpoint-every 500 \
  --log-every 25 \
  --validate-every 0 \
  --validation-samples 0 \
  --phase-jitter-std 0 \
  --seed 42 \
  --device cuda
```

The loader validates the checkpoint in two ways:

1. it parses the iteration from `chkpnt5000.pth`,
2. it reads the saved iteration from the checkpoint payload.

The two values must match. The selected path and iteration are then persisted in the dynamic `config.json`, so later evaluation and notebook visualization use the same canonical state.

Dynamic run layout:

```text
<DYNAMIC_ROOT>/patient113_19991117/phase00_canon5000_full_smoke_iter1000/
├── config.json
├── split_manifest.csv
├── sampling_plan.csv
├── deformation_normalization.json
├── smoothness_gaussian_indices.npy
├── training_history.csv
├── validation_history.csv          # only when validation is active
├── training_summary.csv
├── training_history.pdf
├── completion.json
├── report.pdf
├── report_metrics.csv
├── checkpoints/
│   ├── deformation_iter_000000.pth
│   ├── deformation_iter_000500.pth
│   ├── deformation_iter_001000.pth
│   └── deformation_latest.pth
├── evaluation/
│   ├── per_slice.csv
│   ├── per_phase.csv
│   ├── overall.json
│   ├── metrics.pdf
│   ├── history.csv
│   └── history.pdf
└── visualizations/
```

Artifacts created before or during training:

| File | Meaning |
|---|---|
| `config.json` | Complete reproducible configuration, including the exact canonical checkpoint path and iteration. |
| `split_manifest.csv` | Every phase-slice pair assigned to `canonical`, `train`, or `validation`. |
| `sampling_plan.csv` | Deterministic sample selected for every training iteration, including phase, slice, neighboring phase, and optional temporal jitter. |
| `deformation_normalization.json` | Canonical coordinate statistics and deformation-model dimensions. |
| `smoothness_gaussian_indices.npy` | Reproducible Gaussian subset used by temporal smoothness regularization. |
| `training_history.csv` | Per-iteration optimization metrics, written atomically every `--log-every` iterations. |
| `validation_history.csv` | Metrics on a fixed compact validation subset, written every `--validate-every` iterations. It is absent when validation is disabled or the split has no validation rows. |
| `checkpoints/deformation_iter_*.pth` | MLP, optimizer, iteration, run identity, canonical checkpoint identity, and RNG states. |
| `checkpoints/deformation_latest.pth` | Copy of the most recently selected branch checkpoint for default resume. |

`training_history.csv` includes:

```text
Iteration, PhasePercent, SliceIndex, NeighborPhasePercent,
PhaseJitterStd, PhaseJitter, LearningRate, ElapsedSeconds,
PeakGpuMemoryGB, TotalLoss, ImageLoss, L1, SSIM, PSNR,
DeformationMagnitudeLoss, TemporalSmoothnessLoss, GradientNorm
```

Artifacts created after training:

| File | Meaning |
|---|---|
| `training_summary.csv` | Final-window means overall and per respiratory phase. |
| `training_history.pdf` | Rolling training curves for reconstruction metrics, losses, gradient norm, and GPU memory. |
| `completion.json` | Completion status, final iteration, parameter count, sample counts, canonical checkpoint identity, and elapsed time. |
| `evaluation/per_slice.csv` | Baseline and dynamic metrics for each evaluated phase-slice pair. |
| `evaluation/per_phase.csv` | Mean baseline, dynamic, and improvement metrics per phase and split. |
| `evaluation/overall.json` | Aggregated metrics for all samples, all noncanonical samples, and individual splits. |
| `evaluation/metrics.pdf` | Final baseline-versus-dynamic metrics and improvements across respiratory phases. |
| `evaluation/history.csv` | One full evaluation point per completed dynamic checkpoint stage. |
| `evaluation/history.pdf` | Full evaluation metrics versus dynamic checkpoint iteration. |
| `report_metrics.csv` | Compact baseline/dynamic/improvement table used by the report. |
| `report.pdf` | Concise run summary with headline reconstruction metrics, per-phase PSNR gain, and training curves. |

Key metric columns in `evaluation/per_slice.csv`:

```text
BaselineL1, DynamicL1, L1Reduction,
BaselinePSNR, DynamicPSNR, PSNRGain,
BaselineSSIM, DynamicSSIM, SSIMGain
```

Interpretation:

- lower `L1` is better,
- higher `PSNR` is better,
- higher `SSIM` is better,
- positive `L1Reduction`, `PSNRGain`, and `SSIMGain` mean the dynamic model improved over the frozen canonical baseline.

#### Split modes

`full`:

- canonical phase → `canonical`,
- every other phase → `train`,
- no validation phase.

`phase-holdout`:

- canonical phase → `canonical`,
- phases listed in `--validation-phases` → `validation`,
- remaining phases → `train`.

Example:

```bash
--split-mode phase-holdout \
--validation-phases 10 30 50 70 90
```

### 3.7 Resume or branch a dynamic run

Set the run directory:

```bash
RUN_DIR="$DYNAMIC_ROOT/$STUDY_NAME/$DYNAMIC_RUN_NAME"
```

Extend the same run from 1000 to 2000 iterations:

```bash
$PYTHON scripts/train_medgs4d.py \
  --data-dir "$STUDY_DIR" \
  --canonical-model "$CANONICAL_RUN" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$DYNAMIC_ROOT" \
  --run-name "$DYNAMIC_RUN_NAME" \
  --canonical-phase 0 \
  --target-representation raw \
  --split-mode full \
  --iterations 2000 \
  --learning-rate 5e-4 \
  --checkpoint-every 500 \
  --log-every 25 \
  --validate-every 0 \
  --validation-samples 0 \
  --phase-jitter-std 0 \
  --seed 42 \
  --device cuda \
  --resume
```

The saved canonical checkpoint is reused automatically. The sampling plan is extended while preserving its existing prefix.

Resume from an earlier dynamic checkpoint and create a new training branch:

```bash
$PYTHON scripts/train_medgs4d.py \
  --data-dir "$STUDY_DIR" \
  --canonical-model "$CANONICAL_RUN" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$DYNAMIC_ROOT" \
  --run-name "$DYNAMIC_RUN_NAME" \
  --canonical-phase 0 \
  --target-representation raw \
  --split-mode full \
  --iterations 2000 \
  --learning-rate 5e-4 \
  --checkpoint-every 500 \
  --log-every 25 \
  --validate-every 0 \
  --validation-samples 0 \
  --phase-jitter-std 0 \
  --seed 42 \
  --device cuda \
  --resume \
  --resume-checkpoint "$RUN_DIR/checkpoints/deformation_iter_000500.pth"
```

Resume rules:

- `--resume-checkpoint` requires `--resume`,
- all saved settings except the target iteration must remain unchanged,
- the same canonical checkpoint must be used,
- histories are trimmed to the selected dynamic checkpoint,
- stale final evaluation and report artifacts are removed,
- `deformation_latest.pth` is reset to the selected branch point,
- the target iteration cannot be earlier than the selected resume checkpoint.

### 3.8 Evaluate one dynamic checkpoint

The final evaluation is automatic unless `--skip-final-evaluation` was used.

Manual evaluation:

```bash
$PYTHON scripts/evaluate_medgs4d.py \
  --run-dir "$RUN_DIR" \
  --split all \
  --checkpoint "$RUN_DIR/checkpoints/deformation_iter_001000.pth" \
  --force \
  --device cuda
```

Options:

- `--split all` evaluates canonical, train, and validation rows,
- `--split train` evaluates training rows only,
- `--split validation` evaluates held-out rows only,
- `--checkpoint` selects an exact deformation checkpoint,
- `--force` replaces an existing evaluation for the selected split.

### 3.9 Compare several dynamic checkpoints

Compare selected iterations:

```bash
$PYTHON scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --iterations 0 500 1000 \
  --device cuda
```

Without `--iterations` or `--checkpoints`, every saved `deformation_iter_*.pth` checkpoint is evaluated.

Outputs:

```text
evaluation/
├── checkpoint_comparison.csv
├── checkpoint_comparison.pdf
└── checkpoints/
    ├── iter_000000/
    │   ├── per_slice.csv
    │   ├── per_phase.csv
    │   └── overall.json
    ├── iter_000500/
    └── iter_001000/
```

`checkpoint_comparison.csv` contains full-evaluation baseline and dynamic metrics for each selected checkpoint. `checkpoint_comparison.pdf` plots their evolution versus checkpoint iteration.

Use `--force` to recompute checkpoint-specific files that already exist.

### 3.10 Diagnose the learned deformation

This is a post-checkpoint analysis. It is intentionally not run in every training iteration.

Latest checkpoint:

```bash
$PYTHON scripts/diagnose_deformation.py \
  --run-dir "$RUN_DIR" \
  --device cuda
```

Exact checkpoint:

```bash
$PYTHON scripts/diagnose_deformation.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/checkpoints/deformation_iter_000500.pth" \
  --device cuda
```

Faster sampled diagnostic:

```bash
$PYTHON scripts/diagnose_deformation.py \
  --run-dir "$RUN_DIR" \
  --sample-gaussians 5000 \
  --device cuda
```

Outputs:

```text
evaluation/deformation/iter_000500/
├── per_phase.csv
├── temporal_steps.csv
├── summary.json
└── diagnostics.pdf
```

Artifacts:

| File | Meaning |
|---|---|
| `per_phase.csv` | Mean, median, p95, and maximum `x-z` displacement, absolute `m` change, normalized deformation magnitude, and signed mean `Δx`/`Δz` per phase. |
| `temporal_steps.csv` | Change in deformation between adjacent respiratory phases, including the cyclic last-to-first transition. |
| `summary.json` | Selected checkpoint, canonical checkpoint, evaluated phases, Gaussian counts, anchor residual, and headline maxima. |
| `diagnostics.pdf` | Publication-ready spatial, `m`, normalized, directional, and temporal deformation plots. |

At the canonical phase, the residual should be numerically close to zero because deformation is defined relative to the canonical MLP output.

## 4. Notebook inspection

Use the repository's `MedGS WORF` kernel or another kernel backed by the same editable installation.

### Load a run and inspect metrics

```python
from pathlib import Path

from medgs4d.results import load_run, print_run_summary
from medgs4d.visualization import show_training_curves

RUN_DIR = Path(
    "/home/jovyan/shared/mtm_medgs_stack/results/medgs4d/"
    "patient113_19991117/"
    "phase00_canon5000_full_smoke_iter1000"
)

run = load_run(run_dir=RUN_DIR)

print_run_summary(run)
show_training_curves(run)
```

Reload `run` to refresh CSV data while training is active:

```python
run = load_run(run_dir=RUN_DIR)
display(run.training_history.tail(20))
```

### Browse reconstructions

Run this after training, or on another free GPU:

```python
from medgs4d.visualization import show_slice_browser

show_slice_browser(run, device="cuda")
```

The browser displays:

```text
Ground truth | Canonical baseline | Dynamic reconstruction
```

Full respiratory cycle for one slice:

```python
from medgs4d.visualization import show_breathing_cycle_browser

show_breathing_cycle_browser(run, device="cuda")
```

### Browse error maps

```python
from medgs4d.visualization import show_error_map_browser

show_error_map_browser(run, device="cuda")
```

The browser displays:

```text
Ground truth
Canonical baseline
Dynamic reconstruction
|GT - baseline|
|GT - dynamic|
Error reduction
```

For `Error reduction`:

- positive values mean the dynamic reconstruction is better,
- negative values mean the canonical baseline is better.

Save selected error maps:

```python
from medgs4d.visualization import save_run_error_maps

output_dir = save_run_error_maps(
    run,
    phases=[0, 20, 40, 60, 80],
    slice_indices=[20, 25, 30],
    device="cuda",
)

print(output_dir)
```

Files are written under:

```text
visualizations/error_maps/
```

### Load an exact dynamic checkpoint in the notebook

```python
from medgs4d.results import load_run_models

checkpoint = RUN_DIR / "checkpoints" / "deformation_iter_000500.pth"

canonical, field = load_run_models(
    run,
    checkpoint=checkpoint,
    device="cuda",
)

print("Canonical checkpoint:", canonical.checkpoint_path)
print("Canonical iteration:", canonical.loaded_iteration)
print("Dynamic checkpoint:", checkpoint)
```

## 5. CLI parameter reference

### `scripts/data.py`

| Subcommand | Parameters |
|---|---|
| `list-patients` | Exactly one of `--archives-dir`, `--dicom-dir`. |
| `extract` | `--archives-dir`, `--dicom-dir`, `--patient-id`; optional `--workers 4`, `--force`, `--dry-run`, `--log PATH`. |
| `list-studies` | `--dicom-dir`, `--patient-id`. |
| `inspect` | `--dicom-dir`, `--patient-id`, `--study-uid`. |
| `prepare` | `--dicom-dir`, `--prepared-root`, `--patient-id`, `--study-uid`, `--study-name`; optional `--hu-window -1000 400`, `--denoise-sigma 0.20 0.40 0.40`, `--force`. |

### `scripts/train_canonical.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--data-dir` | required | Prepared study directory containing `manifest.json`. |
| `--medgs-repo` | required | Upstream MedGS repository. |
| `--output-root` | required | Canonical results root. |
| `--run-name` | required | Filesystem-safe run identifier. |
| `--canonical-phase` | required | Respiratory phase used as the static reference. |
| `--representation` | `raw` | Prepared `raw` or `denoised` target volume. |
| `--iterations` | `30000` | Final upstream MedGS iteration. |
| `--poly-degree` | `2` | Upstream temporal polynomial degree. |
| `--batch-size` | `3` | Upstream MedGS batch size. |
| `--camera` | `mirror` | Upstream camera/input mode. |
| `--seed` | `42` | Reproducibility seed. |
| `--log-every` | `100` | Canonical training-history cadence. |
| `--resume` | off | Continue from the newest canonical checkpoint. |
| `--force` | off | Delete and recreate the selected canonical run. |

`--resume` and `--force` are mutually exclusive.

### `scripts/train_medgs4d.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--data-dir` | required | Prepared study directory. |
| `--canonical-model` | required | Canonical run directory. |
| `--canonical-checkpoint` | run config | Exact canonical `chkpnt*.pth` to freeze. |
| `--medgs-repo` | required | Upstream MedGS repository. |
| `--output-root` | required | Dynamic results root. |
| `--run-name` | required | Filesystem-safe dynamic run identifier. |
| `--canonical-phase` | required | Must match the canonical run phase. |
| `--target-representation` | `raw` | Dynamic training target: `raw` or `denoised`. |
| `--split-mode` | `full` | `full` or `phase-holdout`. |
| `--validation-phases` | empty | Complete phases assigned to validation. |
| `--iterations` | `7000` | Final dynamic training iteration. |
| `--learning-rate` | `5e-4` | Adam learning rate. |
| `--spatial-frequencies` | `4` | Fourier frequency count for canonical `x`, `z`, and `m`. |
| `--phase-frequencies` | `2` | Cyclic Fourier frequency count for respiratory phase. |
| `--hidden-dim` | `256` | MLP hidden width. |
| `--hidden-layers` | `4` | Number of hidden MLP layers. |
| `--chunk-size` | `131072` | Gaussians processed per MLP chunk. |
| `--checkpoint-every` | `250` | Dynamic checkpoint cadence. |
| `--log-every` | `25` | Training-history write and console-log cadence. |
| `--validate-every` | `250` | Compact validation cadence; `0` disables it. |
| `--validation-samples` | `20` | Size of the fixed validation subset; `0` disables it. |
| `--seed` | `42` | Sampling, model, and diagnostic reproducibility seed. |
| `--phase-jitter-std` | `0.0` | Initial standard deviation of cyclic phase jitter. |
| `--skip-final-evaluation` | off | Finish after training artifacts; evaluate later manually. |
| `--resume` | off | Resume the selected run. |
| `--resume-checkpoint` | latest | Exact dynamic checkpoint used as the branch point. |
| `--force` | off | Delete and recreate the selected dynamic run. |
| `--device` | `cuda` | Torch device. |

Internal loss and stability defaults in `medgs4d/config.py`:

```text
max_gradient_norm   = 10.0
smoothness_gaussians = 65536
l1_weight           = 2.0
ssim_weight         = 0.25
magnitude_weight    = 1e-4
smoothness_weight   = 1e-3
```

These values are saved in `config.json` but are not currently exposed as CLI flags.

### Evaluation and diagnostics

```text
evaluate_canonical.py:
  --run-dir
  --medgs-repo
  --target-representation {raw,denoised}
  --device

evaluate_medgs4d.py:
  --run-dir
  --split {all,train,validation}
  --checkpoint
  --force
  --device

evaluate_checkpoints.py:
  --run-dir
  --iterations N [N ...]
  --checkpoints PATH [PATH ...]
  --split {all,train,validation}
  --force
  --device

diagnose_deformation.py:
  --run-dir
  --checkpoint
  --phases PHASE [PHASE ...]
  --sample-gaussians N
  --seed
  --device

visualize_medgs4d.py:
  --run-dir
  --slice INDEX
  exactly one of --phase PHASE or --all-phases
  --output PATH
  --device
```

## 6. Operational notes

- Do not load a second copy of the model in a notebook on the same GPU while training unless sufficient VRAM is available.
- `--force` removes only the selected run directory, but it is destructive.
- Canonical `config.json` records the latest completed canonical target. A dynamic run can still freeze an older canonical checkpoint through `--canonical-checkpoint`.
- Dynamic `config.json` records the exact canonical checkpoint, preventing later evaluation from silently switching to a newer canonical model.
- Full evaluation is more reliable than minibatch training curves because it evaluates all selected phase-slice pairs.
- Use checkpoint comparison before assuming that the last checkpoint is the best checkpoint.
- `full` split evaluates fit to observed phases. Use `phase-holdout` when measuring temporal interpolation to unseen phases.
