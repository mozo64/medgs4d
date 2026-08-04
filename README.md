# MedGS4D Technical Guide

MedGS4D builds a phase-conditioned deformation model on top of a frozen canonical MedGS reconstruction of 4D-CT data. The repository supports three connected workflows:

1. **Image reconstruction:** prepare respiratory CT volumes, train a static canonical MedGS model, train a dynamic deformation MLP, and evaluate reconstructed slices across respiratory phases.
2. **Reference geometry:** convert phase-specific RTSTRUCT contours into voxel masks and watertight meshes used as geometric ground truth.
3. **Gaussian-to-mesh baseline:** map MedGS latent coordinates to DICOM patient coordinates, transfer learned Gaussian motion to a fixed-topology canonical tumor mesh, and evaluate predicted meshes against phase-specific RTSTRUCT references.

All workflows use the same DICOM study, respiratory phase convention, and patient-coordinate geometry. RTSTRUCT meshes remain deterministic reference artifacts. Predicted meshes are obtained by deforming the phase-0 reference mesh with a training-free, robust interpolation of the learned MedGS4D Gaussian motion.

The upstream MedGS repository remains external and is passed explicitly with `--medgs-repo`.

## 1. Current scope

The repository currently provides:

- discovery and extraction of 4D-Lung DICOM series,
- inspection of studies, respiratory phases, CT series, and RTSTRUCT objects,
- reusable raw and denoised HU volumes,
- canonical MedGS training, resume, evaluation, and history tracking,
- phase-conditioned MedGS4D training with full or phase-holdout splits,
- checkpoint comparison, error maps, and deformation diagnostics,
- RTSTRUCT inventory and ROI geometry inspection,
- batch construction of reference masks and watertight meshes for all respiratory phases,
- validated MedGS latent-to-DICOM coordinate transforms,
- Gaussian-neighborhood analysis around a canonical tumor mesh,
- robust transfer of Gaussian displacement to fixed-topology mesh vertices,
- Dice, IoU, surface-distance, HD95, centroid, volume, watertightness, and degeneracy evaluation,
- compact saved predictions and phase-comparison PNGs,
- notebook inspection of RTSTRUCT geometry, Gaussian alignment, image reconstruction, and Gaussian-to-mesh results.

Current limitations:

- the Gaussian-to-mesh transfer is heuristic and has no geometry-supervised training,
- nearby Gaussians are selected geometrically, not semantically as tumor Gaussians,
- the worked result covers one patient and one study,
- direct transfer improves centroid tracking but does not yet recover reliable tumor shape or volume,
- independently generated RTSTRUCT reference meshes do not have vertex correspondence across phases.

## 2. Repository structure

```text
medgs4d/
├── medgs4d/
│   ├── canonical.py
│   ├── config.py
│   ├── data.py
│   ├── deformation.py
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── gaussian_geometry.py
│   ├── mesh_series.py
│   ├── mesh_transfer.py
│   ├── mesh_validation.py
│   ├── meshes.py
│   ├── reporting.py
│   ├── results.py
│   ├── rtstruct.py
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
│   ├── export_error_maps.py
│   ├── inspect_gaussian_geometry.py
│   ├── predict_mesh_from_gaussians.py
│   ├── rtstruct_mesh.py
│   ├── visualize_medgs4d.py
│   └── run_patient117_20001024_medgs4d.sh
├── notebooks/
│   ├── 04_validate_rtstruct_mesh.ipynb
│   └── results_browser.ipynb
├── tests/
│   ├── test_gaussian_geometry.py
│   ├── test_mesh_series.py
│   ├── test_mesh_transfer.py
│   ├── test_rtstruct_mesh.py
│   ├── test_rtstruct_roi_inspection.py
│   └── other unit and integration tests
├── pyproject.toml
└── README.md
```

The Python package contains reusable logic. The scripts are intentionally thin CLI entry points. Notebook code loads prepared artifacts rather than reimplementing DICOM parsing, rasterization, training, coordinate mapping, mesh transfer, or evaluation.

## 3. Environment and installation

**What:** install the local package in editable mode and expose the required GPU.

**Why:** scripts and notebooks must import the same working tree, while upstream MedGS remains a separate repository.

Example WORF setup:

```bash
cd /home/jovyan/shared/mtm_medgs_stack/repo/medgs4d

PYTHON=/home/jovyan/shared/mtm_medgs_stack/envs/medgs-worf/bin/python
MEDGS_REPO=/home/jovyan/shared/mtm_medgs_stack/repo/MedGS

$PYTHON -m pip install -e .

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`CUDA_VISIBLE_DEVICES=1` exposes the second physical GPU as logical `cuda:0` inside PyTorch. CLI commands still use `--device cuda`.

Basic checks:

```bash
$PYTHON -m compileall -q medgs4d scripts tests
$PYTHON -m pytest -q -m "not worf"
```

Inspect any command before running it:

```bash
$PYTHON scripts/data.py --help
$PYTHON scripts/train_medgs4d.py --help
$PYTHON scripts/rtstruct_mesh.py --help
$PYTHON scripts/inspect_gaussian_geometry.py --help
$PYTHON scripts/predict_mesh_from_gaussians.py --help
```

## 4. Worked study and shared paths

The examples below use the annotated 4D-Lung study processed during development:

```bash
REPO=/home/jovyan/shared/mtm_medgs_stack/repo/medgs4d
PYTHON=/home/jovyan/shared/mtm_medgs_stack/envs/medgs-worf/bin/python
MEDGS_REPO=/home/jovyan/shared/mtm_medgs_stack/repo/MedGS

ARCHIVES_DIR=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/raw/series_zips
DICOM_DIR=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/raw/dicom_by_series
PREPARED_ROOT=/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/prepared
CANONICAL_ROOT=/home/jovyan/shared/mtm_medgs_stack/results/canonical
DYNAMIC_ROOT=/home/jovyan/shared/mtm_medgs_stack/results/medgs4d

PATIENT_ID=117_HM10395
STUDY_UID=1.3.6.1.4.1.14519.5.2.1.6834.5010.378204929111417980831212264180
STUDY_NAME=patient117_20001024_annotated
STUDY_DIR="$PREPARED_ROOT/$STUDY_NAME"
```

The study contains ten respiratory CT phases and ten phase-specific RTSTRUCT objects. The CT reconstruction workflow uses the prepared study directory; the RTSTRUCT mesh workflow reads the original DICOM geometry and writes its artifacts under the same study root.

## 5. Data discovery and preparation

### 5.1 List available patients

**What:** summarize patients present in the source archives or already extracted DICOM tree.

**Why:** confirm the patient identifier and available archive count before extraction.

```bash
$PYTHON scripts/data.py list-patients \
  --archives-dir "$ARCHIVES_DIR"
```

Already extracted patients:

```bash
$PYTHON scripts/data.py list-patients \
  --dicom-dir "$DICOM_DIR"
```

### 5.2 Extract one patient

**What:** unpack every series archive belonging to one patient into a stable directory per `SeriesInstanceUID`.

**Why:** later study inspection and DICOM loading operate on extracted files, not directly on ZIP archives.

```bash
$PYTHON scripts/data.py extract \
  --archives-dir "$ARCHIVES_DIR" \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --workers 4 \
  --log logs/patient117_extraction.csv
```

Output layout:

```text
<DICOM_DIR>/
└── 117_HM10395/
    └── <SeriesInstanceUID>/
        ├── DICOM files
        └── .unpack_complete.json
```

`.unpack_complete.json` records source and completion metadata. Repeated extraction skips complete series unless `--force` is used.

### 5.3 List studies, series, and RTSTRUCT objects

**What:** inspect the patient hierarchy without yet choosing training targets.

**Why:** one patient may contain several studies, non-respiratory series, RTSTRUCT objects, and studies with different phase or slice coverage.

List studies:

```bash
$PYTHON scripts/data.py list-studies \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID"
```

List every series in the selected study:

```bash
$PYTHON scripts/data.py list-series \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID"
```

Optional modality filter:

```bash
$PYTHON scripts/data.py list-series \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID" \
  --modality CT
```

List RTSTRUCT objects and their referenced CT series:

```bash
$PYTHON scripts/data.py list-rtstructs \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --referenced-study-uid "$STUDY_UID"
```

### 5.4 Inspect respiratory CT geometry

**What:** resolve respiratory phases and report CT dimensions, slice counts, coordinates, and spacing.

**Why:** phase volumes must have compatible geometry before they are prepared, trained, or compared.

```bash
$PYTHON scripts/data.py inspect \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID"
```

For the worked study, confirm ten CT phases `0, 10, ..., 90`, compatible in-plane dimensions, and regular slice spacing.

### 5.5 Prepare reusable 4D-CT volumes

**What:** reconstruct each respiratory CT series as a NumPy HU volume and optionally produce a denoised copy.

**Why:** training and evaluation should not repeatedly parse thousands of DICOM files or reconstruct slice geometry.

The worked training run uses a somewhat stronger Gaussian filter than the package default:

```bash
$PYTHON scripts/data.py prepare \
  --dicom-dir "$DICOM_DIR" \
  --prepared-root "$PREPARED_ROOT" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID" \
  --study-name "$STUDY_NAME" \
  --hu-window -1000 400 \
  --denoise-sigma 0.35 0.70 0.70
```

Prepared study layout:

```text
<PREPARED_ROOT>/patient117_20001024_annotated/
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

| Artifact | Meaning |
|---|---|
| `manifest.json` | Prepared-study contract: patient and study identifiers, phases, volume shape, HU window, denoising sigma, and paths. |
| `phase_summary.csv` | One row per phase with dimensions, slice count, coordinate range, spacing, and volume paths. |
| `phase_slice_manifest.csv` | One row per phase and slice with source DICOM path, index, coordinate, and geometry metadata. |
| `volumes/raw/phase_*.npy` | Reconstructed `float32` CT volumes in Hounsfield units. |
| `volumes/denoised/phase_*.npy` | The same HU volumes after the configured 3-D Gaussian filter. |

The HU window controls conversion to image intensities. The NumPy CT volumes retain HU values.

```bash
cat "$STUDY_DIR/manifest.json"
head "$STUDY_DIR/phase_summary.csv"
head "$STUDY_DIR/phase_slice_manifest.csv"
```

## 6. Canonical MedGS reconstruction

### 6.1 Train the canonical model

**What:** convert one selected respiratory phase into the upstream MedGS image layout and optimize a static Gaussian representation.

**Why:** MedGS4D deforms a frozen canonical representation; it does not learn the canonical Gaussian cloud from scratch.

Worked configuration:

```bash
CANONICAL_PHASE=0
CANONICAL_ITERATIONS=15000
CANONICAL_RUN_NAME=phase00_denoised_iter15000
CANONICAL_RUN="$CANONICAL_ROOT/$STUDY_NAME/$CANONICAL_RUN_NAME"

$PYTHON scripts/train_canonical.py \
  --data-dir "$STUDY_DIR" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$CANONICAL_ROOT" \
  --run-name "$CANONICAL_RUN_NAME" \
  --canonical-phase "$CANONICAL_PHASE" \
  --representation denoised \
  --iterations "$CANONICAL_ITERATIONS" \
  --poly-degree 2 \
  --batch-size 3 \
  --camera mirror \
  --log-every 100
```

The script:

1. converts the selected HU volume into the upstream MedGS image dataset,
2. launches upstream training,
3. stores restartable canonical checkpoints,
4. performs full-slice canonical evaluation,
5. updates evaluation history and PDF plots.

Canonical run layout:

```text
<CANONICAL_ROOT>/patient117_20001024_annotated/phase00_denoised_iter15000/
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
│   ├── chkpnt15000.pth
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

| Artifact | Meaning |
|---|---|
| `config.json` | Requested canonical phase, representation, iterations, polynomial degree, batch size, camera mode, and seed. |
| `canonical_run.json` | Resolved paths and provenance, including the prepared study, upstream repository, checkpoint, and resume metadata. |
| `frame_manifest.csv` | Mapping between CT slices and generated MedGS input images. |
| `dataset/original/*.png` | Windowed canonical CT slices. |
| `dataset/mirror/*.png` | Mirrored images required by the selected upstream camera mode. |
| `model/chkpntN.pth` | Restartable upstream MedGS checkpoint; the payload also records the true iteration. |
| `canonical_training_history.csv` | Minibatch metrics sampled during optimization. |
| `canonical_training_history.pdf` | Canonical training curves. |
| `evaluation/per_slice.csv` | Full-slice L1, PSNR, and SSIM for the selected checkpoint. |
| `evaluation/overall.json` | Mean, standard deviation, minimum, and maximum over all canonical slices. |
| `evaluation/canonical_metrics.pdf` | Full-slice metrics versus slice index. |
| `evaluation/history.csv` | One full-slice evaluation point per completed canonical checkpoint stage. |
| `evaluation/history.pdf` | Full-slice metrics versus canonical checkpoint iteration. |

`canonical_training_history.csv` contains:

```text
Iteration, TotalLoss, L1, InterpolationL1, SSIM, SSIMLoss,
SigmaLoss, PSNR, EMALoss, EMAPSNR, GaussianCount,
IterationTimeMs, ElapsedSeconds
```

### 6.2 Resume canonical training

**What:** extend the same canonical run from its newest `chkpnt*.pth`.

**Why:** a longer run should retain its earlier checkpoints and histories rather than create an unrelated experiment.

Only the target iteration may change:

```bash
$PYTHON scripts/train_canonical.py \
  --data-dir "$STUDY_DIR" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$CANONICAL_ROOT" \
  --run-name "$CANONICAL_RUN_NAME" \
  --canonical-phase 0 \
  --representation denoised \
  --iterations 20000 \
  --poly-degree 2 \
  --batch-size 3 \
  --camera mirror \
  --log-every 100 \
  --resume
```

`--resume` and `--force` are mutually exclusive.

### 6.3 Re-evaluate a canonical run

**What:** recompute full-slice metrics for the currently selected canonical checkpoint.

**Why:** training minibatches are not a substitute for deterministic evaluation over every canonical slice.

```bash
$PYTHON scripts/evaluate_canonical.py \
  --run-dir "$CANONICAL_RUN" \
  --medgs-repo "$MEDGS_REPO" \
  --target-representation denoised \
  --device cuda
```

## 7. Dynamic MedGS4D reconstruction

### 7.1 Model and checkpoint provenance

**What:** freeze an exact canonical checkpoint and train only a phase-conditioned deformation MLP.

**Why:** later evaluation must use the same canonical Gaussian state that was used during dynamic training.

The loader verifies the canonical checkpoint iteration both from the filename and from the saved payload. The exact path and iteration are persisted in the dynamic `config.json`.

```bash
CANONICAL_CHECKPOINT="$CANONICAL_RUN/model/chkpnt15000.pth"
DYNAMIC_RUN_NAME=phase00_canon15000_denoised_holdout_iter7000
RUN_DIR="$DYNAMIC_ROOT/$STUDY_NAME/$DYNAMIC_RUN_NAME"
```

The canonical model remains frozen. The dynamic optimizer updates only the deformation network, which predicts phase-conditioned residuals for canonical Gaussian parameters `x`, `z`, and `m`.

### 7.2 Split design

**What:** assign complete respiratory phases to canonical, training, or validation subsets.

**Why:** holding out whole phases measures temporal interpolation, whereas random slice splitting would leak nearly identical anatomy across train and validation.

`full` split:

- canonical phase → `canonical`,
- every other phase → `train`,
- no held-out phase.

`phase-holdout` split used in the worked case:

- canonical phase `0` → `canonical`,
- phases `20, 40, 60, 80` → `train`,
- phases `10, 30, 50, 70, 90` → `validation`.

### 7.3 Train the dynamic model

```bash
$PYTHON scripts/train_medgs4d.py \
  --data-dir "$STUDY_DIR" \
  --canonical-model "$CANONICAL_RUN" \
  --canonical-checkpoint "$CANONICAL_CHECKPOINT" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$DYNAMIC_ROOT" \
  --run-name "$DYNAMIC_RUN_NAME" \
  --canonical-phase 0 \
  --target-representation denoised \
  --split-mode phase-holdout \
  --validation-phases 10 30 50 70 90 \
  --iterations 7000 \
  --learning-rate 5e-4 \
  --checkpoint-every 500 \
  --log-every 25 \
  --validate-every 250 \
  --validation-samples 20 \
  --phase-jitter-std 0 \
  --seed 42 \
  --device cuda
```

Dynamic run layout:

```text
<DYNAMIC_ROOT>/patient117_20001024_annotated/
└── phase00_canon15000_denoised_holdout_iter7000/
    ├── config.json
    ├── split_manifest.csv
    ├── sampling_plan.csv
    ├── deformation_normalization.json
    ├── smoothness_gaussian_indices.npy
    ├── training_history.csv
    ├── validation_history.csv
    ├── training_summary.csv
    ├── training_history.pdf
    ├── completion.json
    ├── report.pdf
    ├── report_metrics.csv
    ├── checkpoints/
    │   ├── deformation_iter_000000.pth
    │   ├── deformation_iter_000500.pth
    │   ├── ...
    │   ├── deformation_iter_007000.pth
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

| Artifact | Meaning |
|---|---|
| `config.json` | Complete reproducible dynamic configuration, including the exact canonical checkpoint. |
| `split_manifest.csv` | Every phase-slice pair assigned to `canonical`, `train`, or `validation`. |
| `sampling_plan.csv` | Deterministic sample selected for each training iteration, including phase, slice, neighboring phase, and optional phase jitter. |
| `deformation_normalization.json` | Coordinate normalization statistics and deformation-network dimensions. |
| `smoothness_gaussian_indices.npy` | Reproducible Gaussian subset used by temporal smoothness regularization. |
| `training_history.csv` | Per-iteration optimization metrics written every `--log-every` iterations. |
| `validation_history.csv` | Metrics on the fixed compact validation subset, written every `--validate-every`; absent when validation is disabled. |
| `checkpoints/deformation_iter_*.pth` | Network, optimizer, iteration, run identity, canonical checkpoint identity, and RNG states. |
| `checkpoints/deformation_latest.pth` | Most recent selected branch checkpoint used by default resume. |

`training_history.csv` includes:

```text
Iteration, PhasePercent, SliceIndex, NeighborPhasePercent,
PhaseJitterStd, PhaseJitter, LearningRate, ElapsedSeconds,
PeakGpuMemoryGB, TotalLoss, ImageLoss, L1, SSIM, PSNR,
DeformationMagnitudeLoss, TemporalSmoothnessLoss, GradientNorm
```

Artifacts produced after training:

| Artifact | Meaning |
|---|---|
| `training_summary.csv` | Final-window means overall and per respiratory phase. |
| `training_history.pdf` | Rolling reconstruction, loss, gradient, and GPU-memory curves. |
| `completion.json` | Completion status, iteration, parameter count, sample counts, canonical checkpoint identity, and elapsed time. |
| `report_metrics.csv` | Compact baseline/dynamic/improvement table used in the report. |
| `report.pdf` | Concise run summary with headline metrics, per-phase PSNR gain, and training curves. |
| `evaluation/*` | Full deterministic reconstruction evaluation described in the next section. |

### 7.4 Resume or branch a dynamic run

**What:** continue the latest checkpoint or restart from an earlier checkpoint while preserving run provenance.

**Why:** model selection may show that an earlier state is preferable, and a new branch should not silently retain stale later histories or reports.

Continue to a later target:

```bash
$PYTHON scripts/train_medgs4d.py \
  --data-dir "$STUDY_DIR" \
  --canonical-model "$CANONICAL_RUN" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$DYNAMIC_ROOT" \
  --run-name "$DYNAMIC_RUN_NAME" \
  --canonical-phase 0 \
  --target-representation denoised \
  --split-mode phase-holdout \
  --validation-phases 10 30 50 70 90 \
  --iterations 9000 \
  --learning-rate 5e-4 \
  --checkpoint-every 500 \
  --log-every 25 \
  --validate-every 250 \
  --validation-samples 20 \
  --phase-jitter-std 0 \
  --seed 42 \
  --device cuda \
  --resume
```

Branch from an exact checkpoint:

```bash
$PYTHON scripts/train_medgs4d.py \
  --data-dir "$STUDY_DIR" \
  --canonical-model "$CANONICAL_RUN" \
  --medgs-repo "$MEDGS_REPO" \
  --output-root "$DYNAMIC_ROOT" \
  --run-name "$DYNAMIC_RUN_NAME" \
  --canonical-phase 0 \
  --target-representation denoised \
  --split-mode phase-holdout \
  --validation-phases 10 30 50 70 90 \
  --iterations 9000 \
  --learning-rate 5e-4 \
  --checkpoint-every 500 \
  --log-every 25 \
  --validate-every 250 \
  --validation-samples 20 \
  --phase-jitter-std 0 \
  --seed 42 \
  --device cuda \
  --resume \
  --resume-checkpoint "$RUN_DIR/checkpoints/deformation_iter_005000.pth"
```

Resume rules:

- `--resume-checkpoint` requires `--resume`,
- the canonical checkpoint identity must remain unchanged,
- all saved settings except the target iteration must match,
- histories are trimmed to the selected branch point,
- stale final evaluation and report artifacts are removed,
- `deformation_latest.pth` is reset to the selected branch point,
- the new target cannot be earlier than the selected checkpoint.

## 8. Evaluation and model selection

### 8.1 Evaluation protocol

**What:** render the frozen canonical baseline and the dynamically deformed reconstruction for every selected phase-slice pair, then aggregate reconstruction metrics.

**Why:** minibatch training curves measure optimization samples; they do not provide a stable comparison across all slices, phases, and checkpoints.

For each evaluated sample:

- **Ground truth:** the selected prepared CT representation (`raw` or `denoised`).
- **Canonical baseline:** the frozen canonical Gaussians rendered without phase deformation.
- **Dynamic reconstruction:** the same canonical Gaussians after applying the learned phase-conditioned residuals.

Metrics:

- lower `L1` is better,
- higher `PSNR` is better,
- higher `SSIM` is better,
- positive `L1Reduction`, `PSNRGain`, and `SSIMGain` mean the dynamic model improved over the canonical baseline.

Evaluation outputs:

| Artifact | Scope |
|---|---|
| `evaluation/per_slice.csv` | One row per evaluated phase-slice pair with baseline, dynamic, and improvement metrics. |
| `evaluation/per_phase.csv` | Mean metrics grouped by respiratory phase and split. |
| `evaluation/overall.json` | Aggregates for all samples, noncanonical samples, and individual splits. |
| `evaluation/metrics.pdf` | Baseline-versus-dynamic metrics and improvements across phases. |
| `evaluation/history.csv` | One full-evaluation point per completed dynamic checkpoint stage. |
| `evaluation/history.pdf` | Full-evaluation metrics versus checkpoint iteration. |

Key `per_slice.csv` columns:

```text
BaselineL1, DynamicL1, L1Reduction,
BaselinePSNR, DynamicPSNR, PSNRGain,
BaselineSSIM, DynamicSSIM, SSIMGain
```

Interpret splits carefully:

- `full` measures fitting to all observed noncanonical phases; it does not measure interpolation to unseen phases.
- `phase-holdout` provides a proper temporal interpolation test only on rows marked `validation`.
- checkpoint selection should prioritize full validation metrics, not the last training iteration or the lowest minibatch loss.

### 8.2 Evaluate one dynamic checkpoint

**What:** run the complete deterministic protocol for one exact deformation checkpoint.

**Why:** the latest checkpoint is a convenience default, not proof that it is the best checkpoint.

```bash
$PYTHON scripts/evaluate_medgs4d.py \
  --run-dir "$RUN_DIR" \
  --split all \
  --checkpoint "$RUN_DIR/checkpoints/deformation_iter_007000.pth" \
  --force \
  --device cuda
```

Available scopes:

- `--split all` evaluates canonical, train, and validation rows,
- `--split train` evaluates only training rows,
- `--split validation` evaluates only held-out rows.

Final evaluation runs automatically after training unless `--skip-final-evaluation` was used.

### 8.3 Compare multiple checkpoints

**What:** evaluate several saved checkpoints using the same sample set and aggregate them into one comparison.

**Why:** validation quality may peak before the final checkpoint even while training loss continues to decrease.

Worked comparison:

```bash
$PYTHON scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --iterations 0 1000 2500 5000 7000 \
  --device cuda
```

Without `--iterations` or `--checkpoints`, all saved `deformation_iter_*.pth` files are evaluated.

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
    ├── iter_001000/
    └── ...
```

Recommended selection procedure for a phase-holdout run:

1. inspect `checkpoint_comparison.csv`,
2. select the checkpoint with the strongest held-out PSNR/SSIM and acceptable L1,
3. confirm that gains are distributed across validation phases rather than driven by one phase,
4. inspect error maps for spatial failure modes,
5. inspect deformation diagnostics for excessive or unstable motion.

### 8.4 Export reconstruction error maps

**What:** save spatial comparisons between ground truth, canonical baseline, and dynamic reconstruction.

**Why:** scalar metrics cannot reveal where the model improved or where it introduced new local errors.

```bash
$PYTHON scripts/export_error_maps.py \
  --run-dir "$RUN_DIR" \
  --phases 10 30 50 70 90 \
  --slices 27 \
  --checkpoint "$RUN_DIR/checkpoints/deformation_iter_007000.pth" \
  --device cuda
```

Optional shared error scale:

```bash
--error-max 0.25
```

Output directory:

```text
<RUN_DIR>/visualizations/error_maps/
├── phase_10_slice_027.png
├── phase_30_slice_027.png
├── phase_50_slice_027.png
├── phase_70_slice_027.png
└── phase_90_slice_027.png
```

Each figure contains:

```text
Ground truth
Canonical baseline
Dynamic reconstruction
|GT - baseline|
|GT - dynamic|
Error reduction
```

Positive error reduction means the dynamic model reduced the local error. Negative values mean the canonical baseline was locally better.

### 8.5 Diagnose the learned deformation field

**What:** summarize phase-dependent Gaussian displacement and temporal changes after a checkpoint is trained.

**Why:** improved image metrics alone do not guarantee a plausible or stable deformation field.

```bash
FINAL_DYNAMIC_CHECKPOINT="$RUN_DIR/checkpoints/deformation_iter_007000.pth"

$PYTHON scripts/diagnose_deformation.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$FINAL_DYNAMIC_CHECKPOINT" \
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
evaluation/deformation/iter_007000/
├── per_phase.csv
├── temporal_steps.csv
├── summary.json
└── diagnostics.pdf
```

| Artifact | Meaning |
|---|---|
| `per_phase.csv` | Mean, median, p95, and maximum `x-z` displacement, absolute `m` change, normalized magnitude, and signed mean `Δx`/`Δz`. |
| `temporal_steps.csv` | Change between neighboring respiratory phases, including the cyclic last-to-first step. |
| `summary.json` | Checkpoint identities, phases, Gaussian counts, anchor residual, and headline maxima. |
| `diagnostics.pdf` | Spatial, parameter, directional, normalized, and temporal plots. |

The residual at the canonical phase should be numerically close to zero because deformation is defined relative to the canonical network output.

## 9. RTSTRUCT reference mesh workflow

This workflow creates reference geometry without training.

```text
RTSTRUCT contours
→ voxel mask on the referenced CT grid
→ marching-cubes surface mesh
→ mesh rasterized back to a mask
→ geometric and round-trip validation
```

It reads original DICOM files because the RTSTRUCT stores contours in patient physical coordinates and references DICOM CT instances. It writes a complete ten-phase series directly to `tumor_4d`; the earlier exploratory single-phase output is not part of the documented workflow.

### 9.1 Inventory phase-specific RTSTRUCT objects

**What:** list one row per RTSTRUCT object with phase, ROI count, ROI names, and file path.

**Why:** verify that each respiratory phase contains the intended logical ROI before attempting batch construction.

```bash
mkdir -p logs

$PYTHON scripts/rtstruct_mesh.py inventory \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID" \
  --csv logs/patient117_20001024_rtstruct_inventory.csv
```

For the worked study, the inventory contains ten RTSTRUCT objects. Each phase contains four ROIs with phase-specific names:

```text
Tumor_c00, LN_c00, Carina_c00, Vertebra_c00
Tumor_c10, LN_c10, Carina_c10, Vertebra_c10
...
Tumor_c90, LN_c90, Carina_c90, Vertebra_c90
```

Inventory columns include:

```text
PatientID, StudyInstanceUID, PhasePercent,
RTSeriesInstanceUID, SOPInstanceUID, StructureSetLabel,
ROICount, ROINames, RTSTRUCTFile
```

### 9.2 Inspect ROI contour geometry

**What:** describe every ROI in one selected phase: contour count, point count, distinct contour slices, z extent, geometric type, and whether it is a plausible closed 3-D volume.

**Why:** some RTSTRUCT entries are landmarks or single-slice structures and should not be treated as volumetric meshes.

```bash
$PYTHON scripts/rtstruct_mesh.py inspect-rois \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID" \
  --phase 0 \
  --csv logs/patient117_20001024_phase00_roi_inspection.csv
```

Important columns:

| Column | Meaning |
|---|---|
| `ContourCount` | Number of planar contours assigned to the ROI. |
| `GeometricTypes` | DICOM contour type; closed volumetric candidates should normally be `CLOSED_PLANAR`. |
| `PointCount` | Total number of contour points. |
| `DistinctContourSlices` | Number of distinct contour planes. |
| `ContoursPerSliceMax` | Maximum number of separate loops on one plane. |
| `ZMinMm`, `ZMaxMm`, `ZExtentMm` | Physical contour extent along patient z. |
| `ClosedPlanar` | Whether all contours are closed planar polygons. |
| `VolumeCandidate` | Convenience flag requiring closed polygons on at least two distinct planes. |

For phase 0, `Tumor_c00` had seven closed contours on seven slices and was selected as the first volumetric target. `Carina_c00` had only one contour plane and was not a volumetric mesh candidate.

### 9.3 Build the complete ten-phase reference series

**What:** resolve each phase-specific RTSTRUCT and its referenced CT series, rasterize `Tumor_cXX`, create a watertight mesh, validate the round trip, and save a compact CT volume for notebook visualization.

**Why:** downstream analysis needs one consistent artifact contract for every respiratory phase, not repeated ad hoc DICOM processing in the notebook.

```bash
SERIES_DIR="$PREPARED_ROOT/$STUDY_NAME/rtstruct_mesh/tumor_4d"

$PYTHON scripts/rtstruct_mesh.py build-series \
  --dicom-dir "$DICOM_DIR" \
  --patient-id "$PATIENT_ID" \
  --study-uid "$STUDY_UID" \
  --phases 0 10 20 30 40 50 60 70 80 90 \
  --roi-template 'Tumor_c{phase:02d}' \
  --output-dir "$SERIES_DIR" \
  --ct-vis-stride 2 4 4 \
  --hu-window -1000 400 \
  --roundtrip-dice-min 0.95
```

The command fails when `SERIES_DIR` already exists. Use `--force` only when intentionally replacing the entire reference series.

Output layout:

```text
patient117_20001024_annotated/
└── rtstruct_mesh/
    └── tumor_4d/
        ├── series_manifest.json
        ├── series_summary.csv
        ├── phase_00/
        │   ├── manifest.json
        │   ├── geometry.json
        │   ├── contours.json
        │   ├── contour_report.csv
        │   ├── mask.npy
        │   ├── roundtrip_mask.npy
        │   ├── mesh_raw.npz
        │   ├── mesh_raw.ply
        │   ├── validation_report.json
        │   ├── validation_overview.png
        │   ├── ct_vis_hu.npy
        │   └── ct_vis_geometry.json
        ├── phase_10/
        ├── phase_20/
        ├── ...
        └── phase_90/
```

Per-phase artifacts:

| Artifact | Meaning |
|---|---|
| `manifest.json` | Phase contract linking patient, study, phase, ROI, CT series, and all generated files. |
| `geometry.json` | Full-resolution CT geometry in patient coordinates: shape, origin, orientation vectors, spacing, slice coordinates, and SOP UIDs. |
| `contours.json` | Serialized RTSTRUCT contours and their mapping to the CT grid. |
| `contour_report.csv` | Contour-to-slice rasterization diagnostics. |
| `mask.npy` | Full-resolution binary reference mask rasterized from RTSTRUCT. |
| `roundtrip_mask.npy` | Mask reconstructed by slicing and rasterizing the generated mesh. |
| `mesh_raw.npz` | Compressed mesh with `vertices_zyx`, `vertices_xyz`, and triangular `faces`. |
| `mesh_raw.ply` | ASCII PLY surface in DICOM patient coordinates for external 3-D viewers. |
| `validation_report.json` | Mask volume, mesh area and volume, bounds, watertightness, finite-vertex check, Dice, IoU, and relative volume error. |
| `validation_overview.png` | CT overlay for the first, largest, and last annotated axial slices. |
| `ct_vis_hu.npy` | Clipped and downsampled `int16` CT volume for responsive notebook visualization. |
| `ct_vis_geometry.json` | Patient-coordinate geometry matching `ct_vis_hu.npy`. |

Series-level artifacts:

| Artifact | Meaning |
|---|---|
| `series_manifest.json` | Patient, study, ROI template, phases, visualization settings, and relative paths to all phase manifests. |
| `series_summary.csv` | One row per phase with CT shape, visualization shape, mask volume, mesh size, round-trip Dice, and watertightness. |

Inspect the batch summary:

```bash
cat "$SERIES_DIR/series_summary.csv"
```

### 9.4 What the validation proves

For the worked case:

- all ten phases were created,
- each full CT grid was `147 × 512 × 512`,
- each visualization CT was `74 × 128 × 128` using stride `2 × 4 × 4`,
- every generated mesh was watertight,
- every round-trip Dice score was `1.0`,
- mask volume ranged from approximately `8.07 cm³` to `9.49 cm³`,
- mesh size ranged from `2176` to `2531` vertices.

The round trip checks only the technical conversion:

```text
RTSTRUCT mask → mesh → mask on the same voxel grid
```

`Dice = 1.0` means the generated mesh reproduces the source voxel mask exactly under the implemented rasterization. It does not prove that the clinical contour is correct, that MedGS4D predicts the tumor, or that meshes from different phases have vertex correspondence.

Each phase reference mesh was generated independently by marching cubes. Vertex count and indexing therefore vary between phases, so the reference sequence itself has no vertex correspondence. The Gaussian-to-mesh workflow instead preserves correspondence by deforming the phase-0 mesh while keeping its faces fixed.

## 10. MedGS-to-DICOM geometry and Gaussian-to-mesh transfer

The geometry workflow connects the learned MedGS4D deformation field with the RTSTRUCT tumor meshes. It has two stages:

```text
Stage A: MedGS latent coordinates ↔ DICOM patient coordinates
Stage B: Gaussian displacement → fixed-topology tumor mesh
```

No additional network is trained in either stage.

### 10.1 Coordinate semantics

For Gaussian \(i\), let the canonical latent center be:

$$\mathbf{c}_i^0=(x_i^0,z_i^0,m_i^0).$$

For respiratory phase \(p\), the dynamic model produces a phase-dependent latent center \(\mathbf{c}_i(p)\). In this representation:

- \(x\) and \(z\) are coordinates on the MedGS image plane,
- \(m\) is continuous renderer time and therefore continuous position in the CT slice stack,
- the upstream canonical camera maps \((x,z)\) to image row and column,
- cumulative MedGS time steps map \(m\) to a continuous slice index,
- DICOM volume geometry maps \((\text{slice},\text{row},\text{column})\) to patient coordinates in millimetres.

Let \(T\) denote the complete latent-to-patient transform. The patient-space displacement of Gaussian \(i\) at phase \(p\) is:

$$\mathbf{d}_i(p)=T\!\left(\mathbf{c}_i(p)\right)-T\!\left(\mathbf{c}_i^0\right).$$

Inspect the transform and Gaussian coverage:

```bash
GEOMETRY_DIR="$RUN_DIR/evaluation/gaussian_geometry/iter_007000"
FINAL_DYNAMIC_CHECKPOINT="$RUN_DIR/checkpoints/deformation_iter_007000.pth"

$PYTHON scripts/inspect_gaussian_geometry.py \
  --run-dir "$RUN_DIR" \
  --mesh-series-dir "$SERIES_DIR" \
  --checkpoint "$FINAL_DYNAMIC_CHECKPOINT" \
  --knn-k 16 \
  --device cuda
```

Main outputs:

```text
evaluation/gaussian_geometry/iter_007000/
├── summary.json
├── deformation_in_patient_mm.csv
├── mesh_vertex_knn.csv
├── mesh_vertex_knn_indices.npy
├── mesh_vertex_knn_distances_mm.npy
├── canonical_gaussian_sample.npz
├── tumor_neighborhood_gaussian_indices.npy
├── projection_overlay.png
└── tumor_neighborhood_overview.png
```

For the worked study:

- `611,563` canonical Gaussians were loaded,
- `99.20%` mapped inside the CT volume,
- mesh patient-coordinate round-trip error was `0 mm`,
- the nearest Gaussian was on average `2.01 mm` from a mesh vertex,
- the median distance to the 16th neighbor was `4.52 mm`,
- `1,092` unique Gaussians occurred in the tumor-vertex neighborhoods.

These checks validate the coordinate mapping and local geometric coverage. They do not prove that a nearby Gaussian represents tumor tissue.

### 10.2 Notation for robust transfer

Let $\mathbf{v}_j^0$ be vertex $j$ of the canonical phase-0 tumor mesh. Stage A stores the \(K\) nearest canonical Gaussians:

$$\mathcal{N}_j=\operatorname{KNN}_K(\mathbf{v}_j^0).$$

Let $o_i$ denote Gaussian opacity. The opacity confidence used during transfer is:

$$\omega_i=\max(o_i,\varepsilon)^q.$$

The notation $\operatorname{wmed}$ below means a component-wise weighted median of 3-D displacement vectors. It is robust to large individual displacement outliers, but it is not a learned estimator and is not rotation-invariant.

The symbols used in the equations are mapped to CLI and saved configuration names in Section 10.4.

### 10.3 Robust Gaussian-to-mesh transfer

#### Global coherent motion

The algorithm first estimates one dominant displacement for the entire Gaussian neighborhood of the tumor. An opacity-weighted component-wise median gives the initial center:

$$\mathbf{m}_G(p)=\operatorname{wmed}\!\left(\{\mathbf{d}_i(p)\},\{\omega_i\}\right).$$

For every tumor-neighborhood Gaussian, define its residual magnitude:

$$r_i(p)=\left\|\mathbf{d}_i(p)-\mathbf{m}_G(p)\right\|_2.$$

Let the median residual and median absolute deviation be:

$$\widetilde{r}(p)=\operatorname{median}_i r_i(p).$$

$$s_G(p)=\operatorname{median}_i\left|r_i(p)-\widetilde{r}(p)\right|.$$

The robust residual threshold is:

$$\theta_G(p)=\widetilde{r}(p)+\tau\max\!\left(1.4826\,s_G(p),\varepsilon\right).$$

The initial global inlier set is:

$$\mathcal{I}_G(p)=\{i:r_i(p)\leq\theta_G(p)\}.$$

If this set contains fewer than \(\lceil\rho N\rceil\) of the \(N\) tumor-neighborhood Gaussians, the Gaussians with the smallest residuals are retained until the minimum is reached.

The global tumor translation is then:

$$\mathbf{t}(p)=\operatorname{wmed}\!\left(\{\mathbf{d}_i(p):i\in\mathcal{I}_G(p)\},\{\omega_i:i\in\mathcal{I}_G(p)\}\right).$$

#### Local motion of one mesh vertex

For Gaussian $i\in\mathcal{N}_j$, let $\delta_{ji}$ be its canonical patient-space distance to vertex \(j\). The local distance scale is:

$$h_j=\operatorname{median}_{i\in\mathcal{N}_j}\delta_{ji}.$$

The spatial weight is:

$$a_{ji}=\exp\!\left[-\frac{1}{2}\left(\frac{\delta_{ji}}{h_j}\right)^2\right].$$

The combined distance-and-opacity weight is:

$$w_{ji}=a_{ji}\omega_i.$$

For local outlier detection, the component-wise median displacement is:

$$\mathbf{m}_j(p)=\operatorname{median}_{i\in\mathcal{N}_j}\mathbf{d}_i(p).$$

Define local residuals and their MAD:

$$r_{ji}(p)=\left\|\mathbf{d}_i(p)-\mathbf{m}_j(p)\right\|_2.$$

$$\widetilde{r}_j(p)=\operatorname{median}_{i\in\mathcal{N}_j}r_{ji}(p).$$

$$s_j(p)=\operatorname{median}_{i\in\mathcal{N}_j}\left|r_{ji}(p)-\widetilde{r}_j(p)\right|.$$

The local threshold is:

$$\theta_j(p)=\widetilde{r}_j(p)+\tau\max\!\left(1.4826\,s_j(p),\varepsilon\right).$$

The local inlier set combines the local threshold with the global coherent-motion mask:

$$\mathcal{I}_j(p)=\{i\in\mathcal{N}_j\cap\mathcal{I}_G(p):r_{ji}(p)\leq\theta_j(p)\}.$$

If fewer than $L$ neighbors remain, the implementation retains the best candidates according to local residual plus canonical vertex-to-Gaussian distance.

The robust local displacement of vertex $j$ is:

$$\mathbf{u}^{\mathrm{local}}_j(p)=\operatorname{wmed}\!\left(\{\mathbf{d}_i(p):i\in\mathcal{I}_j(p)\},\{w_{ji}:i\in\mathcal{I}_j(p)\}\right).$$

#### Global-local combination

The initial displacement of vertex \(j\) is a convex combination of the global tumor translation and the local estimate:

$$\mathbf{u}_j^{(0)}(p)=(1-\lambda)\mathbf{t}(p)+\lambda\mathbf{u}^{\mathrm{local}}_j(p).$$

For the worked run, \(\lambda=0.25\). The estimate therefore uses 75% global translation and 25% local displacement.

This form was introduced after a direct local weighted-mean baseline produced very large and unstable surface deformations. A purely global translation is stable but cannot change tumor shape. The global-local form is a robust training-free compromise, not a learned anatomical motion model.

#### Mesh-graph smoothing

Let \(\mathcal{A}_j\) be the one-ring mesh neighbors of vertex \(j\). For smoothing pass \(s\):

$$\mathbf{u}_j^{(s+1)}(p)=(1-\alpha)\mathbf{u}_j^{(s)}(p)+\frac{\alpha}{|\mathcal{A}_j|}\sum_{k\in\mathcal{A}_j}\mathbf{u}_k^{(s)}(p).$$

After each pass, the original mean displacement is restored. Smoothing therefore reduces local variation without changing the average tumor translation. After \(S\) passes, denote the smoothed displacement by \(\widetilde{\mathbf{u}}_j(p)\).

#### Final displacement bound

Each smoothed displacement is finally bounded by \(U_{\max}\):

$$\mathbf{u}_j(p)=\widetilde{\mathbf{u}}_j(p)\min\!\left(1,\frac{U_{\max}}{\max(\|\widetilde{\mathbf{u}}_j(p)\|_2,\varepsilon)}\right).$$

The predicted phase mesh is:

$$\widehat{\mathbf{v}}_j(p)=\mathbf{v}_j^0+\mathbf{u}_j(p).$$

Faces remain unchanged. All predicted phases therefore have the same topology and direct vertex correspondence.

### 10.4 Symbols and implementation parameters

| Symbol | CLI option | Saved key | Worked value | Meaning |
|---|---|---|---:|---|
| \(K\) | `--knn-k` in Stage A | `KnnK` | `16` | Canonical Gaussian neighbors stored per mesh vertex. |
| \(\tau\) | `--robust-z` | `robust_z` | `3.5` | Width of the global and local median/MAD thresholds. |
| \(\rho\) | `--global-minimum-fraction` | `global_minimum_fraction` | `0.25` | Minimum fraction of tumor-neighborhood Gaussians retained for the global estimate. |
| \(L\) | `--min-inliers` | `minimum_inliers` | `4` | Minimum retained Gaussian neighbors for one vertex. |
| \(q\) | `--opacity-power` | `opacity_power` | `1.0` | Exponent applied to opacity confidence. |
| \(\lambda\) | `--local-detail-weight` | `local_detail_weight` | `0.25` | Contribution of the local estimate in the global-local combination. |
| \(S\) | `--smoothing-iterations` | `smoothing_iterations` | `8` | Number of mesh-graph smoothing passes. |
| \(\alpha\) | `--smoothing-alpha` | `smoothing_alpha` | `0.35` | Neighbor contribution in one smoothing pass. |
| \(U_{\max}\) | `--max-displacement-mm` | `maximum_displacement_mm` | `20.0 mm` | Final hard bound on vertex displacement magnitude. |
| — | `--surface-samples` | `surface_samples` | `20000` | Surface samples used for distance metrics; does not affect prediction. |
| — | `--seed` | `seed` | `42` | Reproducible surface sampling during evaluation. |
| \(\varepsilon\) | fixed internally | — | small positive tolerance | Prevents division by zero and zero robust scale. |

### 10.5 Run transfer and evaluation

```bash
$PYTHON scripts/predict_mesh_from_gaussians.py \
  --run-dir "$RUN_DIR" \
  --mesh-series-dir "$SERIES_DIR" \
  --checkpoint "$FINAL_DYNAMIC_CHECKPOINT" \
  --geometry-dir "$GEOMETRY_DIR" \
  --surface-samples 20000 \
  --device cuda \
  --save-ply \
  --force
```

Default output:

```text
evaluation/mesh_transfer/iter_007000/
├── per_phase.csv
├── summary.json
├── predictions.npz
├── metrics.png
├── comparisons/
│   ├── phase_00.png
│   ├── phase_10.png
│   └── ...
└── meshes/
    ├── phase_00.ply
    ├── phase_10.ply
    └── ...
```

| Artifact | Meaning |
|---|---|
| `per_phase.csv` | Baseline, transferred, and improvement metrics for every respiratory phase. |
| `summary.json` | Configuration plus aggregates for all noncanonical, training, validation, and canonical splits. |
| `predictions.npz` | All predicted vertices, vertex displacements, phases, and common faces in one compact file. |
| `metrics.png` | Dice, HD95, centroid error, and transferred motion across phases. |
| `comparisons/phase_XX.png` | Canonical, RTSTRUCT reference, and Gaussian-transferred meshes. |
| `meshes/phase_XX.ply` | Optional predicted PLY meshes enabled by `--save-ply`. |

The canonical phase is checked as an invariant:

```text
vertex displacement = 0
Dice = 1
HD95 = 0 mm
centroid error = 0 mm
```

### 10.6 Worked result

The static mesh baseline reuses the phase-0 RTSTRUCT mesh without deformation. The transferred mesh uses the algorithm above. Phase 0 is excluded from noncanonical averages because it is exact by construction.

| Split | Phases | Dice static → transferred | HD95 [mm] static → transferred | Centroid error [mm] static → transferred | Signed volume error [%] static → transferred |
|---|---:|---:|---:|---:|---:|
| All noncanonical | 9 | `0.658 → 0.524` | `5.07 → 8.82` | `4.78 → 3.56` | `-6.4 → -38.9` |
| Training | 4 | `0.629 → 0.470` | `5.24 → 10.04` | `5.06 → 3.72` | `-4.7 → -44.7` |
| Holdout | 5 | `0.682 → 0.568` | `4.94 → 7.84` | `4.56 → 3.43` | `-7.8 → -34.2` |

The same pattern appears on training and holdout phases: the learned Gaussian field contains useful information about global tumor translation, but direct local interpolation degrades overlap, boundary accuracy, and volume preservation.

At the image-reconstruction level, MedGS4D improved mean noncanonical SSIM from `0.563` to `0.625` and reduced L1 from `0.1689` to `0.1671`; mean PSNR changed from `11.13 dB` to `11.09 dB`. Better image-space SSIM therefore did not automatically produce a reliable geometry-space deformation.

## 11. Notebook workflows

Use the repository's environment-backed kernel so that the editable package and its dependencies match the CLI.

### 11.1 Load a dynamic run

**What:** inspect completed metrics and training histories without manually reading every CSV.

**Why:** `load_run` centralizes paths and keeps notebook code thin.

```python
from pathlib import Path

from medgs4d.results import load_run, print_run_summary
from medgs4d.visualization import show_training_curves

RUN_DIR = Path(
    "/home/jovyan/shared/mtm_medgs_stack/results/medgs4d/"
    "patient117_20001024_annotated/"
    "phase00_canon15000_denoised_holdout_iter7000"
)

run = load_run(run_dir=RUN_DIR)
print_run_summary(run)
show_training_curves(run)
```

Refresh CSV-backed state during an active run:

```python
run = load_run(run_dir=RUN_DIR)
display(run.training_history.tail(20))
```

### 11.2 Browse reconstructed slices and breathing phases

```python
from medgs4d.visualization import (
    show_breathing_cycle_browser,
    show_error_map_browser,
    show_slice_browser,
)

show_slice_browser(run, device="cuda")
show_breathing_cycle_browser(run, device="cuda")
show_error_map_browser(run, device="cuda")
```

The slice browser compares:

```text
Ground truth | Canonical baseline | Dynamic reconstruction
```

The error browser adds baseline and dynamic absolute errors plus their local reduction.

Avoid loading a second full model in the notebook on the same GPU while training unless sufficient VRAM is available.

### 11.3 Load the RTSTRUCT reference series

**What:** load one validated phase through the same artifact contract used by the CLI.

**Why:** notebook visualization should not repeat RTSTRUCT parsing or mesh generation.

```python
from pathlib import Path

from medgs4d.mesh_validation import load_case

SERIES_DIR = Path(
    "/home/jovyan/shared/mtm_medgs_stack/data/tcia_4d_lung/prepared/"
    "patient117_20001024_annotated/rtstruct_mesh/tumor_4d"
)

CASE_DIR = SERIES_DIR / "phase_00"
case = load_case(CASE_DIR)

manifest = case["manifest"]
geometry = case["geometry"]
ct = case["ct_volume"]
mask = case["mask"]
roundtrip = case["roundtrip_mask"]
mesh = case["mesh"]
```

The notebook can then display:

- axial, coronal, and sagittal CT views,
- original RTSTRUCT contour overlays,
- source and round-trip masks,
- the reference tumor mesh,
- simple threshold-derived body, lung, and bone surfaces,
- a phase slider loading `phase_00` through `phase_90`.

Body, lung, and bone surfaces are visualization aids derived from simple HU thresholds. They are not clinical segmentations.

A round-trip XOR panel is useful only when masks differ. When `RoundtripDice = 1.0`, the XOR is empty and should be reported as no mismatch rather than displaying an arbitrary empty slice.

### 11.4 Demonstration notebook

`notebooks/04_validate_rtstruct_mesh.ipynb` is the end-to-end demonstration notebook for the reference geometry and Gaussian-to-mesh experiment.

It is organized so that the reader first sees the RTSTRUCT data without any predicted mesh:

- phase-0 axial and orthogonal CT views with the RTSTRUCT tumor,
- phase-specific RTSTRUCT reference meshes,
- CT-derived body, lung, and bone context.

The later sections introduce the Gaussian workflow:

- MedGS-to-DICOM projection overlays,
- Gaussian neighborhoods around the canonical tumor,
- image-reconstruction summaries for the canonical and dynamic models,
- all/train/holdout mesh-transfer metrics,
- phase sliders for RTSTRUCT and Gaussian-transferred contours,
- 3-D anatomy with reference and transferred tumor meshes,
- compact run and transfer configuration tables.

The notebook derives run paths from `SERIES_DIR` when exactly one evaluated transfer result is available. It reads saved PNG, CSV, JSON, and NPZ artifacts and does not rerun training or full evaluation.

## 12. CLI reference

### 12.1 `scripts/data.py`

| Subcommand | Required and notable parameters |
|---|---|
| `list-patients` | Exactly one of `--archives-dir`, `--dicom-dir`. |
| `extract` | `--archives-dir`, `--dicom-dir`, `--patient-id`; optional `--workers`, `--force`, `--dry-run`, `--log`. |
| `list-studies` | `--dicom-dir`, `--patient-id`. |
| `list-series` | `--dicom-dir`, `--patient-id`; optional `--study-uid`, `--modality`. |
| `list-rtstructs` | `--dicom-dir`, `--patient-id`; optional `--referenced-study-uid`. |
| `inspect` | `--dicom-dir`, `--patient-id`, `--study-uid`. |
| `prepare` | `--dicom-dir`, `--prepared-root`, `--patient-id`, `--study-uid`, `--study-name`; optional `--hu-window`, `--denoise-sigma`, `--force`. |

### 12.2 `scripts/train_canonical.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--data-dir` | required | Prepared study containing `manifest.json`. |
| `--medgs-repo` | required | Upstream MedGS repository. |
| `--output-root` | required | Canonical results root. |
| `--run-name` | required | Filesystem-safe run identifier. |
| `--canonical-phase` | required | Static reference phase. |
| `--representation` | `raw` | Prepared `raw` or `denoised` target. |
| `--iterations` | `30000` | Final upstream MedGS iteration. |
| `--poly-degree` | `2` | Upstream temporal polynomial degree. |
| `--batch-size` | `3` | Upstream MedGS batch size. |
| `--camera` | `mirror` | Upstream input/camera mode. |
| `--seed` | `42` | Reproducibility seed. |
| `--log-every` | `100` | Training-history cadence. |
| `--resume` | off | Continue the newest checkpoint. |
| `--force` | off | Delete and recreate the selected run. |

### 12.3 `scripts/train_medgs4d.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--data-dir` | required | Prepared study directory. |
| `--canonical-model` | required | Canonical run directory. |
| `--canonical-checkpoint` | run config | Exact canonical `chkpnt*.pth` to freeze. |
| `--medgs-repo` | required | Upstream MedGS repository. |
| `--output-root` | required | Dynamic results root. |
| `--run-name` | required | Filesystem-safe dynamic run identifier. |
| `--canonical-phase` | required | Must match the canonical run phase. |
| `--target-representation` | `raw` | Dynamic target: `raw` or `denoised`. |
| `--split-mode` | `full` | `full` or `phase-holdout`. |
| `--validation-phases` | empty | Entire phases assigned to validation. |
| `--iterations` | `7000` | Final dynamic iteration. |
| `--learning-rate` | `5e-4` | Adam learning rate. |
| `--spatial-frequencies` | `4` | Fourier frequencies for canonical `x`, `z`, and `m`. |
| `--phase-frequencies` | `2` | Cyclic Fourier frequencies for respiratory phase. |
| `--hidden-dim` | `256` | MLP hidden width. |
| `--hidden-layers` | `4` | Number of hidden layers. |
| `--chunk-size` | `131072` | Gaussians processed per network chunk. |
| `--checkpoint-every` | `250` | Checkpoint cadence. |
| `--log-every` | `25` | Console/history cadence. |
| `--validate-every` | `250` | Compact validation cadence; `0` disables it. |
| `--validation-samples` | `20` | Fixed compact validation subset size; `0` disables it. |
| `--seed` | `42` | Sampling, model, and diagnostic seed. |
| `--phase-jitter-std` | `0.0` | Initial standard deviation of cyclic phase jitter. |
| `--skip-final-evaluation` | off | Finish without automatic full evaluation. |
| `--resume` | off | Resume the selected run. |
| `--resume-checkpoint` | latest | Exact branch-point checkpoint. |
| `--force` | off | Delete and recreate the selected run. |
| `--device` | `cuda` | Torch device. |

Internal loss and stability defaults in `medgs4d/config.py`:

```text
max_gradient_norm    = 10.0
smoothness_gaussians = 65536
l1_weight            = 2.0
ssim_weight          = 0.25
magnitude_weight     = 1e-4
smoothness_weight    = 1e-3
```

These values are saved in `config.json` but are not currently CLI flags.

### 12.4 Evaluation and diagnostics

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

export_error_maps.py:
  --run-dir
  --phases PHASE [PHASE ...]
  --slices INDEX [INDEX ...]
  --checkpoint
  --error-max VALUE
  --device

visualize_medgs4d.py:
  --run-dir
  --slice INDEX
  exactly one of --phase PHASE or --all-phases
  --output PATH
  --device
```

### 12.5 `scripts/rtstruct_mesh.py`

| Subcommand | Parameters and purpose |
|---|---|
| `inventory` | `--dicom-dir`, `--patient-id`, `--study-uid`; optional `--csv`. Lists RTSTRUCT objects and ROI names. |
| `inspect-rois` | `--dicom-dir`, `--patient-id`, `--study-uid`; optional `--phase`, `--rtstruct-file`, `--csv`. Summarizes ROI contour geometry. |
| `build` | Common build parameters plus `--phase`, `--roi`, optional `--rtstruct-file`. Builds one standalone phase case. |
| `build-series` | Common build parameters plus `--phases` and `--roi-template`. Builds the complete phase series. |

Common build parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `--output-dir` | required | New output directory; existing output requires `--force`. |
| `--hu-window` | `-1000 400` | Clip range for the compact visualization CT. |
| `--roundtrip-dice-min` | `0.95` | Minimum accepted mask-to-mesh-to-mask Dice. |
| `--ct-vis-stride Z Y X` | `2 4 4` | Downsampling stride for `ct_vis_hu.npy`. |
| `--force` | off | Replace the selected output directory. |

### 12.6 `scripts/inspect_gaussian_geometry.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--run-dir` | required | Completed dynamic MedGS4D run. |
| `--mesh-series-dir` | required | RTSTRUCT reference mesh series. |
| `--checkpoint` | run default | Exact dynamic deformation checkpoint. |
| `--knn-k` | `16` | Gaussian neighbors stored per canonical mesh vertex. |
| `--device` | `cuda` | Torch device. |

The command validates latent-to-DICOM geometry, exports Gaussian coverage summaries, stores KNN neighborhoods, and writes projection and 3-D overview PNGs.

### 12.7 `scripts/predict_mesh_from_gaussians.py`

| Parameter | Default | Meaning |
|---|---:|---|
| `--run-dir` | required | Completed dynamic MedGS4D run. |
| `--mesh-series-dir` | required | RTSTRUCT reference mesh series. |
| `--checkpoint` | run default | Dynamic checkpoint used to predict Gaussian motion. |
| `--geometry-dir` | inferred | Stage-A geometry and KNN artifacts. |
| `--output-dir` | inferred | Mesh-transfer evaluation directory. |
| `--phases` | all study phases | Optional selected phase list. |
| `--robust-z` | `3.5` | Median/MAD inlier threshold width. |
| `--min-inliers` | `4` | Minimum local neighbors retained per vertex. |
| `--global-minimum-fraction` | `0.25` | Minimum Gaussian fraction for global translation. |
| `--opacity-power` | `1.0` | Opacity-confidence exponent. |
| `--local-detail-weight` | `0.25` | Local-motion contribution around global translation. |
| `--smoothing-iterations` | `8` | Mesh displacement smoothing passes. |
| `--smoothing-alpha` | `0.35` | Neighbor contribution per smoothing pass. |
| `--max-displacement-mm` | `20.0` | Final displacement magnitude bound. |
| `--surface-samples` | `20000` | Surface samples for distance metrics. |
| `--seed` | `42` | Reproducible evaluation sampling. |
| `--save-ply` | off | Save one predicted PLY per phase. |
| `--force` | off | Replace the selected output directory. |

## 13. Reproducibility, overwrite, and runtime rules

- `--force` is destructive for the selected output directory. Prefer a new run name for a new experiment.
- Canonical and dynamic configurations persist checkpoint identity and resolved paths. Evaluation should load those saved identities instead of silently choosing newer files.
- A dynamic run may intentionally freeze an older canonical checkpoint even when later canonical checkpoints exist.
- Full evaluation is the primary basis for model comparison; minibatch histories are optimization diagnostics.
- For temporal generalization, use `phase-holdout` and select checkpoints using held-out phases.
- Keep the upstream MedGS repository separate and explicit through `--medgs-repo`.
- Notebook visualization should read generated artifacts rather than perform data preparation or training.
- Avoid simultaneous full model copies on the same GPU unless VRAM headroom is known.

## 14. Improvement ideas

The current Gaussian-to-mesh result is a technically validated, training-free baseline. The main limitation is semantic: Euclidean proximity to the tumor mesh does not guarantee that a Gaussian represents tumor tissue. The following directions are more promising than further manual tuning on the worked study:

- **Geometry-supervised residual correction:** keep the robust Gaussian transfer as an initial estimate and learn only a small vertex-displacement correction from RTSTRUCT masks, signed-distance fields, or reference surfaces.
- **Semantic Gaussian selection:** estimate which Gaussians belong to the tumor or its boundary and exclude nearby Gaussians representing vessels, bronchi, lung parenchyma, or unrelated intensity edges.
- **Rigid or affine tumor motion before nonrigid deformation:** estimate translation, rotation, and optionally anisotropic scale at the tumor level, then predict only the remaining local deformation.
- **Volume-aware objectives:** penalize implausible shrinkage or expansion and regularize phase-to-phase tumor volume trajectories.
- **Surface-aware regularization:** replace or supplement post-hoc smoothing with Laplacian, edge-length, normal-consistency, or as-rigid-as-possible losses during learning.
- **Joint image and geometry supervision:** retain CT reconstruction losses while adding Dice, signed-distance, surface-distance, centroid, or landmark losses.
- **Temporal and cyclic consistency:** regularize vertex trajectories across neighboring respiratory phases and enforce continuity between the final and initial phase.
- **Geometry-space model selection:** select dynamic checkpoints using held-out mesh metrics rather than image metrics alone.
- **Multi-study validation:** repeat the experiment across patients, tumor sizes, locations, and respiratory motion amplitudes.

Further tuning of clipping, smoothing, or interpolation weights on this single study would risk fitting the heuristic to one example. The strongest next step is geometry-aware supervision evaluated on held-out phases and additional patients.
