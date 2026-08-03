# MedGS4D

A small research codebase for phase-conditioned reconstruction of respiratory
4D-CT with a frozen canonical [MedGS](https://github.com/gmum/MedGS) model and
a trainable deformation MLP.

The long development notebooks are not part of the execution pipeline. Data
preparation, canonical training, dynamic training, evaluation, reporting, and
static visualization are exposed through command-line scripts. The included
notebook only reads completed runs.

## Repository layout

```text
medgs4d/
├── medgs4d/                 reusable Python package
│   ├── config.py
│   ├── runs.py
│   ├── data.py
│   ├── splits.py
│   ├── canonical.py
│   ├── deformation.py
│   ├── training.py
│   ├── evaluation.py
│   ├── reporting.py
│   ├── results.py
│   └── visualization.py
├── scripts/                 thin command-line entry points
├── notebooks/
│   └── results_browser.ipynb
├── tests/
└── pyproject.toml
```

The upstream MedGS repository is cloned separately and passed through
`--medgs-repo`. MedGS4D does not vendor or modify its renderer.

## Data and output contracts

Prepared study:

```text
<prepared-root>/<study-name>/
├── manifest.json
├── phase_summary.csv
├── phase_slice_manifest.csv
└── volumes/
    ├── raw/phase_*.npy
    └── denoised/phase_*.npy
```

Canonical run:

```text
<canonical-output-root>/<study-name>/<run-name>/
├── config.json
├── canonical_run.json
├── frame_manifest.csv
├── dataset/{original,mirror}/
└── model/
```

Dynamic run:

```text
<results-root>/<study-name>/<run-name>/
├── config.json
├── split_manifest.csv
├── sampling_plan.csv
├── training_history.csv
├── validation_history.csv       optional
├── training_summary.csv
├── completion.json
├── checkpoints/
├── evaluation/
│   ├── per_slice.csv
│   ├── per_phase.csv
│   └── overall.json
├── report_metrics.csv
├── report.pdf
└── visualizations/
```

Existing study and run directories are never overwritten implicitly. Training
requires a new `--run-name`, `--resume`, or explicit `--force`.

## Command groups

```text
python scripts/data.py list-patients ...
python scripts/data.py extract ...
python scripts/data.py list-studies ...
python scripts/data.py inspect ...
python scripts/data.py prepare ...

python scripts/train_canonical.py ...
python scripts/evaluate_canonical.py ...
python scripts/train_medgs4d.py ...
python scripts/evaluate_medgs4d.py ...
python scripts/visualize_medgs4d.py ...
```

`train_medgs4d.py` supports `full` and `phase-holdout` splits. Its architecture
and optimization parameters are explicit CLI arguments rather than historical
notebook experiment names. A completed training run performs final evaluation
and writes a minimalist vector `report.pdf` unless
`--skip-final-evaluation` is supplied.

## Tests

Fast tests do not require MedGS, CUDA, or real DICOM files:

```bash
python -m pytest -q
```

The optional WORF integration test loads a real run and renders one image:

```bash
MEDGS4D_TEST_RUN_DIR=/absolute/path/to/run \
python -m pytest -q -m worf
```

The MedGS Python/CUDA environment must already contain PyTorch and the compiled
MedGS rasterizer extensions.
