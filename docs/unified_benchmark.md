# Unified MedGS4D benchmark

## Layout and review base

Original MedGS lives at the repository root, with no nested `medgs4d/` package.
New modules extend its shared structure:

| Module | Responsibility |
| --- | --- |
| `arguments/temporal.py` | Canonical and temporal run configuration |
| `scene/volume_data.py`, `scene/temporal_splits.py` | 4D CT loading and train/evaluation splits |
| `scene/canonical.py` | Prepare, train and load the shared static MedGS representation |
| `models/deformation.py`, `models/trajectory.py`, `models/transport.py` | Temporal neural models and Gaussian views |
| `training/temporal.py`, `training/methods.py` | Temporal optimization and method objectives |
| `evaluate_temporal.py` | Temporal slice evaluation and checkpoint comparisons |
| `utils/run_utils.py`, `utils/temporal_reporting.py` | Run files and reporting |
| `benchmark/` | Six-method experiment orchestration, source record and volume metrics |

The static and temporal training algorithms differ, so they remain distinct modules.
Both use the original Gaussian model, renderer and reconstruction losses. Temporal
rendering passes a deformed view of the same Gaussian primitives to the shared
renderer. There is no copied MedGS tree, generated source or runtime monkeypatch.
Only `notebooks/09_MedGS4D_publication_benchmark.ipynb` is tracked.

Upstream baseline: gmum/MedGS e200ec263e4ae0677e9d6f99ebf6e58195b48c16.
The user's imported baseline commit is d63ae18. Review with:

```bash
git diff d63ae18 --stat
git diff d63ae18 -- train.py utils/general_utils.py utils/training_history.py models/scenes/dataset_readers.py
```

Shared upstream changes are intentionally limited:
- Explicit `--seed`, preserving upstream default 0; the benchmark passes 42.
- Optional CSV history callback in image training. The existing logging fields
  and insertion point are preserved, with no generated or patched train script.
- `np.uint8` instead of `np.byte` in one dataset reader, matching the WORF snapshot.
- Packaging, dependency declarations and development ignore entries.
- Deferred CUDA imports in `models/__init__.py` and `scene/__init__.py`, so
  configuration and CT data processing can be imported without a CUDA renderer.
  Existing `gaussianModel`, `gaussianModelRender` and `Scene` APIs are preserved.

The original img/seg/joint paths remain. This integration does not add temporal
mask/mesh workflows or import the earlier experimental notebooks.

## WORF setup

The reference October 1 environment was Python 3.11.5, PyTorch
2.4.1+cu124 and NumPy 1.26.4. The notebook records actual versions without
requiring these exact versions or a fixed interpreter location. The existing
`medgs-worf` environment is suitable for the first comparison run.
From the unified repository root on WORF:

```bash
git submodule update --init --recursive
/home/jovyan/shared/mtm_medgs_stack/envs/medgs-worf/bin/python -m pip install --no-deps -e .
```

This is an editable checkout: the shared top-level `train.py` is deliberately
executed in place. A standalone wheel deployment is not part of this workflow.
The command does not upgrade PyTorch or CUDA dependencies. Existing compiled
CUDA extensions must correspond to the pinned submodule sources. If those
extensions are not installed, build them in this same environment using the
upstream installation instructions. CPU tests do not validate compiled CUDA.

Restart the notebook kernel after replacing the old editable package. Set
`REPO_ROOT` in the notebook to this checkout's absolute path **on WORF**. The
notebook checks that its imported metric module comes from that checkout.
Workers run `python -m benchmark.worker` with the checkout as their working
directory; inherited `PYTHONPATH` overrides are removed. No second repository
path is configured.

## First run and full run

The default is `SMOKE = True`: series `119_0`, all six methods, 50 canonical
iterations and 12 temporal iterations. Shortened schedules activate the cycle
and consistency stages. Checkpoints are saved every four temporal iterations.
The smoke run also reloads the final checkpoints and renders all four hidden
phases before evaluation. Its scores are not publication results.

Set `SMOKE = False` for the original 14-series protocol: 15,000 canonical and
7,000 temporal iterations with unchanged full-run hyperparameters. The smoke
and full experiment IDs are separate and differ from the October 1 experiment.
Use another ID for changed code or settings. `RESUME=True` is only for the same
manifest and configuration. Source files are hashed before workers start;
editing code during a run is unsupported. Canonical resume retains the original
upstream sampler limitation (not an exact RNG continuation).

## Preserved numerical behavior

The canonical model is frozen during temporal training. The six method IDs,
endpoint-only fitting, axis order, uint8 training-target conversion, clamp/mean
prediction transform and original 3D PSNR/NCC/SSIM/NMSE definitions are retained.
Intermediate CT ground truth is first read in the evaluation stage. Preparation
loads only frame0/frame5. Trajectory methods still support only phases 0–50%.

Method selection is explicit in `MedGS4DConfig.method` and `method_options`.
Training and inference use `create_field`; no global class or function is
replaced. New checkpoints store and validate method identity and options.
This notebook starts new runs; it is not a migration tool for legacy result
folders with absolute paths and benchmark-specific metadata.

## Validation

CPU checks cover all six methods using a small differentiable substitute for
the CUDA renderer: finite losses, active auxiliary losses, checkpoint reload,
exact continuation from an earlier checkpoint, method mismatch rejection and
absence of global mutation. Additional tests cover geometry anchoring, sampling,
configuration, source verification, notebook syntax and metric identities.

This revision is checked against the preceding integration using the same CPU
fixture for all six methods, comparing losses and final primary model weights.
CPU tests do not establish CUDA parity. Actual CUDA rendering, original static
img/seg/joint execution and full-volume reproducibility still need testing on WORF.

The curvature finite-difference step must be smaller than 0.05: probes are
sampled at u in [0.05, 0.95], and u +/- step must remain inside the supported
[0, 1] trajectory interval. This is an algorithm constraint, not an environment
restriction. Short runs may end before the auxiliary-loss ramp completes.

```bash
python -m pytest -q
```
