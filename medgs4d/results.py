from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd

from .canonical import CanonicalAssets, load_frozen_canonical
from .config import MedGS4DConfig, load_medgs4d_config
from .data import StudyManifest, load_study_manifest
from .deformation import DeformationField
from .runs import find_latest_checkpoint
from .training import load_checkpoint


@dataclass
class RunResults:
    """Provide notebook-friendly access to one self-contained MedGS4D run."""

    run_dir: Path
    config: MedGS4DConfig
    study: StudyManifest
    split_manifest: pd.DataFrame
    training_history: pd.DataFrame
    validation_history: pd.DataFrame | None
    per_slice_metrics: pd.DataFrame | None
    per_phase_metrics: pd.DataFrame | None
    overall_metrics: dict[str, Any] | None

    @property
    def checkpoint(self) -> Path | None:
        """Return the latest checkpoint currently available in the run."""

        return find_latest_checkpoint(self.run_dir / "checkpoints")


def list_runs(
    results_root: Path,
    *,
    study_name: str | None = None,
) -> pd.DataFrame:
    """List available runs and their completion and evaluation status."""

    roots = [results_root / study_name] if study_name else sorted(results_root.glob("*"))
    rows = []
    for study_root in roots:
        if not study_root.is_dir():
            continue
        for run_dir in sorted(path for path in study_root.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            if not config_path.is_file():
                continue
            config = load_medgs4d_config(config_path)
            checkpoint = find_latest_checkpoint(run_dir / "checkpoints")
            rows.append(
                {
                    "StudyName": config.study_name,
                    "RunName": config.run_name,
                    "SplitMode": config.split.mode,
                    "Iterations": config.training.iterations,
                    "CanonicalCheckpointIteration": (
                        config.canonical_checkpoint_iteration
                    ),
                    "Checkpoint": str(checkpoint) if checkpoint else "",
                    "Complete": (run_dir / "completion.json").is_file(),
                    "Evaluated": (run_dir / "evaluation" / "overall.json").is_file(),
                    "RunDirectory": str(run_dir),
                }
            )
    return pd.DataFrame(rows)


def resolve_run_dir(
    *,
    run_dir: Path | None = None,
    results_root: Path | None = None,
    study_name: str | None = None,
    run_name: str | None = None,
) -> Path:
    """Resolve a run either by direct path or by study and run names."""

    if run_dir is not None:
        resolved = run_dir.resolve()
    else:
        if results_root is None or study_name is None or run_name is None:
            raise ValueError(
                "Provide run_dir or results_root together with study_name and run_name"
            )
        resolved = (results_root / study_name / run_name).resolve()
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"MedGS4D run not found: {resolved}")
    return resolved


def _optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def load_run(
    *,
    run_dir: Path | None = None,
    results_root: Path | None = None,
    study_name: str | None = None,
    run_name: str | None = None,
) -> RunResults:
    """Load configuration, manifests, history, and available metrics for one run."""

    resolved = resolve_run_dir(
        run_dir=run_dir,
        results_root=results_root,
        study_name=study_name,
        run_name=run_name,
    )
    config = load_medgs4d_config(resolved / "config.json")
    study = load_study_manifest(Path(config.data_dir))
    training_history = _optional_csv(resolved / "training_history.csv")
    return RunResults(
        run_dir=resolved,
        config=config,
        study=study,
        split_manifest=pd.read_csv(resolved / "split_manifest.csv"),
        training_history=(
            training_history if training_history is not None else pd.DataFrame()
        ),
        validation_history=_optional_csv(resolved / "validation_history.csv"),
        per_slice_metrics=_optional_csv(resolved / "evaluation" / "per_slice.csv"),
        per_phase_metrics=_optional_csv(resolved / "evaluation" / "per_phase.csv"),
        overall_metrics=(
            json.loads(
                (resolved / "evaluation" / "overall.json").read_text(encoding="utf-8")
            )
            if (resolved / "evaluation" / "overall.json").is_file()
            else None
        ),
    )


def load_run_models(
    run: RunResults,
    *,
    checkpoint: Path | None = None,
    device: str = "cuda",
) -> tuple[CanonicalAssets, DeformationField]:
    """Load the canonical model and deformation network required for rendering."""

    canonical = load_frozen_canonical(
        Path(run.config.canonical_model_dir),
        Path(run.config.medgs_repository),
        checkpoint=(
            Path(run.config.canonical_checkpoint)
            if run.config.canonical_checkpoint
            else None
        ),
        device=device,
    )
    field = DeformationField(
        canonical,
        run.config.deformation,
        run.config.canonical_phase,
        seed=run.config.training.seed,
    )
    checkpoint_path = checkpoint or run.checkpoint
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run.run_dir}")
    load_checkpoint(
        checkpoint_path,
        field,
        device=device,
        config=run.config,
    )
    field.model.eval()
    return canonical, field


def print_run_summary(run: RunResults) -> None:
    """Print run configuration, split sizes, progress, and headline metrics."""

    config = run.config
    split_counts = run.split_manifest["Split"].value_counts().to_dict()
    final_iteration = (
        int(run.training_history["Iteration"].max())
        if not run.training_history.empty
        else 0
    )
    print(f"Study:           {config.study_name}")
    print(f"Run:             {config.run_name}")
    print(f"Canonical phase: {config.canonical_phase:g}%")
    print(
        "Canonical ckpt:  "
        f"{config.canonical_checkpoint_iteration} "
        f"({config.canonical_checkpoint or 'run default'})"
    )
    print(f"Split:           {config.split.mode}")
    print(f"Samples:         {split_counts}")
    print(f"Progress:        {final_iteration:,}/{config.training.iterations:,}")
    print(f"Checkpoint:      {run.checkpoint}")
    if run.overall_metrics:
        metrics = run.overall_metrics.get("AllNoncanonical", {})
        if metrics:
            print(
                "Final metrics:   "
                f"L1={metrics.get('DynamicL1', float('nan')):.6f}, "
                f"PSNR={metrics.get('DynamicPSNR', float('nan')):.4f}, "
                f"SSIM={metrics.get('DynamicSSIM', float('nan')):.6f}"
            )
