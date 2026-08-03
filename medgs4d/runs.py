from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import re
import shutil

import pandas as pd

from .config import MedGS4DConfig, config_to_dict


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunPaths:
    """Collect all standard paths belonging to one MedGS4D run."""

    root: Path
    checkpoints: Path
    evaluation: Path
    visualizations: Path
    config: Path
    split_manifest: Path
    sampling_plan: Path
    training_history: Path
    validation_history: Path
    training_summary: Path
    completion: Path
    smoothness_indices: Path
    report: Path
    report_metrics: Path


def validate_name(value: str, kind: str) -> str:
    """Validate a filesystem-safe study or run name."""

    if not SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"Invalid {kind} {value!r}; use letters, digits, '.', '_' or '-'."
        )
    return value


def build_study_dir(prepared_root: Path, study_name: str) -> Path:
    """Return the prepared-data directory for a named study."""

    return prepared_root / validate_name(study_name, "study name")


def build_run_paths(
    results_root: Path,
    study_name: str,
    run_name: str,
) -> RunPaths:
    """Build the standard output paths for one named MedGS4D run."""

    root = (
        results_root
        / validate_name(study_name, "study name")
        / validate_name(run_name, "run name")
    )
    return RunPaths(
        root=root,
        checkpoints=root / "checkpoints",
        evaluation=root / "evaluation",
        visualizations=root / "visualizations",
        config=root / "config.json",
        split_manifest=root / "split_manifest.csv",
        sampling_plan=root / "sampling_plan.csv",
        training_history=root / "training_history.csv",
        validation_history=root / "validation_history.csv",
        training_summary=root / "training_summary.csv",
        completion=root / "completion.json",
        smoothness_indices=root / "smoothness_gaussian_indices.npy",
        report=root / "report.pdf",
        report_metrics=root / "report_metrics.csv",
    )


def prepare_output_directory(
    path: Path,
    *,
    force: bool = False,
    resume: bool = False,
) -> None:
    """Create a new output directory or explicitly replace or resume it."""

    if force and resume:
        raise ValueError("--force and --resume are mutually exclusive")
    if path.exists():
        if force:
            remove_output_directory(path)
        elif not resume:
            raise FileExistsError(
                f"Output directory already exists: {path}\n"
                "Use --resume to continue or --force to replace it."
            )
    elif resume:
        raise FileNotFoundError(f"Cannot resume missing output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def remove_output_directory(path: Path) -> None:
    """Remove one exact run directory after conservative safety checks."""

    resolved = path.resolve()
    if resolved == resolved.parent or len(resolved.parts) < 4:
        raise ValueError(f"Refusing to remove unsafe path: {resolved}")
    shutil.rmtree(resolved)


def assert_resume_compatible(
    saved_config: MedGS4DConfig,
    requested_config: MedGS4DConfig,
) -> None:
    """Reject resume changes except for the requested target iteration."""

    saved = config_to_dict(saved_config)
    requested = config_to_dict(requested_config)

    saved_iterations = int(saved["training"].pop("iterations"))
    requested_iterations = int(requested["training"].pop("iterations"))

    if saved != requested:
        raise ValueError(
            "Requested configuration differs from the saved run configuration. "
            "Only training.iterations may change during resume."
        )
    if requested_iterations <= 0:
        raise ValueError("Requested training iterations must be positive")
    if requested_iterations == saved_iterations:
        return


def find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the latest valid deformation checkpoint in a run."""

    latest = checkpoints_dir / "deformation_latest.pth"
    if latest.is_file():
        return latest
    checkpoints = sorted(checkpoints_dir.glob("deformation_iter_*.pth"))
    return checkpoints[-1] if checkpoints else None


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON metadata atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV table atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
