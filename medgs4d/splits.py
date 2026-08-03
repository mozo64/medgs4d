from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .runs import write_dataframe


SplitName = Literal["train", "validation", "canonical", "all"]


def create_split_manifest(
    phase_slice_index: pd.DataFrame,
    *,
    mode: Literal["full", "phase-holdout"],
    canonical_phase: float,
    validation_phases: Sequence[float] = (),
) -> pd.DataFrame:
    """Assign every phase-slice pair to train, validation, or canonical."""

    frame = phase_slice_index.copy()
    phases = frame["PhasePercent"].astype(float).to_numpy()
    validation = np.asarray([float(value) for value in validation_phases])
    if validation.size and np.any(np.isclose(validation, canonical_phase)):
        raise ValueError("canonical phase cannot be a validation phase")

    frame["Split"] = "train"
    frame.loc[np.isclose(phases, canonical_phase), "Split"] = "canonical"
    if mode == "phase-holdout":
        for phase in validation:
            frame.loc[np.isclose(phases, phase), "Split"] = "validation"
    elif mode != "full":
        raise ValueError(f"Unknown split mode: {mode}")

    validate_split_manifest(
        frame,
        available_phases=sorted(frame["PhasePercent"].astype(float).unique()),
        canonical_phase=canonical_phase,
    )
    return frame


def validate_split_manifest(
    split_manifest: pd.DataFrame,
    *,
    available_phases: Sequence[float],
    canonical_phase: float,
) -> None:
    """Validate split coverage, exclusivity, and phase assignments."""

    required = {"PhasePercent", "SliceIndex", "Split"}
    missing = required - set(split_manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if split_manifest.duplicated(["PhasePercent", "SliceIndex"]).any():
        raise ValueError("Split manifest contains duplicate phase-slice rows")
    if not set(split_manifest["Split"]).issubset(
        {"train", "validation", "canonical"}
    ):
        raise ValueError("Split manifest contains an unknown split label")
    canonical_rows = split_manifest.loc[
        np.isclose(split_manifest["PhasePercent"].astype(float), canonical_phase)
    ]
    if canonical_rows.empty or not canonical_rows["Split"].eq("canonical").all():
        raise ValueError("Canonical phase must be assigned only to canonical")
    observed = sorted(split_manifest["PhasePercent"].astype(float).unique())
    if len(observed) != len(available_phases) or not np.allclose(
        observed, sorted(float(value) for value in available_phases)
    ):
        raise ValueError("Split manifest does not cover all available phases")
    validation_phases = split_manifest.loc[
        split_manifest["Split"] == "validation", "PhasePercent"
    ].unique()
    for phase in validation_phases:
        labels = split_manifest.loc[
            np.isclose(split_manifest["PhasePercent"].astype(float), phase), "Split"
        ]
        if not labels.eq("validation").all():
            raise ValueError("Phase holdout must assign the complete phase")


def get_split_rows(
    split_manifest: pd.DataFrame,
    split: SplitName,
) -> pd.DataFrame:
    """Return rows belonging to the requested evaluation or training split."""

    if split == "all":
        return split_manifest.copy().reset_index(drop=True)
    return split_manifest.loc[split_manifest["Split"] == split].reset_index(drop=True)


def save_split_manifest(split_manifest: pd.DataFrame, path: Path) -> None:
    """Save the exact split used by one run."""

    write_dataframe(path, split_manifest)


def load_split_manifest(path: Path) -> pd.DataFrame:
    """Load a previously saved train-validation split."""

    return pd.read_csv(path)
