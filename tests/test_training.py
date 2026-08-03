from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medgs4d.training import (
    build_sampling_plan,
    load_training_history,
    reconcile_training_history,
)


def train_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PhasePercent": phase, "SliceIndex": slice_index}
            for phase in (0, 40, 60, 80)
            for slice_index in range(5)
        ]
    )


def test_sampling_plan_is_deterministic() -> None:
    first = build_sampling_plan(
        train_rows(), iterations=50, seed=42, canonical_phase=20
    )
    second = build_sampling_plan(
        train_rows(), iterations=50, seed=42, canonical_phase=20
    )
    pd.testing.assert_frame_equal(first, second)


def test_sampling_plan_contains_only_train_rows() -> None:
    rows = train_rows()
    plan = build_sampling_plan(rows, iterations=100, seed=4, canonical_phase=20)
    pairs = set(zip(rows.PhasePercent, rows.SliceIndex))
    assert all(
        (phase, slice_index) in pairs
        for phase, slice_index in zip(plan.PhasePercent, plan.SliceIndex)
    )


def test_empty_history_does_not_block_resume(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    path.write_text("")
    assert load_training_history(path).empty


def test_history_is_trimmed_to_checkpoint() -> None:
    history = pd.DataFrame({"Iteration": [1, 2, 3, 3, 4], "L1": range(5)})
    reconciled = reconcile_training_history(history, 3)
    assert reconciled.Iteration.tolist() == [1, 2, 3]
    assert reconciled.iloc[-1].L1 == 3


def test_phase_jitter_does_not_change_sampling_sequence() -> None:
    plain = build_sampling_plan(
        train_rows(), iterations=60, seed=42, canonical_phase=20
    )
    jittered = build_sampling_plan(
        train_rows(),
        iterations=60,
        seed=42,
        canonical_phase=20,
        phase_jitter_initial_std=0.02,
    )
    columns = ["PhasePercent", "SliceIndex", "NeighborPhasePercent"]
    pd.testing.assert_frame_equal(plain[columns], jittered[columns])
