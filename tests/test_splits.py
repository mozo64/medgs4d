import pandas as pd
import pytest

from medgs4d.splits import create_split_manifest


def index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PhasePercent": phase, "SliceIndex": slice_index, "SliceCoordinate": slice_index}
            for phase in (0, 10, 20, 30, 40)
            for slice_index in range(3)
        ]
    )


def test_full_split_uses_all_noncanonical_samples() -> None:
    split = create_split_manifest(index(), mode="full", canonical_phase=20)
    assert (split.loc[split.PhasePercent != 20, "Split"] == "train").all()
    assert (split.loc[split.PhasePercent == 20, "Split"] == "canonical").all()


def test_phase_holdout_excludes_validation_phases_from_training() -> None:
    split = create_split_manifest(
        index(),
        mode="phase-holdout",
        canonical_phase=20,
        validation_phases=(10, 30),
    )
    assert set(split.loc[split.Split == "validation", "PhasePercent"]) == {10, 30}
    assert not set(split.loc[split.Split == "train", "PhasePercent"]) & {10, 30}


def test_canonical_phase_is_not_validation() -> None:
    with pytest.raises(ValueError):
        create_split_manifest(
            index(),
            mode="phase-holdout",
            canonical_phase=20,
            validation_phases=(20,),
        )
