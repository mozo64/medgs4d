from pathlib import Path

import pytest

from medgs4d.config import (
    DeformationConfig,
    MedGS4DConfig,
    SplitConfig,
    TrainingConfig,
)
from medgs4d.runs import (
    assert_resume_compatible,
    prepare_output_directory,
)


def make_config(run_name: str = "run1") -> MedGS4DConfig:
    return MedGS4DConfig(
        study_name="study1",
        run_name=run_name,
        data_dir="/data/study1",
        canonical_model_dir="/results/canonical/study1/canon",
        medgs_repository="/repo/MedGS",
        canonical_phase=20.0,
        split=SplitConfig(),
        deformation=DeformationConfig(),
        training=TrainingConfig(iterations=10),
    )


def test_existing_run_requires_resume_or_force(tmp_path: Path) -> None:
    run = tmp_path / "results" / "study" / "run"
    run.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        prepare_output_directory(run)


def test_force_removes_only_selected_run(tmp_path: Path) -> None:
    study = tmp_path / "results" / "study"
    selected = study / "selected"
    sibling = study / "sibling"
    selected.mkdir(parents=True)
    sibling.mkdir()
    (selected / "old.txt").write_text("old")
    prepare_output_directory(selected, force=True)
    assert selected.is_dir()
    assert not (selected / "old.txt").exists()
    assert sibling.is_dir()


def test_resume_rejects_changed_configuration() -> None:
    with pytest.raises(ValueError):
        assert_resume_compatible(make_config("first"), make_config("second"))
