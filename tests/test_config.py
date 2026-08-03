from pathlib import Path

from medgs4d.config import (
    DeformationConfig,
    MedGS4DConfig,
    SplitConfig,
    TrainingConfig,
    load_medgs4d_config,
    save_config,
)
from medgs4d.runs import assert_resume_compatible


def test_medgs4d_config_round_trip_preserves_tuple_fields(tmp_path: Path) -> None:
    config = MedGS4DConfig(
        study_name="study",
        run_name="run",
        data_dir="/data/study",
        canonical_model_dir="/results/canonical",
        medgs_repository="/repo/MedGS",
        canonical_phase=20,
        split=SplitConfig(mode="phase-holdout", validation_phases=(10, 30, 50)),
        deformation=DeformationConfig(),
        training=TrainingConfig(iterations=100),
    )
    path = tmp_path / "config.json"
    save_config(config, path)
    loaded = load_medgs4d_config(path)
    assert loaded == config
    assert_resume_compatible(loaded, config)
