from pathlib import Path

import numpy as np
import pandas as pd

from medgs4d.config import (
    DeformationConfig,
    MedGS4DConfig,
    SplitConfig,
    TrainingConfig,
    save_config,
)
from medgs4d.data import StudyManifest, save_study_manifest
from medgs4d.results import load_run, resolve_run_dir


def make_run(tmp_path: Path, with_metrics: bool) -> Path:
    study_dir = tmp_path / "prepared" / "study1"
    study_dir.mkdir(parents=True)
    phase_slice = study_dir / "phase_slice_manifest.csv"
    pd.DataFrame(
        {"PhasePercent": [0, 20], "SliceIndex": [0, 0], "SliceCoordinate": [0, 0]}
    ).to_csv(phase_slice, index=False)
    summary = study_dir / "phase_summary.csv"
    pd.DataFrame({"PhasePercent": [0, 20]}).to_csv(summary, index=False)
    for phase in (0.0, 20.0):
        path = study_dir / f"phase_{phase:g}.npy"
        np.save(path, np.zeros((1, 2, 2), dtype=np.float32))
    manifest = StudyManifest(
        study_name="study1",
        patient_id="p1",
        study_instance_uid="uid",
        phases=(0.0, 20.0),
        slice_count=1,
        volume_shape=(1, 2, 2),
        hu_window=(-1000, 400),
        denoise_sigma=(0.2, 0.4, 0.4),
        raw_volume_paths={0.0: str(study_dir / "phase_0.npy"), 20.0: str(study_dir / "phase_20.npy")},
        denoised_volume_paths={0.0: str(study_dir / "phase_0.npy"), 20.0: str(study_dir / "phase_20.npy")},
        phase_slice_manifest_path=str(phase_slice),
        phase_summary_path=str(summary),
    )
    save_study_manifest(manifest, study_dir)

    run_dir = tmp_path / "results" / "study1" / "run1"
    (run_dir / "evaluation").mkdir(parents=True)
    save_config(
        MedGS4DConfig(
            study_name="study1",
            run_name="run1",
            data_dir=str(study_dir),
            canonical_model_dir="/canonical",
            medgs_repository="/MedGS",
            canonical_phase=20,
            split=SplitConfig(),
            deformation=DeformationConfig(),
            training=TrainingConfig(iterations=10),
        ),
        run_dir / "config.json",
    )
    pd.DataFrame(
        {"PhasePercent": [0, 20], "SliceIndex": [0, 0], "Split": ["train", "canonical"]}
    ).to_csv(run_dir / "split_manifest.csv", index=False)
    pd.DataFrame({"Iteration": [1], "L1": [0.1], "PSNR": [20], "SSIM": [0.8]}).to_csv(
        run_dir / "training_history.csv", index=False
    )
    if with_metrics:
        pd.DataFrame({"Split": ["train"], "DynamicPSNR": [21]}).to_csv(
            run_dir / "evaluation" / "per_slice.csv", index=False
        )
        pd.DataFrame({"Split": ["train"], "PhasePercent": [0]}).to_csv(
            run_dir / "evaluation" / "per_phase.csv", index=False
        )
        (run_dir / "evaluation" / "overall.json").write_text('{"All": {}}')
    return run_dir


def test_load_run_without_evaluation(tmp_path: Path) -> None:
    run = load_run(run_dir=make_run(tmp_path, False))
    assert run.per_slice_metrics is None
    assert run.config.run_name == "run1"


def test_load_run_with_complete_metrics(tmp_path: Path) -> None:
    run = load_run(run_dir=make_run(tmp_path, True))
    assert run.per_slice_metrics is not None
    assert run.overall_metrics == {"All": {}}


def test_resolve_run_by_path_or_names(tmp_path: Path) -> None:
    direct = make_run(tmp_path, False)
    by_path = resolve_run_dir(run_dir=direct)
    by_names = resolve_run_dir(
        results_root=tmp_path / "results", study_name="study1", run_name="run1"
    )
    assert by_path == by_names
