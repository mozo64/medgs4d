from pathlib import Path

import pandas as pd

from medgs4d.config import (
    DeformationConfig,
    MedGS4DConfig,
    SplitConfig,
    TrainingConfig,
    save_config,
)
from medgs4d.reporting import generate_run_report


def test_report_pdf_and_metrics_are_created(tmp_path: Path) -> None:
    run = tmp_path / "run"
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    save_config(
        MedGS4DConfig(
            study_name="study",
            run_name="run",
            data_dir="/data",
            canonical_model_dir="/canonical",
            medgs_repository="/MedGS",
            canonical_phase=20,
            split=SplitConfig(mode="phase-holdout", validation_phases=(10, 30)),
            deformation=DeformationConfig(),
            training=TrainingConfig(iterations=100),
        ),
        run / "config.json",
    )
    rows = []
    for split, phase in (("train", 0), ("validation", 10)):
        rows.append(
            {
                "Split": split,
                "PhasePercent": phase,
                "SliceIndex": 0,
                "BaselineL1": 0.02,
                "DynamicL1": 0.01,
                "L1Reduction": 0.01,
                "BaselinePSNR": 28,
                "DynamicPSNR": 30,
                "PSNRGain": 2,
                "BaselineSSIM": 0.78,
                "DynamicSSIM": 0.80,
                "SSIMGain": 0.02,
            }
        )
    pd.DataFrame(rows).to_csv(evaluation / "per_slice.csv", index=False)
    pd.DataFrame(rows).to_csv(evaluation / "per_phase.csv", index=False)
    pd.DataFrame(
        {
            "Iteration": [1, 50, 100],
            "L1": [0.03, 0.02, 0.01],
            "PSNR": [25, 28, 30],
            "SSIM": [0.7, 0.76, 0.8],
        }
    ).to_csv(run / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "Iteration": [50, 100],
            "L1": [0.025, 0.015],
            "PSNR": [27, 29],
            "SSIM": [0.74, 0.78],
        }
    ).to_csv(run / "validation_history.csv", index=False)
    output = generate_run_report(run)
    assert output.is_file() and output.stat().st_size > 0
    assert (run / "report_metrics.csv").is_file()
