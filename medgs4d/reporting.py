from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import load_medgs4d_config
from .runs import write_dataframe


def build_report_metrics(per_slice: pd.DataFrame) -> pd.DataFrame:
    """Build the compact train and optional validation table used in report.pdf."""

    rows = []
    for split in ("train", "validation"):
        frame = per_slice.loc[per_slice["Split"] == split]
        if frame.empty:
            continue
        rows.extend(
            [
                {
                    "Split": split,
                    "Metric": "L1",
                    "Baseline": frame["BaselineL1"].mean(),
                    "Dynamic": frame["DynamicL1"].mean(),
                    "Improvement": frame["L1Reduction"].mean(),
                },
                {
                    "Split": split,
                    "Metric": "PSNR",
                    "Baseline": frame["BaselinePSNR"].mean(),
                    "Dynamic": frame["DynamicPSNR"].mean(),
                    "Improvement": frame["PSNRGain"].mean(),
                },
                {
                    "Split": split,
                    "Metric": "SSIM",
                    "Baseline": frame["BaselineSSIM"].mean(),
                    "Dynamic": frame["DynamicSSIM"].mean(),
                    "Improvement": frame["SSIMGain"].mean(),
                },
            ]
        )
    return pd.DataFrame(rows)


def generate_run_report(run_dir: Path, output_path: Path | None = None) -> Path:
    """Generate a minimal publication-ready PDF from saved run artifacts."""

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    run_dir = run_dir.resolve()
    output_path = output_path or run_dir / "report.pdf"
    config = load_medgs4d_config(run_dir / "config.json")
    per_slice = pd.read_csv(run_dir / "evaluation" / "per_slice.csv")
    per_phase = pd.read_csv(run_dir / "evaluation" / "per_phase.csv")
    history = pd.read_csv(run_dir / "training_history.csv")
    validation_path = run_dir / "validation_history.csv"
    validation_history = (
        pd.read_csv(validation_path) if validation_path.is_file() else pd.DataFrame()
    )
    report_metrics = build_report_metrics(per_slice)
    write_dataframe(run_dir / "report_metrics.csv", report_metrics)

    with PdfPages(output_path) as pdf:
        figure = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=[0.8, 1.4])
        title = figure.add_subplot(grid[0, :])
        title.axis("off")
        split_text = config.split.mode
        if config.split.validation_phases:
            split_text += " (validation: " + ", ".join(
                f"{phase:g}%" for phase in config.split.validation_phases
            ) + ")"
        title.text(
            0.0,
            1.0,
            "MedGS4D run summary",
            fontsize=14,
            fontweight="bold",
            va="top",
        )
        title.text(
            0.0,
            0.68,
            f"Study: {config.study_name}    Run: {config.run_name}\n"
            f"Canonical phase: {config.canonical_phase:g}%    Split: {split_text}\n"
            f"Iterations: {config.training.iterations:,}    "
            f"MLP: {config.deformation.hidden_layers} × {config.deformation.hidden_dim}    "
            f"Seed: {config.training.seed}",
            fontsize=9,
            va="top",
        )

        table_axis = figure.add_subplot(grid[1, 0])
        table_axis.axis("off")
        table_values = []
        for row in report_metrics.itertuples():
            precision = 4 if row.Metric != "SSIM" else 6
            table_values.append(
                [
                    row.Split,
                    row.Metric,
                    f"{row.Baseline:.{precision}f}",
                    f"{row.Dynamic:.{precision}f}",
                    f"{row.Improvement:+.{precision}f}",
                ]
            )
        table = table_axis.table(
            cellText=table_values,
            colLabels=["Split", "Metric", "Baseline", "Dynamic", "Δ"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.25)
        table_axis.set_title("Reconstruction metrics", fontsize=10, pad=8)

        phase_axis = figure.add_subplot(grid[1, 1])
        noncanonical = per_phase.loc[per_phase["Split"] != "canonical"]
        for split, frame in noncanonical.groupby("Split", sort=False):
            phase_axis.plot(
                frame["PhasePercent"], frame["PSNRGain"], marker="o", label=split
            )
        phase_axis.axhline(0.0, linewidth=0.8)
        phase_axis.set_xlabel("Respiratory phase [%]")
        phase_axis.set_ylabel("ΔPSNR [dB]")
        phase_axis.set_title("Per-phase improvement", fontsize=10)
        if noncanonical["Split"].nunique() > 1:
            phase_axis.legend(frameon=False, fontsize=8)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
        grid = figure.add_gridspec(1, 3)
        for axis, metric in zip(
            [figure.add_subplot(grid[0, index]) for index in range(3)],
            ["L1", "PSNR", "SSIM"],
        ):
            rolling = history.set_index("Iteration")[metric].rolling(50, min_periods=1).mean()
            axis.plot(rolling.index, rolling.values, label="train")
            if not validation_history.empty and metric in validation_history.columns:
                axis.plot(
                    validation_history["Iteration"],
                    validation_history[metric],
                    marker="o",
                    label="validation",
                )
            axis.set_xlabel("Iteration")
            axis.set_title(metric)
            if metric == "L1":
                axis.set_ylabel("Metric value")
            if not validation_history.empty:
                axis.legend(frameon=False, fontsize=8)
        figure.suptitle("Training history", fontsize=12)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return output_path
