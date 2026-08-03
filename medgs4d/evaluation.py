from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from .canonical import (
    CanonicalAssets,
    get_camera_for_slice,
    load_frozen_canonical,
    render_canonical_slice,
)
from .config import load_medgs4d_config
from .data import StudyManifest, load_study_manifest, load_target_tensor
from .deformation import DeformationField
from .runs import find_latest_checkpoint, write_dataframe, write_json
from .training import load_checkpoint


EvaluationSplit = Literal["train", "validation", "all"]


def calculate_reconstruction_metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Calculate L1, PSNR, and a simple global SSIM for one rendered slice."""

    rendered_batch = rendered.unsqueeze(0) if rendered.ndim == 3 else rendered
    target_batch = target.unsqueeze(0) if target.ndim == 3 else target
    l1_value = torch.mean(torch.abs(rendered_batch - target_batch))
    mse = torch.mean((rendered_batch - target_batch) ** 2)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))

    # Used only when upstream SSIM is not passed. Evaluation below uses MedGS SSIM.
    mu_x = rendered_batch.mean()
    mu_y = target_batch.mean()
    var_x = rendered_batch.var(unbiased=False)
    var_y = target_batch.var(unbiased=False)
    covariance = ((rendered_batch - mu_x) * (target_batch - mu_y)).mean()
    c1, c2 = 0.01**2, 0.03**2
    ssim_value = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return {
        "L1": float(l1_value.item()),
        "PSNR": float(psnr.item()),
        "SSIM": float(ssim_value.item()),
    }


def _medgs_metrics(
    canonical: CanonicalAssets,
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    rendered_batch = rendered.unsqueeze(0) if rendered.ndim == 3 else rendered
    target_batch = target.unsqueeze(0) if target.ndim == 3 else target
    l1_value = canonical.runtime.l1_loss(rendered_batch, target_batch)
    ssim_value = canonical.runtime.ssim(rendered_batch, target_batch)
    mse = torch.mean((rendered_batch - target_batch) ** 2)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {
        "L1": float(l1_value.item()),
        "PSNR": float(psnr.item()),
        "SSIM": float(ssim_value.item()),
    }


def evaluate_canonical_model(
    study: StudyManifest,
    canonical: CanonicalAssets,
    *,
    target_representation: Literal["raw", "denoised"] = "raw",
) -> pd.DataFrame:
    """Evaluate a static canonical model on every slice of its reference phase."""

    rows = []
    phase = canonical.config.canonical_phase
    with torch.no_grad():
        for slice_index in range(study.slice_count):
            target = load_target_tensor(
                study,
                phase,
                slice_index,
                representation=target_representation,
                device=str(canonical.xyz.device),
            )
            rendered = render_canonical_slice(canonical, slice_index)
            metrics = _medgs_metrics(canonical, rendered, target)
            rows.append(
                {
                    "PhasePercent": phase,
                    "SliceIndex": slice_index,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)



def _metric_summary(series: pd.Series) -> dict[str, float]:
    """Return standard descriptive statistics for one reconstruction metric."""

    return {
        "Mean": float(series.mean()),
        "StandardDeviation": float(series.std()),
        "Minimum": float(series.min()),
        "Maximum": float(series.max()),
    }


def save_canonical_metrics_pdf(
    per_slice: pd.DataFrame,
    output_path: Path,
    *,
    study_name: str,
    canonical_phase: float,
    iteration: int,
    target_representation: str,
) -> Path:
    """Save publication-ready per-slice PSNR, SSIM, and L1 plots as a PDF."""

    metric_specs = (
        ("PSNR", "PSNR [dB]", "Higher is better"),
        ("SSIM", "SSIM", "Higher is better"),
        ("L1", "L1 error", "Lower is better"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_metadata = {
        "Title": "Canonical MedGS reconstruction metrics",
        "Subject": (
            f"{study_name}, phase {canonical_phase:g}%, "
            f"iteration {iteration}"
        ),
        "Creator": "medgs4d",
    }

    rc = {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc), PdfPages(output_path, metadata=pdf_metadata) as pdf:
        for column, ylabel, interpretation in metric_specs:
            values = per_slice[column]
            mean_value = float(values.mean())

            figure, axis = plt.subplots(
                figsize=(7.2, 4.4),
                constrained_layout=True,
            )
            axis.plot(
                per_slice["SliceIndex"],
                values,
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=column,
            )
            axis.axhline(
                mean_value,
                linestyle="--",
                linewidth=1.0,
                label=f"Mean = {mean_value:.4f}",
            )
            axis.set_xlabel("Slice index")
            axis.set_ylabel(ylabel)
            axis.set_title(f"Canonical reconstruction: {ylabel}")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(True, linewidth=0.5, alpha=0.3)
            axis.legend(frameon=False)

            figure.text(
                0.01,
                0.01,
                (
                    f"{study_name} | phase {canonical_phase:g}% | "
                    f"iteration {iteration} | "
                    f"target: {target_representation} | {interpretation}"
                ),
                fontsize=8,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    return output_path


def _load_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _save_metric_history_pdf(
    history: pd.DataFrame,
    output_path: Path,
    *,
    x_column: str,
    title_prefix: str,
    footer: str,
    metric_specs: tuple[tuple[str, str, str], ...],
) -> Path:
    """Save one PDF page per metric against an ordered training axis."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rc = {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc), PdfPages(output_path) as pdf:
        for column, ylabel, interpretation in metric_specs:
            if column not in history.columns:
                continue
            values = pd.to_numeric(history[column], errors="coerce")
            valid = values.notna()
            if not valid.any():
                continue

            figure, axis = plt.subplots(
                figsize=(7.2, 4.4),
                constrained_layout=True,
            )
            axis.plot(
                history.loc[valid, x_column],
                values.loc[valid],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=column,
            )
            axis.set_xlabel(x_column)
            axis.set_ylabel(ylabel)
            axis.set_title(f"{title_prefix}: {ylabel}")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(True, linewidth=0.5, alpha=0.3)
            axis.legend(frameon=False)
            figure.text(
                0.01,
                0.01,
                f"{footer} | {interpretation}",
                fontsize=8,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return output_path


def save_canonical_training_history_pdf(
    canonical_run_dir: Path,
    *,
    study_name: str,
    canonical_phase: float,
) -> Path | None:
    """Plot minibatch metrics recorded by the upstream MedGS loop."""

    history_path = canonical_run_dir / "canonical_training_history.csv"
    history = _load_csv_or_empty(history_path)
    if history.empty:
        return None

    history = (
        history.sort_values("Iteration")
        .drop_duplicates("Iteration", keep="last")
        .reset_index(drop=True)
    )
    write_dataframe(history_path, history)

    return _save_metric_history_pdf(
        history,
        canonical_run_dir / "canonical_training_history.pdf",
        x_column="Iteration",
        title_prefix="Canonical MedGS training",
        footer=f"{study_name} | phase {canonical_phase:g}%",
        metric_specs=(
            ("TotalLoss", "Total loss", "Lower is better"),
            ("L1", "L1", "Lower is better"),
            (
                "InterpolationL1",
                "Interpolation L1",
                "Lower is better; active after interpolation warm-up",
            ),
            ("PSNR", "PSNR [dB]", "Higher is better"),
            ("SSIM", "SSIM", "Higher is better"),
            ("SigmaLoss", "Sigma regularization loss", "Lower is better"),
            ("GaussianCount", "Gaussian count", "Model size over training"),
        ),
    )


def _existing_evaluation_history_seed(
    evaluation_dir: Path,
) -> pd.DataFrame:
    history = _load_csv_or_empty(evaluation_dir / "history.csv")
    if not history.empty:
        return history

    overall_path = evaluation_dir / "overall.json"
    if not overall_path.is_file():
        return pd.DataFrame()

    previous = json.loads(overall_path.read_text(encoding="utf-8"))
    required = ("Iteration", "MeanL1", "MeanPSNR", "MeanSSIM")
    if not all(name in previous for name in required):
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Iteration": int(previous["Iteration"]),
                "MeanL1": float(previous["MeanL1"]),
                "MeanPSNR": float(previous["MeanPSNR"]),
                "MeanSSIM": float(previous["MeanSSIM"]),
                "SliceCount": int(previous.get("SliceCount", 0)),
                "TargetRepresentation": previous.get(
                    "TargetRepresentation",
                    "",
                ),
            }
        ]
    )


def update_canonical_evaluation_history(
    evaluation_dir: Path,
    summary: Mapping[str, Any],
) -> pd.DataFrame:
    """Upsert one full-slice evaluation point by checkpoint iteration."""

    history = _existing_evaluation_history_seed(evaluation_dir)
    row = pd.DataFrame(
        [
            {
                "Iteration": int(summary["Iteration"]),
                "MeanL1": float(summary["MeanL1"]),
                "MeanPSNR": float(summary["MeanPSNR"]),
                "MeanSSIM": float(summary["MeanSSIM"]),
                "SliceCount": int(summary["SliceCount"]),
                "TargetRepresentation": summary[
                    "TargetRepresentation"
                ],
            }
        ]
    )
    history = (
        pd.concat([history, row], ignore_index=True)
        .sort_values("Iteration")
        .drop_duplicates("Iteration", keep="last")
        .reset_index(drop=True)
    )
    write_dataframe(evaluation_dir / "history.csv", history)
    return history


def save_canonical_evaluation_history_pdf(
    history: pd.DataFrame,
    output_path: Path,
    *,
    study_name: str,
    canonical_phase: float,
) -> Path:
    """Plot full-slice metrics at every completed checkpoint."""

    return _save_metric_history_pdf(
        history,
        output_path,
        x_column="Iteration",
        title_prefix="Canonical full-slice evaluation",
        footer=f"{study_name} | phase {canonical_phase:g}%",
        metric_specs=(
            ("MeanPSNR", "Mean PSNR [dB]", "Higher is better"),
            ("MeanSSIM", "Mean SSIM", "Higher is better"),
            ("MeanL1", "Mean L1", "Lower is better"),
        ),
    )


def save_canonical_evaluation(
    canonical_run_dir: Path,
    medgs_repository: Path,
    *,
    target_representation: Literal["raw", "denoised"] = "raw",
    device: str = "cuda",
) -> Path:
    """Evaluate one canonical run and save CSV, JSON, and PDF outputs."""

    canonical_run_dir = canonical_run_dir.resolve()
    metadata = json.loads(
        (canonical_run_dir / "canonical_run.json").read_text(encoding="utf-8")
    )
    study = load_study_manifest(Path(metadata["study_dir"]))
    canonical = load_frozen_canonical(
        canonical_run_dir,
        medgs_repository,
        device=device,
    )
    per_slice = evaluate_canonical_model(
        study,
        canonical,
        target_representation=target_representation,
    )

    evaluation_dir = canonical_run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(evaluation_dir / "per_slice.csv", per_slice)

    l1_summary = _metric_summary(per_slice["L1"])
    psnr_summary = _metric_summary(per_slice["PSNR"])
    ssim_summary = _metric_summary(per_slice["SSIM"])
    summary = {
        "StudyName": study.study_name,
        "CanonicalPhase": float(canonical.config.canonical_phase),
        "Iteration": int(canonical.loaded_iteration),
        "TargetRepresentation": target_representation,
        "SliceCount": int(len(per_slice)),
        "MeanL1": l1_summary["Mean"],
        "MeanPSNR": psnr_summary["Mean"],
        "MeanSSIM": ssim_summary["Mean"],
        "L1": l1_summary,
        "PSNR": psnr_summary,
        "SSIM": ssim_summary,
    }
    evaluation_history = update_canonical_evaluation_history(
        evaluation_dir,
        summary,
    )
    write_json(evaluation_dir / "overall.json", summary)

    save_canonical_metrics_pdf(
        per_slice,
        evaluation_dir / "canonical_metrics.pdf",
        study_name=study.study_name,
        canonical_phase=canonical.config.canonical_phase,
        iteration=canonical.loaded_iteration,
        target_representation=target_representation,
    )
    save_canonical_evaluation_history_pdf(
        evaluation_history,
        evaluation_dir / "history.pdf",
        study_name=study.study_name,
        canonical_phase=canonical.config.canonical_phase,
    )
    save_canonical_training_history_pdf(
        canonical_run_dir,
        study_name=study.study_name,
        canonical_phase=canonical.config.canonical_phase,
    )
    return evaluation_dir


def evaluate_medgs4d_model(
    study: StudyManifest,
    canonical: CanonicalAssets,
    field: DeformationField,
    split_manifest: pd.DataFrame,
    *,
    split: EvaluationSplit = "all",
    target_representation: Literal["raw", "denoised"] = "raw",
) -> pd.DataFrame:
    """Evaluate baseline and dynamic reconstructions on the selected split."""

    rows_to_evaluate = split_manifest
    if split != "all":
        rows_to_evaluate = split_manifest.loc[split_manifest["Split"] == split]
    records = []
    field.model.eval()
    with torch.no_grad():
        for row in rows_to_evaluate.itertuples():
            phase = float(row.PhasePercent)
            slice_index = int(row.SliceIndex)
            target = load_target_tensor(
                study,
                phase,
                slice_index,
                representation=target_representation,
                device=str(field.xyz.device),
            )
            camera = get_camera_for_slice(canonical, slice_index)
            baseline = canonical.runtime.render(
                camera,
                canonical.gaussians,
                canonical.pipeline,
                canonical.background,
            )["render"]
            view, _ = field.build_phase_state(
                phase / 100.0, use_checkpointing=False
            )
            dynamic = canonical.runtime.render(
                camera,
                view,
                canonical.pipeline,
                canonical.background,
            )["render"]
            baseline_metrics = _medgs_metrics(canonical, baseline, target)
            dynamic_metrics = _medgs_metrics(canonical, dynamic, target)
            records.append(
                {
                    "Split": str(row.Split),
                    "PhasePercent": phase,
                    "SliceIndex": slice_index,
                    "BaselineL1": baseline_metrics["L1"],
                    "DynamicL1": dynamic_metrics["L1"],
                    "L1Reduction": baseline_metrics["L1"] - dynamic_metrics["L1"],
                    "BaselinePSNR": baseline_metrics["PSNR"],
                    "DynamicPSNR": dynamic_metrics["PSNR"],
                    "PSNRGain": dynamic_metrics["PSNR"] - baseline_metrics["PSNR"],
                    "BaselineSSIM": baseline_metrics["SSIM"],
                    "DynamicSSIM": dynamic_metrics["SSIM"],
                    "SSIMGain": dynamic_metrics["SSIM"] - baseline_metrics["SSIM"],
                }
            )
    return pd.DataFrame(records)


def aggregate_metrics_per_phase(per_slice: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconstruction metrics separately for every phase and split."""

    metric_columns = [
        "BaselineL1",
        "DynamicL1",
        "L1Reduction",
        "BaselinePSNR",
        "DynamicPSNR",
        "PSNRGain",
        "BaselineSSIM",
        "DynamicSSIM",
        "SSIMGain",
    ]
    return (
        per_slice.groupby(["Split", "PhasePercent"], as_index=False)[metric_columns]
        .mean()
        .merge(
            per_slice.groupby(["Split", "PhasePercent"], as_index=False)
            .size()
            .rename(columns={"size": "SliceCount"}),
            on=["Split", "PhasePercent"],
        )
        .sort_values("PhasePercent")
        .reset_index(drop=True)
    )


def _summary_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"SampleCount": 0}
    return {
        "SampleCount": int(len(frame)),
        "BaselineL1": float(frame["BaselineL1"].mean()),
        "DynamicL1": float(frame["DynamicL1"].mean()),
        "L1Reduction": float(frame["L1Reduction"].mean()),
        "BaselinePSNR": float(frame["BaselinePSNR"].mean()),
        "DynamicPSNR": float(frame["DynamicPSNR"].mean()),
        "PSNRGain": float(frame["PSNRGain"].mean()),
        "BaselineSSIM": float(frame["BaselineSSIM"].mean()),
        "DynamicSSIM": float(frame["DynamicSSIM"].mean()),
        "SSIMGain": float(frame["SSIMGain"].mean()),
    }


def aggregate_metrics_overall(per_slice: pd.DataFrame) -> dict[str, Any]:
    """Aggregate metrics over all samples and separately by split."""

    noncanonical = per_slice.loc[per_slice["Split"] != "canonical"]
    by_split = {
        split: _summary_for_frame(frame)
        for split, frame in per_slice.groupby("Split", sort=False)
    }
    return {
        "All": _summary_for_frame(per_slice),
        "AllNoncanonical": _summary_for_frame(noncanonical),
        "BySplit": by_split,
    }


def save_evaluation_results(
    evaluation_dir: Path,
    *,
    per_slice: pd.DataFrame,
    per_phase: pd.DataFrame,
    overall: Mapping[str, Any],
    prefix: str = "",
) -> None:
    """Save per-slice, per-phase, and overall evaluation outputs."""

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    write_dataframe(evaluation_dir / f"{stem}per_slice.csv", per_slice)
    write_dataframe(evaluation_dir / f"{stem}per_phase.csv", per_phase)
    write_json(evaluation_dir / f"{stem}overall.json", overall)


def evaluate_run(
    run_dir: Path,
    *,
    split: EvaluationSplit = "all",
    checkpoint: Path | None = None,
    force: bool = False,
    device: str = "cuda",
) -> Path:
    """Load one run, evaluate it, save metrics, and regenerate report.pdf."""

    run_dir = run_dir.resolve()
    config = load_medgs4d_config(run_dir / "config.json")
    study = load_study_manifest(Path(config.data_dir))
    canonical = load_frozen_canonical(
        Path(config.canonical_model_dir),
        Path(config.medgs_repository),
        device=device,
    )
    field = DeformationField(
        canonical,
        config.deformation,
        config.canonical_phase,
        seed=config.training.seed,
    )
    checkpoint_path = checkpoint or find_latest_checkpoint(run_dir / "checkpoints")
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")
    load_checkpoint(checkpoint_path, field, device=device)
    split_manifest = pd.read_csv(run_dir / "split_manifest.csv")

    prefix = "" if split == "all" else split
    expected = run_dir / "evaluation" / f"{prefix + '_' if prefix else ''}per_slice.csv"
    if expected.exists() and not force:
        raise FileExistsError(f"Evaluation already exists: {expected}; use --force")
    per_slice = evaluate_medgs4d_model(
        study,
        canonical,
        field,
        split_manifest,
        split=split,
        target_representation=config.target_representation,
    )
    per_phase = aggregate_metrics_per_phase(per_slice)
    overall = aggregate_metrics_overall(per_slice)
    overall["Checkpoint"] = str(checkpoint_path)
    overall["CheckpointIteration"] = int(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)["iteration"]
    )
    save_evaluation_results(
        run_dir / "evaluation",
        per_slice=per_slice,
        per_phase=per_phase,
        overall=overall,
        prefix=prefix,
    )
    if split == "all":
        from .reporting import generate_run_report

        generate_run_report(run_dir)
    return run_dir / "evaluation"
