from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from .deformation import DeformationField
from .results import load_run, load_run_models
from .runs import write_dataframe, write_json


def _statistics(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().flatten()
    return {
        f"Mean{prefix}": float(values.mean().item()),
        f"Median{prefix}": float(values.median().item()),
        f"P95{prefix}": float(torch.quantile(values, 0.95).item()),
        f"Max{prefix}": float(values.max().item()),
    }


def _gaussian_indices(
    field: DeformationField,
    *,
    sample_gaussians: int,
    seed: int,
) -> torch.Tensor:
    gaussian_count = int(field.xyz.shape[0])
    if sample_gaussians <= 0 or sample_gaussians >= gaussian_count:
        return torch.arange(
            gaussian_count,
            device=field.xyz.device,
            dtype=torch.long,
        )

    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(
            gaussian_count,
            size=sample_gaussians,
            replace=False,
        )
    )
    return torch.as_tensor(
        selected,
        device=field.xyz.device,
        dtype=torch.long,
    )


def evaluate_deformation_field(
    field: DeformationField,
    phases: list[float],
    *,
    sample_gaussians: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Measure deformation magnitude and temporal continuity by phase."""

    ordered_phases = [float(value) for value in phases]
    indices = _gaussian_indices(
        field,
        sample_gaussians=sample_gaussians,
        seed=seed,
    )
    selected_count = int(indices.numel())

    phase_rows = []
    saved_states: list[tuple[float, torch.Tensor, torch.Tensor]] = []

    field.model.eval()
    with torch.no_grad():
        for phase in ordered_phases:
            state = field.build_subset_state(
                phase / 100.0,
                indices,
            )
            delta_xz = state["delta_xz"]
            delta_m = state["delta_m"]
            displacement_xz = torch.linalg.vector_norm(
                delta_xz,
                dim=-1,
            )
            absolute_delta_m = delta_m.abs().flatten()
            normalized = torch.cat(
                [
                    delta_xz / field.xz_scale,
                    delta_m / field.m_scale,
                ],
                dim=-1,
            )
            normalized_magnitude = torch.linalg.vector_norm(
                normalized,
                dim=-1,
            )

            phase_rows.append(
                {
                    "PhasePercent": phase,
                    "GaussianCount": selected_count,
                    "MeanDeltaX": float(delta_xz[:, 0].mean().item()),
                    "MeanDeltaZ": float(delta_xz[:, 1].mean().item()),
                    **_statistics(
                        displacement_xz,
                        "DisplacementXZ",
                    ),
                    **_statistics(
                        absolute_delta_m,
                        "AbsDeltaM",
                    ),
                    **_statistics(
                        normalized_magnitude,
                        "NormalizedDeformation",
                    ),
                }
            )
            saved_states.append(
                (
                    phase,
                    delta_xz.detach().cpu(),
                    delta_m.detach().cpu(),
                )
            )

    temporal_rows = []
    if len(saved_states) > 1:
        for index, (phase_from, delta_xz_from, delta_m_from) in enumerate(
            saved_states
        ):
            phase_to, delta_xz_to, delta_m_to = saved_states[
                (index + 1) % len(saved_states)
            ]
            step_xz = torch.linalg.vector_norm(
                delta_xz_to - delta_xz_from,
                dim=-1,
            )
            step_m = (delta_m_to - delta_m_from).abs().flatten()
            temporal_rows.append(
                {
                    "PhaseFrom": phase_from,
                    "PhaseTo": phase_to,
                    "CyclicBoundary": index == len(saved_states) - 1,
                    **_statistics(
                        step_xz,
                        "StepDisplacementXZ",
                    ),
                    **_statistics(
                        step_m,
                        "StepAbsDeltaM",
                    ),
                }
            )

    per_phase = pd.DataFrame(phase_rows)
    temporal = pd.DataFrame(temporal_rows)

    canonical_phase = float(field.canonical_phase)
    anchor_index = int(
        np.argmin(
            np.abs(
                per_phase["PhasePercent"].to_numpy()
                - canonical_phase
            )
        )
    )
    anchor = per_phase.iloc[anchor_index]
    summary = {
        "CanonicalPhase": canonical_phase,
        "EvaluatedPhases": ordered_phases,
        "TotalGaussianCount": int(field.xyz.shape[0]),
        "EvaluatedGaussianCount": selected_count,
        "AnchorEvaluatedPhase": float(anchor["PhasePercent"]),
        "AnchorMaxDisplacementXZ": float(
            anchor["MaxDisplacementXZ"]
        ),
        "AnchorMaxAbsDeltaM": float(anchor["MaxAbsDeltaM"]),
        "MaximumP95DisplacementXZ": float(
            per_phase["P95DisplacementXZ"].max()
        ),
        "MaximumP95AbsDeltaM": float(
            per_phase["P95AbsDeltaM"].max()
        ),
    }
    return per_phase, temporal, summary


def save_deformation_diagnostics_pdf(
    per_phase: pd.DataFrame,
    temporal: pd.DataFrame,
    output_path: Path,
    *,
    checkpoint_iteration: int,
    canonical_checkpoint_iteration: int,
) -> Path:
    """Save publication-ready deformation diagnostics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = per_phase.sort_values("PhasePercent")
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
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 3.8),
            constrained_layout=True,
        )
        axes[0].plot(
            ordered["PhasePercent"],
            ordered["MeanDisplacementXZ"],
            marker="o",
            label="mean",
        )
        axes[0].plot(
            ordered["PhasePercent"],
            ordered["P95DisplacementXZ"],
            marker="o",
            label="p95",
        )
        axes[0].set_xlabel("Respiratory phase [%]")
        axes[0].set_ylabel("x-z displacement [canonical units]")
        axes[0].set_title("Spatial deformation")
        axes[0].grid(True, linewidth=0.5, alpha=0.3)
        axes[0].legend(frameon=False)

        axes[1].plot(
            ordered["PhasePercent"],
            ordered["MeanAbsDeltaM"],
            marker="o",
            label="mean",
        )
        axes[1].plot(
            ordered["PhasePercent"],
            ordered["P95AbsDeltaM"],
            marker="o",
            label="p95",
        )
        axes[1].set_xlabel("Respiratory phase [%]")
        axes[1].set_ylabel("|Δm|")
        axes[1].set_title("m deformation")
        axes[1].grid(True, linewidth=0.5, alpha=0.3)
        axes[1].legend(frameon=False)

        figure.text(
            0.01,
            0.01,
            (
                f"Dynamic checkpoint {checkpoint_iteration} | "
                f"canonical checkpoint {canonical_checkpoint_iteration}"
            ),
            fontsize=8,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 3.8),
            constrained_layout=True,
        )
        axes[0].plot(
            ordered["PhasePercent"],
            ordered["MeanNormalizedDeformation"],
            marker="o",
            label="mean",
        )
        axes[0].plot(
            ordered["PhasePercent"],
            ordered["P95NormalizedDeformation"],
            marker="o",
            label="p95",
        )
        axes[0].set_xlabel("Respiratory phase [%]")
        axes[0].set_ylabel("Normalized deformation magnitude")
        axes[0].set_title("Scale-normalized deformation")
        axes[0].grid(True, linewidth=0.5, alpha=0.3)
        axes[0].legend(frameon=False)

        axes[1].plot(
            ordered["PhasePercent"],
            ordered["MeanDeltaX"],
            marker="o",
            label="mean Δx",
        )
        axes[1].plot(
            ordered["PhasePercent"],
            ordered["MeanDeltaZ"],
            marker="o",
            label="mean Δz",
        )
        axes[1].axhline(0.0, linewidth=0.8)
        axes[1].set_xlabel("Respiratory phase [%]")
        axes[1].set_ylabel("Signed displacement [canonical units]")
        axes[1].set_title("Mean deformation direction")
        axes[1].grid(True, linewidth=0.5, alpha=0.3)
        axes[1].legend(frameon=False)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        if not temporal.empty:
            figure, axes = plt.subplots(
                1,
                2,
                figsize=(8.8, 3.8),
                constrained_layout=True,
            )
            step_labels = [
                f"{row.PhaseFrom:g}→{row.PhaseTo:g}"
                for row in temporal.itertuples()
            ]
            x = np.arange(len(temporal))
            axes[0].plot(
                x,
                temporal["MeanStepDisplacementXZ"],
                marker="o",
                label="mean",
            )
            axes[0].plot(
                x,
                temporal["P95StepDisplacementXZ"],
                marker="o",
                label="p95",
            )
            axes[0].set_xticks(x, step_labels, rotation=45)
            axes[0].set_ylabel("Change in x-z deformation")
            axes[0].set_title("Temporal deformation steps")
            axes[0].grid(True, linewidth=0.5, alpha=0.3)
            axes[0].legend(frameon=False)

            axes[1].plot(
                x,
                temporal["MeanStepAbsDeltaM"],
                marker="o",
                label="mean",
            )
            axes[1].plot(
                x,
                temporal["P95StepAbsDeltaM"],
                marker="o",
                label="p95",
            )
            axes[1].set_xticks(x, step_labels, rotation=45)
            axes[1].set_ylabel("Change in |Δm|")
            axes[1].set_title("Temporal m steps")
            axes[1].grid(True, linewidth=0.5, alpha=0.3)
            axes[1].legend(frameon=False)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    return output_path


def diagnose_run_deformation(
    run_dir: Path,
    *,
    checkpoint: Path | None = None,
    phases: list[float] | None = None,
    sample_gaussians: int = 0,
    seed: int = 42,
    device: str = "cuda",
) -> Path:
    """Load a run checkpoint and save detailed deformation diagnostics."""

    run = load_run(run_dir=run_dir)
    checkpoint_path = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else run.checkpoint
    )
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run.run_dir}")

    _, field = load_run_models(
        run,
        checkpoint=checkpoint_path,
        device=device,
    )
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_iteration = int(checkpoint_payload["iteration"])
    selected_phases = (
        [float(value) for value in run.study.phases]
        if phases is None
        else [float(value) for value in phases]
    )

    per_phase, temporal, summary = evaluate_deformation_field(
        field,
        selected_phases,
        sample_gaussians=sample_gaussians,
        seed=seed,
    )
    summary.update(
        {
            "Checkpoint": str(checkpoint_path),
            "CheckpointIteration": checkpoint_iteration,
            "CanonicalCheckpoint": run.config.canonical_checkpoint,
            "CanonicalCheckpointIteration": (
                run.config.canonical_checkpoint_iteration
            ),
        }
    )

    output_dir = (
        run.run_dir
        / "evaluation"
        / "deformation"
        / f"iter_{checkpoint_iteration:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_dir / "per_phase.csv", per_phase)
    write_dataframe(output_dir / "temporal_steps.csv", temporal)
    write_json(output_dir / "summary.json", summary)
    save_deformation_diagnostics_pdf(
        per_phase,
        temporal,
        output_dir / "diagnostics.pdf",
        checkpoint_iteration=checkpoint_iteration,
        canonical_checkpoint_iteration=(
            run.config.canonical_checkpoint_iteration
        ),
    )
    return output_dir
