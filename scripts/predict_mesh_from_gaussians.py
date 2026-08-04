#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer local MedGS4D Gaussian motion to the canonical RTSTRUCT "
            "tumor mesh and evaluate predicted meshes against all reference phases."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mesh-series-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--geometry-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phases", type=float, nargs="*")
    parser.add_argument("--robust-z", type=float, default=3.5)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--opacity-power", type=float, default=1.0)
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser


def checkpoint_iteration(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["iteration"])


def percentile_columns(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}Mean": float(np.mean(values)),
        f"{prefix}Median": float(np.median(values)),
        f"{prefix}P95": float(np.percentile(values, 95)),
        f"{prefix}Maximum": float(np.max(values)),
    }


def phase_split_table(split_manifest: pd.DataFrame) -> dict[float, str]:
    mapping = {}
    for phase, group in split_manifest.groupby("PhasePercent"):
        values = group["Split"].drop_duplicates().tolist()
        if len(values) != 1:
            raise ValueError(f"Phase {phase:g} has multiple split labels: {values}")
        mapping[float(phase)] = str(values[0])
    return mapping


def aggregate_metrics(table: pd.DataFrame) -> dict[str, object]:
    metrics = (
        "Dice",
        "IoU",
        "MeanSurfaceDistanceMm",
        "HD95Mm",
        "CentroidErrorMm",
        "VolumeErrorPercent",
    )

    def summarize(selected: pd.DataFrame) -> dict[str, float | int]:
        result: dict[str, float | int] = {"PhaseCount": int(len(selected))}
        for metric in metrics:
            baseline = pd.to_numeric(selected[f"Baseline{metric}"])
            predicted = pd.to_numeric(selected[f"Predicted{metric}"])
            result[f"Baseline{metric}"] = float(baseline.mean())
            result[f"Predicted{metric}"] = float(predicted.mean())
            if metric in {"Dice", "IoU"}:
                result[f"{metric}Improvement"] = float((predicted - baseline).mean())
            else:
                result[f"{metric}Reduction"] = float((baseline - predicted).mean())
        return result

    noncanonical = table[table["Split"] != "canonical"]
    by_split = {
        split: summarize(group)
        for split, group in table.groupby("Split")
    }
    return {
        "AllNoncanonical": summarize(noncanonical),
        "BySplit": by_split,
    }


def save_metrics_pdf(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    specifications = (
        ("Dice", "Dice", "Higher is better"),
        ("MeanSurfaceDistanceMm", "Mean surface distance [mm]", "Lower is better"),
        ("HD95Mm", "HD95 [mm]", "Lower is better"),
        ("CentroidErrorMm", "Centroid error [mm]", "Lower is better"),
        ("VolumeErrorPercent", "Signed volume error [%]", "Closer to zero is better"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for metric, ylabel, interpretation in specifications:
            figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
            axis.plot(
                table["PhasePercent"],
                table[f"Baseline{metric}"],
                marker="o",
                label="Static phase-0 mesh",
            )
            axis.plot(
                table["PhasePercent"],
                table[f"Predicted{metric}"],
                marker="o",
                label="Gaussian-transferred mesh",
            )
            axis.set_xlabel("Respiratory phase [%]")
            axis.set_ylabel(ylabel)
            axis.set_title(f"Mesh transfer evaluation: {ylabel}")
            axis.grid(True, linewidth=0.5, alpha=0.3)
            axis.legend(frameon=False)
            figure.text(0.01, 0.01, interpretation, fontsize=8)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        axis.plot(
            table["PhasePercent"],
            table["VertexDisplacementMmMedian"],
            marker="o",
            label="Median",
        )
        axis.plot(
            table["PhasePercent"],
            table["VertexDisplacementMmP95"],
            marker="o",
            label="P95",
        )
        axis.set_xlabel("Respiratory phase [%]")
        axis.set_ylabel("Transferred vertex displacement [mm]")
        axis.set_title("Transferred canonical-mesh motion")
        axis.grid(True, linewidth=0.5, alpha=0.3)
        axis.legend(frameon=False)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def save_phase_artifacts(
    output_dir: Path,
    phase: float,
    predicted_mesh,
    predicted_mask: np.ndarray,
    vertex_displacement: np.ndarray,
    inlier_mask: np.ndarray,
    weights: np.ndarray,
    neighbor_indices: np.ndarray,
    metrics: dict[str, object],
) -> None:
    from medgs4d.meshes import save_mesh_npz, write_ply
    from medgs4d.runs import write_dataframe, write_json

    phase_dir = output_dir / "predictions" / f"phase_{int(round(phase)):02d}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    save_mesh_npz(phase_dir / "mesh_predicted.npz", predicted_mesh)
    write_ply(
        phase_dir / "mesh_predicted.ply",
        predicted_mesh.vertices_xyz,
        predicted_mesh.faces,
    )
    np.save(phase_dir / "mask_predicted.npy", predicted_mask)
    np.save(phase_dir / "vertex_displacement_xyz_mm.npy", vertex_displacement)

    vertex_table = pd.DataFrame(
        {
            "VertexIndex": np.arange(len(vertex_displacement), dtype=np.int64),
            "DeltaPatientXmm": vertex_displacement[:, 0],
            "DeltaPatientYmm": vertex_displacement[:, 1],
            "DeltaPatientZmm": vertex_displacement[:, 2],
            "DisplacementMm": np.linalg.norm(vertex_displacement, axis=1),
            "InlierGaussianCount": inlier_mask.sum(axis=1),
            "MaximumTransferWeight": weights.max(axis=1),
        }
    )
    write_dataframe(phase_dir / "vertex_transfer.csv", vertex_table)
    write_json(
        phase_dir / "manifest.json",
        {
            "phase_percent": float(phase),
            "mesh_file": "mesh_predicted.npz",
            "mesh_ply_file": "mesh_predicted.ply",
            "mask_file": "mask_predicted.npy",
            "vertex_displacement_file": "vertex_displacement_xyz_mm.npy",
            "vertex_transfer_file": "vertex_transfer.csv",
            "canonical_faces_preserved": True,
            "neighbor_count": int(neighbor_indices.shape[1]),
            "metrics": metrics,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from medgs4d.gaussian_geometry import (
        GaussianDicomTransform,
        canonical_latent_centers,
    )
    from medgs4d.mesh_transfer import (
        evaluate_mesh_against_reference,
        mesh_in_geometry,
        prefix_metrics,
        robust_distance_weights,
    )
    from medgs4d.mesh_validation import load_case
    from medgs4d.results import load_run, load_run_models
    from medgs4d.runs import prepare_output_directory, write_dataframe, write_json

    run = load_run(run_dir=args.run_dir)
    checkpoint = args.checkpoint or run.checkpoint
    if checkpoint is None:
        raise FileNotFoundError(f"No dynamic checkpoint found in {args.run_dir}")
    iteration = checkpoint_iteration(checkpoint)

    geometry_dir = args.geometry_dir or (
        args.run_dir
        / "evaluation"
        / "gaussian_geometry"
        / f"iter_{iteration:06d}"
    )
    neighbor_indices = np.load(geometry_dir / "mesh_vertex_knn_indices.npy")
    neighbor_distances = np.load(geometry_dir / "mesh_vertex_knn_distances_mm.npy")

    output_dir = args.output_dir or (
        args.run_dir
        / "evaluation"
        / "mesh_transfer"
        / f"iter_{iteration:06d}"
    )
    prepare_output_directory(output_dir, force=args.force)

    canonical_phase = int(round(run.config.canonical_phase))
    canonical_case = load_case(
        args.mesh_series_dir / f"phase_{canonical_phase:02d}"
    )
    canonical, field = load_run_models(
        run,
        checkpoint=checkpoint,
        device=args.device,
    )
    transform = GaussianDicomTransform.from_canonical(
        canonical,
        canonical_case["geometry"],
    )
    canonical_latent = canonical_latent_centers(canonical)
    canonical_patient = transform.latent_to_patient_xyz(canonical_latent)
    canonical_mesh = canonical_case["mesh"]

    if neighbor_indices.shape[0] != len(canonical_mesh.vertices_xyz):
        raise ValueError("Stage-A KNN rows do not match canonical mesh vertices")
    if neighbor_indices.shape != neighbor_distances.shape:
        raise ValueError("Stage-A KNN indices and distances have different shapes")

    unique_neighbors = np.unique(neighbor_indices)
    global_to_local = np.full(len(canonical_latent), -1, dtype=np.int64)
    global_to_local[unique_neighbors] = np.arange(len(unique_neighbors), dtype=np.int64)
    local_neighbor_indices = global_to_local[neighbor_indices]
    torch_indices = torch.as_tensor(
        unique_neighbors,
        dtype=torch.long,
        device=field.xyz.device,
    )

    opacity = (
        canonical.gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
    )
    neighbor_confidence = np.maximum(
        opacity[neighbor_indices],
        1e-8,
    ) ** float(args.opacity_power)

    phases = [float(value) for value in (args.phases or run.study.phases)]
    split_by_phase = phase_split_table(run.split_manifest)
    local_rows = []
    metric_rows = []

    with torch.no_grad():
        for phase_index, phase in enumerate(phases):
            target_case = load_case(
                args.mesh_series_dir / f"phase_{int(round(phase)):02d}"
            )

            state = field.build_subset_state(
                phase / 100.0,
                torch_indices,
            )
            base_latent = canonical_latent[unique_neighbors]
            dynamic_xz = (
                base_latent[:, :2]
                + state["delta_xz"].detach().cpu().numpy()
            )
            dynamic_m = state["dynamic_m"].detach().cpu().numpy().reshape(-1)
            dynamic_latent = np.column_stack([dynamic_xz, dynamic_m])
            dynamic_patient = transform.latent_to_patient_xyz(dynamic_latent)
            local_gaussian_displacement = (
                dynamic_patient - canonical_patient[unique_neighbors]
            )
            local_magnitude = np.linalg.norm(local_gaussian_displacement, axis=1)

            neighbor_displacements = local_gaussian_displacement[
                local_neighbor_indices
            ]
            transfer = robust_distance_weights(
                neighbor_displacements,
                neighbor_distances,
                neighbor_confidence=neighbor_confidence,
                robust_z=args.robust_z,
                min_inliers=args.min_inliers,
            )
            vertex_displacement = transfer.displacement_xyz_mm
            predicted_vertices = canonical_mesh.vertices_xyz + vertex_displacement

            baseline_mesh = mesh_in_geometry(
                canonical_mesh.vertices_xyz,
                canonical_mesh.faces,
                target_case["geometry"],
            )
            predicted_mesh = mesh_in_geometry(
                predicted_vertices,
                canonical_mesh.faces,
                target_case["geometry"],
            )

            baseline_metrics = evaluate_mesh_against_reference(
                baseline_mesh,
                target_case["mesh"],
                target_case["mask"],
                target_case["geometry"],
                surface_samples=args.surface_samples,
                seed=args.seed + phase_index * 100,
            )
            predicted_metrics = evaluate_mesh_against_reference(
                predicted_mesh,
                target_case["mesh"],
                target_case["mask"],
                target_case["geometry"],
                surface_samples=args.surface_samples,
                seed=args.seed + phase_index * 100 + 10,
            )

            vertex_magnitude = np.linalg.norm(vertex_displacement, axis=1)
            row = {
                "PhasePercent": phase,
                "Split": split_by_phase.get(phase, "unknown"),
                "ReferenceROI": target_case["manifest"]["roi_name"],
                "CanonicalVertexCount": int(len(canonical_mesh.vertices_xyz)),
                "CanonicalFaceCount": int(len(canonical_mesh.faces)),
                "NeighborhoodGaussianCount": int(len(unique_neighbors)),
                "MeanInlierGaussianCount": float(transfer.inlier_mask.sum(axis=1).mean()),
                **percentile_columns(vertex_magnitude, "VertexDisplacementMm"),
                **prefix_metrics(baseline_metrics, "Baseline"),
                **prefix_metrics(predicted_metrics, "Predicted"),
            }
            row["DiceImprovement"] = row["PredictedDice"] - row["BaselineDice"]
            row["IoUImprovement"] = row["PredictedIoU"] - row["BaselineIoU"]
            row["MeanSurfaceDistanceReductionMm"] = (
                row["BaselineMeanSurfaceDistanceMm"]
                - row["PredictedMeanSurfaceDistanceMm"]
            )
            row["HD95ReductionMm"] = row["BaselineHD95Mm"] - row["PredictedHD95Mm"]
            row["CentroidErrorReductionMm"] = (
                row["BaselineCentroidErrorMm"]
                - row["PredictedCentroidErrorMm"]
            )
            metric_rows.append(row)

            local_rows.append(
                {
                    "PhasePercent": phase,
                    "GaussianCount": int(len(unique_neighbors)),
                    **percentile_columns(local_magnitude, "LocalGaussianDisplacementMm"),
                    "MeanDeltaPatientXmm": float(local_gaussian_displacement[:, 0].mean()),
                    "MeanDeltaPatientYmm": float(local_gaussian_displacement[:, 1].mean()),
                    "MeanDeltaPatientZmm": float(local_gaussian_displacement[:, 2].mean()),
                }
            )

            save_phase_artifacts(
                output_dir,
                phase,
                predicted_mesh,
                predicted_metrics["CandidateMask"],
                vertex_displacement,
                transfer.inlier_mask,
                transfer.normalized_weights,
                neighbor_indices,
                {
                    **prefix_metrics(baseline_metrics, "Baseline"),
                    **prefix_metrics(predicted_metrics, "Predicted"),
                },
            )

            print(
                f"Phase {phase:g}%: "
                f"Dice {row['BaselineDice']:.4f} -> {row['PredictedDice']:.4f}, "
                f"HD95 {row['BaselineHD95Mm']:.3f} -> "
                f"{row['PredictedHD95Mm']:.3f} mm"
            )

    per_phase = pd.DataFrame(metric_rows).sort_values("PhasePercent")
    local_table = pd.DataFrame(local_rows).sort_values("PhasePercent")
    write_dataframe(output_dir / "per_phase.csv", per_phase)
    write_dataframe(output_dir / "local_gaussian_deformation.csv", local_table)

    overall = {
        "RunDirectory": str(args.run_dir.resolve()),
        "Checkpoint": str(checkpoint.resolve()),
        "CheckpointIteration": int(iteration),
        "MeshSeriesDirectory": str(args.mesh_series_dir.resolve()),
        "GeometryDirectory": str(geometry_dir.resolve()),
        "CanonicalPhase": float(run.config.canonical_phase),
        "KnnK": int(neighbor_indices.shape[1]),
        "UniqueNeighborhoodGaussianCount": int(len(unique_neighbors)),
        "RobustZ": float(args.robust_z),
        "MinimumInliers": int(args.min_inliers),
        "OpacityPower": float(args.opacity_power),
        "SurfaceSamples": int(args.surface_samples),
        **aggregate_metrics(per_phase),
    }
    write_json(output_dir / "overall.json", overall)
    save_metrics_pdf(per_phase, output_dir / "metrics.pdf")
    write_json(
        output_dir / "config.json",
        {
            "run_dir": str(args.run_dir.resolve()),
            "mesh_series_dir": str(args.mesh_series_dir.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_iteration": int(iteration),
            "geometry_dir": str(geometry_dir.resolve()),
            "phases": phases,
            "robust_z": float(args.robust_z),
            "min_inliers": int(args.min_inliers),
            "opacity_power": float(args.opacity_power),
            "surface_samples": int(args.surface_samples),
            "seed": int(args.seed),
        },
    )

    print(json.dumps(overall, indent=2))
    print(f"Output directory: {output_dir}")
    print(f"Per-phase metrics: {output_dir / 'per_phase.csv'}")
    print(f"Overall metrics: {output_dir / 'overall.json'}")
    print(f"Metrics PDF: {output_dir / 'metrics.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
