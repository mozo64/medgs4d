#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
            "Transfer robust local MedGS4D Gaussian motion to the canonical "
            "RTSTRUCT tumor mesh and evaluate all respiratory phases."
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
    parser.add_argument("--global-minimum-fraction", type=float, default=0.25)
    parser.add_argument("--opacity-power", type=float, default=1.0)
    parser.add_argument("--local-detail-weight", type=float, default=0.25)
    parser.add_argument("--smoothing-iterations", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--max-displacement-mm", type=float, default=20.0)
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-ply", action="store_true")
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
    return {
        "AllNoncanonical": summarize(noncanonical),
        "BySplit": {
            split: summarize(group)
            for split, group in table.groupby("Split")
        },
    }


def save_metrics_png(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specifications = (
        ("Dice", "Dice", "Higher is better"),
        ("HD95Mm", "HD95 [mm]", "Lower is better"),
        ("CentroidErrorMm", "Centroid error [mm]", "Lower is better"),
    )
    for axis, (metric, ylabel, title) in zip(axes.ravel()[:3], specifications):
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
            label="Gaussian transfer",
        )
        axis.set_xlabel("Respiratory phase [%]")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linewidth=0.5, alpha=0.3)
        axis.legend(frameon=False)

    axis = axes.ravel()[3]
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
    axis.set_ylabel("Vertex displacement [mm]")
    axis.set_title("Transferred mesh motion")
    axis.grid(True, linewidth=0.5, alpha=0.3)
    axis.legend(frameon=False)

    figure.suptitle("Gaussian-to-mesh transfer evaluation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def set_equal_3d_axes(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.55 * float(np.max(maximum - minimum))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def save_phase_comparison_png(
    path: Path,
    phase: float,
    canonical_mesh,
    predicted_mesh,
    reference_mesh,
    metrics: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(9, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    layers = (
        (canonical_mesh, "tab:blue", 0.08, "Canonical phase 0"),
        (reference_mesh, "tab:green", 0.22, "RTSTRUCT reference"),
        (predicted_mesh, "tab:red", 0.35, "Gaussian prediction"),
    )
    for mesh, color, alpha, label in layers:
        collection = Poly3DCollection(
            mesh.vertices_xyz[mesh.faces],
            alpha=alpha,
            facecolor=color,
            edgecolor="none",
            label=label,
        )
        axis.add_collection3d(collection)

    all_points = np.concatenate(
        [
            canonical_mesh.vertices_xyz,
            reference_mesh.vertices_xyz,
            predicted_mesh.vertices_xyz,
        ],
        axis=0,
    )
    set_equal_3d_axes(axis, all_points)
    axis.view_init(elev=22, azim=-58)
    axis.set_xlabel("Patient x [mm]")
    axis.set_ylabel("Patient y [mm]")
    axis.set_zlabel("Patient z [mm]")
    axis.set_title(
        f"Phase {phase:g}%\n"
        f"Dice {metrics['BaselineDice']:.3f} → {metrics['PredictedDice']:.3f}; "
        f"HD95 {metrics['BaselineHD95Mm']:.2f} → "
        f"{metrics['PredictedHD95Mm']:.2f} mm"
    )

    # Matplotlib cannot reliably build a legend from Poly3DCollection labels.
    from matplotlib.patches import Patch

    axis.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.3, label="Canonical phase 0"),
            Patch(facecolor="tab:green", alpha=0.5, label="RTSTRUCT reference"),
            Patch(facecolor="tab:red", alpha=0.6, label="Gaussian prediction"),
        ],
        loc="upper left",
        frameon=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from medgs4d.gaussian_geometry import (
        GaussianDicomTransform,
        canonical_latent_centers,
    )
    from medgs4d.mesh_transfer import (
        evaluate_mesh_against_reference,
        limit_displacement_magnitude,
        mesh_in_geometry,
        prefix_metrics,
        robust_global_gaussian_inliers,
        robust_weighted_median_transfer,
        smooth_vertex_displacements,
    )
    from medgs4d.mesh_validation import load_case
    from medgs4d.meshes import write_ply
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

    opacity = canonical.gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
    unique_confidence = np.maximum(opacity[unique_neighbors], 1e-8) ** float(
        args.opacity_power
    )
    neighbor_confidence = unique_confidence[local_neighbor_indices]

    phases = [float(value) for value in (args.phases or run.study.phases)]
    split_by_phase = phase_split_table(run.split_manifest)
    metric_rows = []
    predicted_vertices_by_phase = []
    displacement_by_phase = []

    with torch.no_grad():
        for phase_index, phase in enumerate(phases):
            target_case = load_case(
                args.mesh_series_dir / f"phase_{int(round(phase)):02d}"
            )

            state = field.build_subset_state(phase / 100.0, torch_indices)
            base_latent = canonical_latent[unique_neighbors]
            dynamic_xz = base_latent[:, :2] + state["delta_xz"].cpu().numpy()
            dynamic_m = state["dynamic_m"].cpu().numpy().reshape(-1)
            dynamic_latent = np.column_stack([dynamic_xz, dynamic_m])
            dynamic_patient = transform.latent_to_patient_xyz(dynamic_latent)
            local_gaussian_displacement = (
                dynamic_patient - canonical_patient[unique_neighbors]
            )
            local_gaussian_magnitude = np.linalg.norm(
                local_gaussian_displacement, axis=1
            )

            global_inliers, global_translation = robust_global_gaussian_inliers(
                local_gaussian_displacement,
                confidence=unique_confidence,
                robust_z=args.robust_z,
                minimum_fraction=args.global_minimum_fraction,
            )
            transfer = robust_weighted_median_transfer(
                local_gaussian_displacement[local_neighbor_indices],
                neighbor_distances,
                global_translation_xyz_mm=global_translation,
                global_neighbor_inliers=global_inliers[local_neighbor_indices],
                neighbor_confidence=neighbor_confidence,
                robust_z=args.robust_z,
                min_inliers=args.min_inliers,
                local_detail_weight=args.local_detail_weight,
            )
            vertex_displacement = smooth_vertex_displacements(
                transfer.displacement_xyz_mm,
                canonical_mesh.faces,
                iterations=args.smoothing_iterations,
                alpha=args.smoothing_alpha,
            )
            vertex_displacement = limit_displacement_magnitude(
                vertex_displacement,
                args.max_displacement_mm,
            )
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
                "GlobalInlierGaussianCount": int(global_inliers.sum()),
                "GlobalInlierGaussianFraction": float(global_inliers.mean()),
                "GlobalTranslationPatientXmm": float(global_translation[0]),
                "GlobalTranslationPatientYmm": float(global_translation[1]),
                "GlobalTranslationPatientZmm": float(global_translation[2]),
                "MeanInlierGaussianCount": float(
                    transfer.inlier_mask.sum(axis=1).mean()
                ),
                **percentile_columns(
                    local_gaussian_magnitude,
                    "LocalGaussianDisplacementMm",
                ),
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
            row["HD95ReductionMm"] = (
                row["BaselineHD95Mm"] - row["PredictedHD95Mm"]
            )
            row["CentroidErrorReductionMm"] = (
                row["BaselineCentroidErrorMm"]
                - row["PredictedCentroidErrorMm"]
            )
            metric_rows.append(row)
            predicted_vertices_by_phase.append(predicted_vertices)
            displacement_by_phase.append(vertex_displacement)

            if phase == float(canonical_phase):
                if not np.allclose(vertex_displacement, 0.0, atol=1e-8):
                    raise RuntimeError("Canonical phase produced non-zero mesh motion")
                if baseline_metrics["Dice"] < 0.999999 or predicted_metrics["Dice"] < 0.999999:
                    raise RuntimeError("Canonical phase voxel evaluation is not exact")

            save_phase_comparison_png(
                output_dir / "comparisons" / f"phase_{int(round(phase)):02d}.png",
                phase,
                baseline_mesh,
                predicted_mesh,
                target_case["mesh"],
                row,
            )
            if args.save_ply:
                ply_dir = output_dir / "meshes"
                ply_dir.mkdir(parents=True, exist_ok=True)
                write_ply(
                    ply_dir / f"phase_{int(round(phase)):02d}.ply",
                    predicted_mesh.vertices_xyz,
                    predicted_mesh.faces,
                )

            print(
                f"Phase {phase:g}%: "
                f"Dice {row['BaselineDice']:.4f} -> {row['PredictedDice']:.4f}, "
                f"HD95 {row['BaselineHD95Mm']:.3f} -> "
                f"{row['PredictedHD95Mm']:.3f} mm"
            )

    per_phase = pd.DataFrame(metric_rows).sort_values("PhasePercent")
    write_dataframe(output_dir / "per_phase.csv", per_phase)
    np.savez_compressed(
        output_dir / "predictions.npz",
        phases=np.asarray(phases, dtype=np.float64),
        vertices_xyz_mm=np.stack(predicted_vertices_by_phase),
        vertex_displacement_xyz_mm=np.stack(displacement_by_phase),
        faces=canonical_mesh.faces,
    )
    save_metrics_png(per_phase, output_dir / "metrics.png")

    summary = {
        "RunDirectory": str(args.run_dir.resolve()),
        "Checkpoint": str(checkpoint.resolve()),
        "CheckpointIteration": int(iteration),
        "MeshSeriesDirectory": str(args.mesh_series_dir.resolve()),
        "GeometryDirectory": str(geometry_dir.resolve()),
        "CanonicalPhase": float(run.config.canonical_phase),
        "KnnK": int(neighbor_indices.shape[1]),
        "UniqueNeighborhoodGaussianCount": int(len(unique_neighbors)),
        "Parameters": {
            "robust_z": float(args.robust_z),
            "minimum_inliers": int(args.min_inliers),
            "global_minimum_fraction": float(args.global_minimum_fraction),
            "opacity_power": float(args.opacity_power),
            "local_detail_weight": float(args.local_detail_weight),
            "smoothing_iterations": int(args.smoothing_iterations),
            "smoothing_alpha": float(args.smoothing_alpha),
            "maximum_displacement_mm": float(args.max_displacement_mm),
            "surface_samples": int(args.surface_samples),
            "seed": int(args.seed),
        },
        "Artifacts": {
            "per_phase": "per_phase.csv",
            "predictions": "predictions.npz",
            "metrics": "metrics.png",
            "comparisons": "comparisons/phase_XX.png",
            "ply_meshes": "meshes/phase_XX.ply" if args.save_ply else None,
        },
        **aggregate_metrics(per_phase),
    }
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"Output directory: {output_dir}")
    print(f"Per-phase metrics: {output_dir / 'per_phase.csv'}")
    print(f"Predicted mesh series: {output_dir / 'predictions.npz'}")
    print(f"Metrics PNG: {output_dir / 'metrics.png'}")
    print(f"Phase comparisons: {output_dir / 'comparisons'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
