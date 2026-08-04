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
            "Map canonical MedGS Gaussian coordinates to DICOM patient space "
            "and validate the mapping against the phase-0 RTSTRUCT mesh."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mesh-series-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--sample-gaussians", type=int, default=100000)
    parser.add_argument("--overlay-gaussians", type=int, default=20000)
    parser.add_argument("--neighborhood-margin-mm", type=float, default=20.0)
    parser.add_argument("--phases", type=float, nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


def percentile_dict(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}Mean": float(np.mean(values)),
        f"{prefix}Median": float(np.median(values)),
        f"{prefix}P95": float(np.percentile(values, 95)),
        f"{prefix}Maximum": float(np.max(values)),
    }


def checkpoint_iteration(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["iteration"])


def save_projection_overlay(
    path: Path,
    case: dict,
    canonical,
    transform,
    *,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    from medgs4d.gaussian_geometry import (
        effective_gaussian_xz_at_slice,
        project_medgs_xz_to_pixel,
    )
    from medgs4d.mesh_validation import window_hu

    mask = case["mask"]
    ct = case["ct_volume"]
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    occupied = np.flatnonzero(areas)
    selected_slices = [
        int(occupied[0]),
        int(np.argmax(areas)),
        int(occupied[-1]),
    ]

    base_xz = canonical.xz.detach().cpu().numpy()
    m = canonical.m.detach().cpu().numpy()
    sigma = canonical.gaussians.get_sigma.detach().cpu().numpy()
    weights = canonical.gaussians._w1.detach().cpu().numpy()
    opacity = canonical.gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
    time_steps = canonical.gaussians.get_time.detach().cpu().numpy().reshape(-1)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, slice_index in zip(axes, selected_slices):
        effective_xz, visibility = effective_gaussian_xz_at_slice(
            base_xz,
            m,
            sigma,
            weights,
            canonical.gaussians.polynomial_degree,
            slice_index,
            time_steps,
        )
        points_rc = project_medgs_xz_to_pixel(
            effective_xz,
            transform.full_projection,
            transform.image_width,
            transform.image_height,
        )
        score = visibility * opacity
        inside = (
            (points_rc[:, 0] >= 0)
            & (points_rc[:, 0] < transform.image_height)
            & (points_rc[:, 1] >= 0)
            & (points_rc[:, 1] < transform.image_width)
            & (visibility > 0.1)
        )
        indices = np.flatnonzero(inside)
        if len(indices) > max_points:
            local = np.argpartition(score[indices], -max_points)[-max_points:]
            indices = indices[local]

        axis.imshow(window_hu(ct[slice_index]), cmap="gray")
        axis.contour(mask[slice_index], levels=[0.5], colors="red", linewidths=1.5)
        axis.scatter(
            points_rc[indices, 1],
            points_rc[indices, 0],
            s=1.0,
            alpha=0.12,
        )
        axis.set_title(
            f"slice {slice_index}: {len(indices):,} visible Gaussian centers"
        )
        axis.set_xlim(0, transform.image_width - 1)
        axis.set_ylim(transform.image_height - 1, 0)
        axis.set_axis_off()

    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_neighborhood_overview(
    path: Path,
    gaussian_xyz: np.ndarray,
    mesh_vertices: np.ndarray,
    *,
    margin_mm: float,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    lower = mesh_vertices.min(axis=0) - margin_mm
    upper = mesh_vertices.max(axis=0) + margin_mm
    neighborhood = np.logical_and(
        gaussian_xyz >= lower,
        gaussian_xyz <= upper,
    ).all(axis=1)
    indices = np.flatnonzero(neighborhood)

    rng = np.random.default_rng(seed)
    if len(indices) > sample_size:
        indices = np.sort(rng.choice(indices, size=sample_size, replace=False))

    points = gaussian_xyz[indices]
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.12)
    axis.scatter(
        mesh_vertices[:, 0],
        mesh_vertices[:, 1],
        mesh_vertices[:, 2],
        s=2,
        alpha=0.8,
    )
    axis.set_xlabel("Patient x [mm]")
    axis.set_ylabel("Patient y [mm]")
    axis.set_zlabel("Patient z [mm]")
    axis.set_title("Canonical Gaussian centers around the phase-0 tumor mesh")
    axis.set_box_aspect(np.maximum(upper - lower, 1e-6))
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return indices


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from scipy.spatial import cKDTree

    from medgs4d.gaussian_geometry import (
        GaussianDicomTransform,
        canonical_latent_centers,
        inside_volume_mask,
        latent_centers_from_state,
    )
    from medgs4d.mesh_validation import load_case
    from medgs4d.results import load_run, load_run_models
    from medgs4d.runs import write_dataframe, write_json

    run = load_run(run_dir=args.run_dir)
    checkpoint = args.checkpoint or run.checkpoint
    if checkpoint is None:
        raise FileNotFoundError(f"No dynamic checkpoint found in {args.run_dir}")
    iteration = checkpoint_iteration(checkpoint)

    canonical_phase = int(round(run.config.canonical_phase))
    case_dir = args.mesh_series_dir / f"phase_{canonical_phase:02d}"
    case = load_case(case_dir)
    canonical, field = load_run_models(
        run,
        checkpoint=checkpoint,
        device=args.device,
    )

    output_dir = args.output_dir or (
        args.run_dir
        / "evaluation"
        / "gaussian_geometry"
        / f"iter_{iteration:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = GaussianDicomTransform.from_canonical(
        canonical,
        case["geometry"],
    )
    canonical_latent = canonical_latent_centers(canonical)
    canonical_indices = transform.latent_to_index_zyx(canonical_latent)
    canonical_patient = transform.latent_to_patient_xyz(canonical_latent)
    inside = inside_volume_mask(canonical_indices, case["geometry"].shape_zyx)

    mesh_vertices = case["mesh"].vertices_xyz
    mesh_roundtrip = transform.patient_roundtrip(mesh_vertices)
    roundtrip_error = np.linalg.norm(mesh_roundtrip - mesh_vertices, axis=1)

    tree_indices = np.flatnonzero(inside)
    tree = cKDTree(canonical_patient[tree_indices])
    distances, local_neighbors = tree.query(mesh_vertices, k=args.knn_k)
    if distances.ndim == 1:
        distances = distances[:, None]
        local_neighbors = local_neighbors[:, None]
    neighbor_indices = tree_indices[local_neighbors]

    vertex_table = pd.DataFrame(
        {
            "VertexIndex": np.arange(len(mesh_vertices), dtype=np.int64),
            "NearestGaussianDistanceMm": distances[:, 0],
            "KthGaussianDistanceMm": distances[:, -1],
            "GaussianCountWithin5Mm": [
                len(tree.query_ball_point(point, 5.0)) for point in mesh_vertices
            ],
            "GaussianCountWithin10Mm": [
                len(tree.query_ball_point(point, 10.0)) for point in mesh_vertices
            ],
        }
    )
    write_dataframe(output_dir / "mesh_vertex_knn.csv", vertex_table)
    np.save(output_dir / "mesh_vertex_knn_indices.npy", neighbor_indices)
    np.save(output_dir / "mesh_vertex_knn_distances_mm.npy", distances)

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sample_gaussians, len(canonical_patient))
    sample_indices = np.sort(
        rng.choice(len(canonical_patient), size=sample_count, replace=False)
    )
    np.savez_compressed(
        output_dir / "canonical_gaussian_sample.npz",
        gaussian_indices=sample_indices,
        latent_xzm=canonical_latent[sample_indices].astype(np.float32),
        patient_xyz=canonical_patient[sample_indices].astype(np.float32),
        index_zyx=canonical_indices[sample_indices].astype(np.float32),
        inside_volume=inside[sample_indices],
    )

    neighborhood_indices = save_neighborhood_overview(
        output_dir / "tumor_neighborhood_overview.png",
        canonical_patient,
        mesh_vertices,
        margin_mm=args.neighborhood_margin_mm,
        sample_size=min(args.sample_gaussians, 50000),
        seed=args.seed,
    )
    np.save(output_dir / "tumor_neighborhood_gaussian_indices.npy", neighborhood_indices)

    save_projection_overlay(
        output_dir / "projection_overlay.png",
        case,
        canonical,
        transform,
        max_points=args.overlay_gaussians,
    )

    phases = args.phases or run.study.phases
    deformation_rows = []
    canonical_patient_float = canonical_patient.astype(np.float64)
    import torch

    with torch.no_grad():
        for phase in phases:
            _, state = field.build_phase_state(
                float(phase) / 100.0,
                use_checkpointing=False,
            )
            dynamic_latent = latent_centers_from_state(
                state["dynamic_xyz"],
                state["dynamic_m"],
            )
            dynamic_patient = transform.latent_to_patient_xyz(dynamic_latent)
            displacement = dynamic_patient - canonical_patient_float
            magnitude = np.linalg.norm(displacement, axis=1)
            row = {
                "PhasePercent": float(phase),
                **percentile_dict(magnitude, "DisplacementMm"),
                "MeanDeltaPatientXmm": float(displacement[:, 0].mean()),
                "MeanDeltaPatientYmm": float(displacement[:, 1].mean()),
                "MeanDeltaPatientZmm": float(displacement[:, 2].mean()),
                "P95AbsDeltaPatientXmm": float(
                    np.percentile(np.abs(displacement[:, 0]), 95)
                ),
                "P95AbsDeltaPatientYmm": float(
                    np.percentile(np.abs(displacement[:, 1]), 95)
                ),
                "P95AbsDeltaPatientZmm": float(
                    np.percentile(np.abs(displacement[:, 2]), 95)
                ),
            }
            deformation_rows.append(row)
    deformation_table = pd.DataFrame(deformation_rows)
    write_dataframe(output_dir / "deformation_in_patient_mm.csv", deformation_table)

    summary = {
        "RunDirectory": str(args.run_dir.resolve()),
        "Checkpoint": str(checkpoint.resolve()),
        "CheckpointIteration": int(iteration),
        "CanonicalPhase": float(run.config.canonical_phase),
        "MeshCaseDirectory": str(case_dir.resolve()),
        "GaussianCount": int(len(canonical_latent)),
        "InsideVolumeGaussianCount": int(inside.sum()),
        "InsideVolumeGaussianFraction": float(inside.mean()),
        "ImageShapeHW": [transform.image_height, transform.image_width],
        "VolumeShapeZYX": list(case["geometry"].shape_zyx),
        "TimeStepCount": int(len(transform.time_steps)),
        "TimeStepMinimum": float(transform.time_steps.min()),
        "TimeStepMaximum": float(transform.time_steps.max()),
        "TimeStepSum": float(transform.time_steps.sum()),
        "CanonicalLatentBoundsXZM": [
            canonical_latent.min(axis=0).tolist(),
            canonical_latent.max(axis=0).tolist(),
        ],
        "MappedPatientBoundsXYZmm": [
            canonical_patient.min(axis=0).tolist(),
            canonical_patient.max(axis=0).tolist(),
        ],
        "MeshBoundsXYZmm": [
            mesh_vertices.min(axis=0).tolist(),
            mesh_vertices.max(axis=0).tolist(),
        ],
        "MeshPatientRoundtripMeanErrorMm": float(roundtrip_error.mean()),
        "MeshPatientRoundtripMaximumErrorMm": float(roundtrip_error.max()),
        "KnnK": int(args.knn_k),
        **percentile_dict(distances[:, 0], "NearestGaussianDistanceMm"),
        **percentile_dict(distances[:, -1], "KthGaussianDistanceMm"),
        "Orientation": transform.orientation_dict(),
        "Semantics": {
            "MedGS_x": "horizontal image coordinate projected to DICOM columns",
            "MedGS_z": "vertical image coordinate; positive z projects toward smaller image rows",
            "MedGS_m": (
                "continuous slice time; camera k uses cumulative time before step k"
            ),
            "Canonical_center": (
                "[x, z, m] is the Gaussian center because polynomial renderer "
                "offsets vanish when camera time equals m"
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"Output directory: {output_dir}")
    print(f"Projection overlay: {output_dir / 'projection_overlay.png'}")
    print(
        "Tumor neighborhood overview: "
        f"{output_dir / 'tumor_neighborhood_overview.png'}"
    )
    print(f"Patient-space deformation: {output_dir / 'deformation_in_patient_mm.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
