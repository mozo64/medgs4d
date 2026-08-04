from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .meshes import (
    TriangleMesh,
    binary_dice,
    binary_iou,
    mesh_is_watertight,
    mesh_surface_area,
    mesh_to_mask_roundtrip,
    mesh_volume,
)
from .rtstruct import VolumeGeometry


@dataclass(frozen=True)
class VertexTransferResult:
    """Store interpolated vertex motion and robust-neighbor diagnostics."""

    displacement_xyz_mm: np.ndarray
    inlier_mask: np.ndarray
    normalized_weights: np.ndarray


def robust_distance_weights(
    neighbor_displacements_xyz_mm: np.ndarray,
    neighbor_distances_mm: np.ndarray,
    *,
    neighbor_confidence: np.ndarray | None = None,
    robust_z: float = 3.5,
    min_inliers: int = 4,
) -> VertexTransferResult:
    """Interpolate local Gaussian motion after rejecting displacement outliers.

    Neighbor motion is filtered independently for each mesh vertex. The robust
    center is the component-wise median displacement. Gaussian distance weights
    are multiplied by optional confidence values, normally canonical opacity.
    """

    displacement = np.asarray(neighbor_displacements_xyz_mm, dtype=np.float64)
    distances = np.asarray(neighbor_distances_mm, dtype=np.float64)
    if displacement.ndim != 3 or displacement.shape[-1] != 3:
        raise ValueError(
            "neighbor_displacements_xyz_mm must have shape [vertices, k, 3]"
        )
    if distances.shape != displacement.shape[:2]:
        raise ValueError("neighbor_distances_mm must have shape [vertices, k]")

    vertex_count, neighbor_count = distances.shape
    required = min(max(int(min_inliers), 1), neighbor_count)

    center = np.median(displacement, axis=1)
    residual = np.linalg.norm(displacement - center[:, None, :], axis=-1)
    residual_median = np.median(residual, axis=1)
    mad = np.median(
        np.abs(residual - residual_median[:, None]),
        axis=1,
    )
    robust_scale = np.maximum(1.4826 * mad, 1e-6)
    threshold = residual_median + float(robust_z) * robust_scale
    inliers = residual <= threshold[:, None]

    insufficient = np.flatnonzero(inliers.sum(axis=1) < required)
    if len(insufficient):
        nearest_to_center = np.argsort(residual[insufficient], axis=1)[:, :required]
        inliers[insufficient] = False
        inliers[
            insufficient[:, None],
            nearest_to_center,
        ] = True

    bandwidth = np.maximum(np.median(distances, axis=1), 1e-6)
    weights = np.exp(-0.5 * (distances / bandwidth[:, None]) ** 2)

    if neighbor_confidence is not None:
        confidence = np.asarray(neighbor_confidence, dtype=np.float64)
        if confidence.shape != distances.shape:
            raise ValueError("neighbor_confidence must have shape [vertices, k]")
        weights *= np.maximum(confidence, 1e-8)

    weights *= inliers
    weight_sum = weights.sum(axis=1, keepdims=True)
    zero_weight = weight_sum[:, 0] <= 0.0
    if np.any(zero_weight):
        weights[zero_weight] = inliers[zero_weight].astype(np.float64)
        weight_sum = weights.sum(axis=1, keepdims=True)

    normalized = weights / weight_sum
    interpolated = np.sum(normalized[..., None] * displacement, axis=1)

    return VertexTransferResult(
        displacement_xyz_mm=interpolated,
        inlier_mask=inliers,
        normalized_weights=normalized,
    )


def mesh_in_geometry(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    geometry: VolumeGeometry,
) -> TriangleMesh:
    """Create one mesh in patient coordinates and a selected CT voxel grid."""

    vertices = np.asarray(vertices_xyz, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    return TriangleMesh(
        vertices_zyx=geometry.patient_xyz_to_index_zyx(vertices),
        vertices_xyz=vertices,
        faces=triangles,
    )


def triangle_degenerate_count(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    *,
    area_tolerance_mm2: float = 1e-8,
) -> int:
    triangles = np.asarray(vertices_xyz, dtype=np.float64)[
        np.asarray(faces, dtype=np.int64)
    ]
    double_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    return int(np.count_nonzero(0.5 * double_area <= area_tolerance_mm2))


def mesh_volume_centroid(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Return the centroid of the signed tetrahedral decomposition."""

    triangles = np.asarray(vertices_xyz, dtype=np.float64)[
        np.asarray(faces, dtype=np.int64)
    ]
    signed_volume = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    ) / 6.0
    total = signed_volume.sum()
    if abs(total) < 1e-12:
        return np.asarray(vertices_xyz, dtype=np.float64).mean(axis=0)
    tetrahedron_centroids = triangles.sum(axis=1) / 4.0
    return (signed_volume[:, None] * tetrahedron_centroids).sum(axis=0) / total


def sample_mesh_surface(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    sample_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample points uniformly by triangle area."""

    vertices = np.asarray(vertices_xyz, dtype=np.float64)
    triangles = vertices[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probabilities = areas / areas.sum()

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(triangles), size=int(sample_count), p=probabilities)
    chosen = triangles[selected]
    first = np.sqrt(rng.random(sample_count))
    second = rng.random(sample_count)
    return (
        (1.0 - first)[:, None] * chosen[:, 0]
        + (first * (1.0 - second))[:, None] * chosen[:, 1]
        + (first * second)[:, None] * chosen[:, 2]
    )


def sampled_surface_distances(
    left_vertices_xyz: np.ndarray,
    left_faces: np.ndarray,
    right_vertices_xyz: np.ndarray,
    right_faces: np.ndarray,
    *,
    sample_count: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute symmetric sampled surface distance and HD95 in millimeters."""

    left_vertices = np.asarray(left_vertices_xyz, dtype=np.float64)
    right_vertices = np.asarray(right_vertices_xyz, dtype=np.float64)
    left_triangles = np.asarray(left_faces, dtype=np.int64)
    right_triangles = np.asarray(right_faces, dtype=np.int64)

    if (
        left_vertices.shape == right_vertices.shape
        and left_triangles.shape == right_triangles.shape
        and np.array_equal(left_triangles, right_triangles)
        and np.allclose(left_vertices, right_vertices, atol=1e-12)
    ):
        return {
            "MeanSurfaceDistanceMm": 0.0,
            "HD95Mm": 0.0,
        }

    from scipy.spatial import cKDTree

    left_points = sample_mesh_surface(
        left_vertices,
        left_triangles,
        sample_count,
        seed=seed,
    )
    right_points = sample_mesh_surface(
        right_vertices,
        right_triangles,
        sample_count,
        seed=seed + 1,
    )
    left_to_right = cKDTree(right_points).query(left_points, k=1)[0]
    right_to_left = cKDTree(left_points).query(right_points, k=1)[0]
    combined = np.concatenate([left_to_right, right_to_left])
    return {
        "MeanSurfaceDistanceMm": float(combined.mean()),
        "HD95Mm": float(np.percentile(combined, 95)),
    }


def evaluate_mesh_against_reference(
    candidate: TriangleMesh,
    reference: TriangleMesh,
    reference_mask: np.ndarray,
    geometry: VolumeGeometry,
    *,
    surface_samples: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate one candidate mesh against an RTSTRUCT-derived reference."""

    candidate_mask = mesh_to_mask_roundtrip(
        candidate.vertices_zyx,
        candidate.faces,
        geometry.shape_zyx,
    )
    candidate_centroid = mesh_volume_centroid(
        candidate.vertices_xyz,
        candidate.faces,
    )
    reference_centroid = mesh_volume_centroid(
        reference.vertices_xyz,
        reference.faces,
    )
    candidate_volume = mesh_volume(candidate.vertices_xyz, candidate.faces)
    reference_volume = mesh_volume(reference.vertices_xyz, reference.faces)
    candidate_area = mesh_surface_area(candidate.vertices_xyz, candidate.faces)
    reference_area = mesh_surface_area(reference.vertices_xyz, reference.faces)

    surface = sampled_surface_distances(
        candidate.vertices_xyz,
        candidate.faces,
        reference.vertices_xyz,
        reference.faces,
        sample_count=surface_samples,
        seed=seed,
    )

    return {
        "Dice": binary_dice(candidate_mask, reference_mask),
        "IoU": binary_iou(candidate_mask, reference_mask),
        **surface,
        "CentroidErrorMm": float(
            np.linalg.norm(candidate_centroid - reference_centroid)
        ),
        "VolumeMm3": candidate_volume,
        "ReferenceVolumeMm3": reference_volume,
        "VolumeErrorMm3": float(candidate_volume - reference_volume),
        "VolumeErrorPercent": float(
            100.0 * (candidate_volume - reference_volume) / reference_volume
        ),
        "SurfaceAreaMm2": candidate_area,
        "ReferenceSurfaceAreaMm2": reference_area,
        "SurfaceAreaErrorPercent": float(
            100.0 * (candidate_area - reference_area) / reference_area
        ),
        "Watertight": mesh_is_watertight(candidate.faces),
        "DegenerateFaceCount": triangle_degenerate_count(
            candidate.vertices_xyz,
            candidate.faces,
        ),
        "MaskVoxelCount": int(candidate_mask.sum()),
        "CandidateMask": candidate_mask,
    }


def prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in metrics.items()
        if key != "CandidateMask"
    }
