from __future__ import annotations

from dataclasses import dataclass
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
    """Interpolated mesh motion and diagnostics for one respiratory phase."""

    raw_displacement_xyz_mm: np.ndarray
    displacement_xyz_mm: np.ndarray
    inlier_mask: np.ndarray
    normalized_weights: np.ndarray
    global_translation_xyz_mm: np.ndarray


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute a weighted median along axis 1."""

    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if data.shape != mass.shape:
        raise ValueError("values and weights must have the same shape")
    if data.ndim != 2:
        raise ValueError("values and weights must have shape [rows, samples]")

    order = np.argsort(data, axis=1)
    sorted_data = np.take_along_axis(data, order, axis=1)
    sorted_mass = np.take_along_axis(mass, order, axis=1)
    cumulative = np.cumsum(sorted_mass, axis=1)
    half = 0.5 * sorted_mass.sum(axis=1, keepdims=True)
    index = np.argmax(cumulative >= half, axis=1)
    return sorted_data[np.arange(len(sorted_data)), index]


def weighted_median_vector(values_xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute component-wise weighted medians for [rows, samples, 3]."""

    values = np.asarray(values_xyz, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("values_xyz must have shape [rows, samples, 3]")
    if mass.shape != values.shape[:2]:
        raise ValueError("weights must have shape [rows, samples]")
    return np.column_stack(
        [weighted_median(values[:, :, axis], mass) for axis in range(3)]
    )


def robust_global_gaussian_inliers(
    displacements_xyz_mm: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    robust_z: float = 3.5,
    minimum_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the coherent motion mode among Gaussian neighbors of the tumor."""

    displacement = np.asarray(displacements_xyz_mm, dtype=np.float64)
    if displacement.ndim != 2 or displacement.shape[1] != 3:
        raise ValueError("displacements_xyz_mm must have shape [gaussians, 3]")

    weights = (
        np.ones(len(displacement), dtype=np.float64)
        if confidence is None
        else np.maximum(np.asarray(confidence, dtype=np.float64), 1e-12)
    )
    if weights.shape != (len(displacement),):
        raise ValueError("confidence must have shape [gaussians]")

    center = weighted_median_vector(
        displacement[None, :, :],
        weights[None, :],
    )[0]
    residual = np.linalg.norm(displacement - center, axis=1)
    residual_median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_median)))
    threshold = residual_median + float(robust_z) * max(1.4826 * mad, 1e-6)
    inliers = residual <= threshold

    required = min(
        len(displacement),
        max(1, int(np.ceil(float(minimum_fraction) * len(displacement)))),
    )
    if int(inliers.sum()) < required:
        selected = np.argsort(residual)[:required]
        inliers[:] = False
        inliers[selected] = True

    translation = weighted_median_vector(
        displacement[inliers][None, :, :],
        weights[inliers][None, :],
    )[0]
    return inliers, translation


def robust_weighted_median_transfer(
    neighbor_displacements_xyz_mm: np.ndarray,
    neighbor_distances_mm: np.ndarray,
    *,
    global_translation_xyz_mm: np.ndarray,
    global_neighbor_inliers: np.ndarray | None = None,
    neighbor_confidence: np.ndarray | None = None,
    robust_z: float = 3.5,
    min_inliers: int = 4,
    local_detail_weight: float = 0.25,
) -> VertexTransferResult:
    """Transfer coherent local Gaussian motion to canonical mesh vertices.

    The estimator uses a component-wise weighted median instead of a weighted
    mean. It combines a robust phase-wide translation with a reduced local
    residual, which is suitable for a small tumor expected to move mostly as a
    coherent object.
    """

    displacement = np.asarray(neighbor_displacements_xyz_mm, dtype=np.float64)
    distances = np.asarray(neighbor_distances_mm, dtype=np.float64)
    if displacement.ndim != 3 or displacement.shape[-1] != 3:
        raise ValueError(
            "neighbor_displacements_xyz_mm must have shape [vertices, k, 3]"
        )
    if distances.shape != displacement.shape[:2]:
        raise ValueError("neighbor_distances_mm must have shape [vertices, k]")

    _, neighbor_count = distances.shape
    required = min(max(int(min_inliers), 1), neighbor_count)

    bandwidth = np.maximum(np.median(distances, axis=1), 1e-6)
    weights = np.exp(-0.5 * (distances / bandwidth[:, None]) ** 2)

    if neighbor_confidence is not None:
        confidence = np.asarray(neighbor_confidence, dtype=np.float64)
        if confidence.shape != distances.shape:
            raise ValueError("neighbor_confidence must have shape [vertices, k]")
        weights *= np.maximum(confidence, 1e-12)

    local_center = np.median(displacement, axis=1)
    residual = np.linalg.norm(displacement - local_center[:, None, :], axis=-1)
    residual_median = np.median(residual, axis=1)
    mad = np.median(np.abs(residual - residual_median[:, None]), axis=1)
    threshold = residual_median + float(robust_z) * np.maximum(1.4826 * mad, 1e-6)
    inliers = residual <= threshold[:, None]

    if global_neighbor_inliers is not None:
        global_inliers = np.asarray(global_neighbor_inliers, dtype=bool)
        if global_inliers.shape != distances.shape:
            raise ValueError("global_neighbor_inliers must have shape [vertices, k]")
        inliers &= global_inliers

    insufficient = np.flatnonzero(inliers.sum(axis=1) < required)
    if len(insufficient):
        candidate_score = residual[insufficient] + distances[insufficient]
        selected = np.argsort(candidate_score, axis=1)[:, :required]
        inliers[insufficient] = False
        inliers[insufficient[:, None], selected] = True

    weights *= inliers
    weight_sum = weights.sum(axis=1, keepdims=True)
    zero_weight = weight_sum[:, 0] <= 0.0
    if np.any(zero_weight):
        weights[zero_weight] = inliers[zero_weight].astype(np.float64)
        weight_sum = weights.sum(axis=1, keepdims=True)
    normalized = weights / weight_sum

    local = weighted_median_vector(displacement, normalized)
    translation = np.asarray(global_translation_xyz_mm, dtype=np.float64).reshape(3)
    raw = translation + float(local_detail_weight) * (local - translation)

    return VertexTransferResult(
        raw_displacement_xyz_mm=raw,
        displacement_xyz_mm=raw.copy(),
        inlier_mask=inliers,
        normalized_weights=normalized,
        global_translation_xyz_mm=translation,
    )


def smooth_vertex_displacements(
    displacement_xyz_mm: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 8,
    alpha: float = 0.35,
) -> np.ndarray:
    """Laplacian-smooth mesh motion while preserving its mean translation."""

    displacement = np.asarray(displacement_xyz_mm, dtype=np.float64).copy()
    if int(iterations) <= 0 or float(alpha) <= 0.0:
        return displacement

    from scipy.sparse import coo_matrix

    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 0]],
            triangles[:, [1, 2]],
            triangles[:, [2, 1]],
            triangles[:, [2, 0]],
            triangles[:, [0, 2]],
        ],
        axis=0,
    )
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(displacement), len(displacement)),
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degree = np.maximum(degree, 1.0)

    target_mean = displacement.mean(axis=0)
    for _ in range(int(iterations)):
        neighbor_mean = adjacency @ displacement / degree[:, None]
        displacement = (1.0 - float(alpha)) * displacement + float(alpha) * neighbor_mean
        displacement += target_mean - displacement.mean(axis=0)
    return displacement


def limit_displacement_magnitude(
    displacement_xyz_mm: np.ndarray,
    maximum_mm: float | None,
) -> np.ndarray:
    """Apply a final safety bound to interpolated vertex motion."""

    displacement = np.asarray(displacement_xyz_mm, dtype=np.float64).copy()
    if maximum_mm is None or float(maximum_mm) <= 0.0:
        return displacement
    magnitude = np.linalg.norm(displacement, axis=1)
    scale = np.minimum(1.0, float(maximum_mm) / np.maximum(magnitude, 1e-12))
    return displacement * scale[:, None]


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
        and np.allclose(left_vertices, right_vertices, atol=1e-10)
    ):
        return {"MeanSurfaceDistanceMm": 0.0, "HD95Mm": 0.0}

    from scipy.spatial import cKDTree

    left_points = sample_mesh_surface(
        left_vertices, left_triangles, sample_count, seed=seed
    )
    right_points = sample_mesh_surface(
        right_vertices, right_triangles, sample_count, seed=seed + 1
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

    reference_roundtrip = mesh_to_mask_roundtrip(
        reference.vertices_zyx,
        reference.faces,
        geometry.shape_zyx,
    )
    reference_roundtrip_dice = binary_dice(reference_roundtrip, reference_mask)
    if reference_roundtrip_dice < 0.99:
        raise RuntimeError(
            "Reference mesh voxelization is inconsistent with its source mask: "
            f"Dice={reference_roundtrip_dice:.6f}"
        )

    identical = (
        candidate.vertices_xyz.shape == reference.vertices_xyz.shape
        and candidate.faces.shape == reference.faces.shape
        and np.array_equal(candidate.faces, reference.faces)
        and np.allclose(candidate.vertices_xyz, reference.vertices_xyz, atol=1e-10)
    )
    candidate_mask = (
        np.asarray(reference_mask, dtype=bool).copy()
        if identical
        else mesh_to_mask_roundtrip(
            candidate.vertices_zyx,
            candidate.faces,
            geometry.shape_zyx,
        )
    )

    candidate_centroid = mesh_volume_centroid(candidate.vertices_xyz, candidate.faces)
    reference_centroid = mesh_volume_centroid(reference.vertices_xyz, reference.faces)
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
        "CentroidErrorMm": float(np.linalg.norm(candidate_centroid - reference_centroid)),
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
            candidate.vertices_xyz, candidate.faces
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
