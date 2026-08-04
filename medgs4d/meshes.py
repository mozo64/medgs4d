from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .rtstruct import VolumeGeometry


@dataclass(frozen=True)
class TriangleMesh:
    """Triangle mesh in both voxel-index and DICOM patient coordinates."""

    vertices_zyx: np.ndarray
    vertices_xyz: np.ndarray
    faces: np.ndarray


def mask_to_mesh(mask: np.ndarray, geometry: VolumeGeometry) -> TriangleMesh:
    """Extract a closed level-0.5 surface from a binary mask."""

    from skimage.measure import marching_cubes

    binary = np.asarray(mask, dtype=bool)
    if binary.shape != geometry.shape_zyx:
        raise ValueError(
            f"Mask shape {binary.shape} does not match CT shape {geometry.shape_zyx}"
        )
    if not binary.any():
        raise ValueError("Cannot create a mesh from an empty mask")

    padded = np.pad(binary.astype(np.uint8), 1, mode="constant")
    vertices_zyx, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        allow_degenerate=False,
    )
    vertices_zyx -= 1.0
    vertices_xyz = geometry.index_zyx_to_patient_xyz(vertices_zyx)
    return TriangleMesh(
        vertices_zyx=vertices_zyx.astype(np.float64),
        vertices_xyz=vertices_xyz.astype(np.float64),
        faces=faces.astype(np.int64),
    )


def write_ply(path: Path, vertices_xyz: np.ndarray, faces: np.ndarray) -> None:
    """Write an ASCII PLY triangle mesh without optional dependencies."""

    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices_xyz, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    with path.open("w", encoding="utf-8") as output:
        output.write("ply\n")
        output.write("format ascii 1.0\n")
        output.write(f"element vertex {len(vertices)}\n")
        output.write("property float x\nproperty float y\nproperty float z\n")
        output.write(f"element face {len(triangles)}\n")
        output.write("property list uchar int vertex_indices\n")
        output.write("end_header\n")
        for vertex in vertices:
            output.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in triangles:
            output.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_mesh_npz(path: Path, mesh: TriangleMesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices_zyx=mesh.vertices_zyx,
        vertices_xyz=mesh.vertices_xyz,
        faces=mesh.faces,
    )


def load_mesh_npz(path: Path) -> TriangleMesh:
    data = np.load(path)
    return TriangleMesh(
        vertices_zyx=data["vertices_zyx"],
        vertices_xyz=data["vertices_xyz"],
        faces=data["faces"],
    )


def mesh_surface_area(vertices_xyz: np.ndarray, faces: np.ndarray) -> float:
    triangles = np.asarray(vertices_xyz)[np.asarray(faces)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def mesh_volume(vertices_xyz: np.ndarray, faces: np.ndarray) -> float:
    """Return absolute enclosed volume from oriented triangle tetrahedra."""

    triangles = np.asarray(vertices_xyz)[np.asarray(faces)]
    signed = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    ) / 6.0
    return float(abs(signed.sum()))


def mesh_is_watertight(faces: np.ndarray) -> bool:
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=0,
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _triangle_plane_segment(triangle: np.ndarray, plane_z: float) -> np.ndarray | None:
    """Intersect one zyx triangle with z=plane_z and return two [y, x] points."""

    distances = triangle[:, 0] - plane_z
    points = []
    for start, end, d_start, d_end in (
        (triangle[0], triangle[1], distances[0], distances[1]),
        (triangle[1], triangle[2], distances[1], distances[2]),
        (triangle[2], triangle[0], distances[2], distances[0]),
    ):
        if d_start * d_end < 0:
            t = d_start / (d_start - d_end)
            point = start + t * (end - start)
            points.append(point[1:])
        elif abs(d_start) < 1e-10 and abs(d_end) >= 1e-10:
            points.append(start[1:])
        elif abs(d_end) < 1e-10 and abs(d_start) >= 1e-10:
            points.append(end[1:])

    if len(points) < 2:
        return None
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - other) < 1e-8 for other in unique):
            unique.append(point)
    if len(unique) != 2:
        return None
    return np.stack(unique, axis=0)


def mesh_to_mask_roundtrip(
    vertices_zyx: np.ndarray,
    faces: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Voxelize a closed mesh by polygonizing intersections at voxel-center slices."""

    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    from skimage.draw import polygon

    vertices = np.asarray(vertices_zyx, dtype=np.float64)
    triangles = vertices[np.asarray(faces, dtype=np.int64)]
    z_min = triangles[:, :, 0].min(axis=1)
    z_max = triangles[:, :, 0].max(axis=1)
    mask = np.zeros(shape_zyx, dtype=bool)

    for slice_index in range(shape_zyx[0]):
        plane_z = float(slice_index) + 1e-5
        candidate_indices = np.flatnonzero((z_min <= plane_z) & (z_max >= plane_z))
        lines = []
        for triangle in triangles[candidate_indices]:
            segment_yx = _triangle_plane_segment(triangle, plane_z)
            if segment_yx is None:
                continue
            # Adjacent triangles should yield identical intersection endpoints,
            # but patient-to-voxel transforms introduce tiny floating-point
            # differences. Quantization closes those contours before polygonize.
            segment_yx = np.round(segment_yx, decimals=6)
            if np.linalg.norm(segment_yx[0] - segment_yx[1]) < 1e-8:
                continue

            # Shapely coordinates are x, y; triangle coordinates are y, x.
            lines.append(
                LineString(
                    [
                        (float(segment_yx[0, 1]), float(segment_yx[0, 0])),
                        (float(segment_yx[1, 1]), float(segment_yx[1, 0])),
                    ]
                )
            )
        if not lines:
            continue

        polygons = list(polygonize(unary_union(lines)))
        slice_mask = np.zeros(shape_zyx[1:], dtype=bool)
        for region in polygons:
            exterior = np.asarray(region.exterior.coords)
            rr, cc = polygon(
                exterior[:, 1],
                exterior[:, 0],
                shape=shape_zyx[1:],
            )
            slice_mask[rr, cc] = True
            for interior in region.interiors:
                hole = np.asarray(interior.coords)
                rr, cc = polygon(
                    hole[:, 1],
                    hole[:, 0],
                    shape=shape_zyx[1:],
                )
                slice_mask[rr, cc] = False
        mask[slice_index] = slice_mask
    return mask


def binary_dice(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    denominator = int(left.sum()) + int(right.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(left, right).sum() / denominator)


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def mesh_validation_report(
    mask: np.ndarray,
    roundtrip_mask: np.ndarray,
    mesh: TriangleMesh,
    geometry: VolumeGeometry,
) -> dict[str, Any]:
    """Compute mask, mesh, and round-trip validation measurements."""

    spacing_z, spacing_y, spacing_x = geometry.spacing_zyx
    voxel_volume = spacing_z * spacing_y * spacing_x
    mask_volume = float(np.asarray(mask, dtype=bool).sum() * voxel_volume)
    enclosed_volume = mesh_volume(mesh.vertices_xyz, mesh.faces)
    relative_volume_error = (
        abs(enclosed_volume - mask_volume) / mask_volume if mask_volume else float("nan")
    )
    return {
        "shape_zyx": list(mask.shape),
        "spacing_zyx_mm": list(geometry.spacing_zyx),
        "mask_voxels": int(np.asarray(mask, dtype=bool).sum()),
        "mask_volume_mm3": mask_volume,
        "roundtrip_voxels": int(np.asarray(roundtrip_mask, dtype=bool).sum()),
        "roundtrip_dice": binary_dice(mask, roundtrip_mask),
        "roundtrip_iou": binary_iou(mask, roundtrip_mask),
        "mesh_vertices": int(len(mesh.vertices_xyz)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_surface_area_mm2": mesh_surface_area(mesh.vertices_xyz, mesh.faces),
        "mesh_volume_mm3": enclosed_volume,
        "mesh_mask_relative_volume_error": relative_volume_error,
        "mesh_watertight": mesh_is_watertight(mesh.faces),
        "mesh_bounds_xyz_mm": [
            mesh.vertices_xyz.min(axis=0).tolist(),
            mesh.vertices_xyz.max(axis=0).tolist(),
        ],
        "finite_vertices": bool(np.isfinite(mesh.vertices_xyz).all()),
    }


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
