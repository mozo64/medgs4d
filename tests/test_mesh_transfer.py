import numpy as np

from medgs4d.mesh_transfer import (
    mesh_in_geometry,
    mesh_volume_centroid,
    robust_distance_weights,
    sampled_surface_distances,
    triangle_degenerate_count,
)
from medgs4d.rtstruct import VolumeGeometry


def simple_geometry() -> VolumeGeometry:
    return VolumeGeometry(
        shape_zyx=(20, 20, 20),
        origin_xyz=(0.0, 0.0, 0.0),
        column_direction_xyz=(1.0, 0.0, 0.0),
        row_direction_xyz=(0.0, 1.0, 0.0),
        slice_direction_xyz=(0.0, 0.0, 1.0),
        spacing_zyx=(1.0, 1.0, 1.0),
        slice_coordinates=tuple(float(i) for i in range(20)),
        sop_instance_uids=tuple(str(i) for i in range(20)),
        series_instance_uid="series",
        series_path="/tmp/series",
    )


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [2.0, 2.0, 2.0],
            [6.0, 2.0, 2.0],
            [2.0, 6.0, 2.0],
            [2.0, 2.0, 6.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_robust_transfer_rejects_one_large_outlier() -> None:
    displacement = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ]
    )
    distances = np.array([[1.0, 1.2, 1.3, 1.5, 1.1]])
    result = robust_distance_weights(
        displacement,
        distances,
        robust_z=3.5,
        min_inliers=3,
    )
    assert not result.inlier_mask[0, -1]
    assert np.allclose(result.displacement_xyz_mm[0], [1.0, 0.02, 0.0], atol=0.08)
    assert np.allclose(result.normalized_weights.sum(axis=1), 1.0)


def test_zero_neighbor_motion_preserves_vertices() -> None:
    displacement = np.zeros((4, 8, 3), dtype=np.float64)
    distances = np.tile(np.linspace(1.0, 4.0, 8), (4, 1))
    result = robust_distance_weights(displacement, distances)
    assert np.array_equal(result.displacement_xyz_mm, np.zeros((4, 3)))
    assert result.inlier_mask.all()


def test_patient_vertices_are_converted_to_target_geometry() -> None:
    vertices, faces = tetrahedron()
    mesh = mesh_in_geometry(vertices, faces, simple_geometry())
    assert np.allclose(mesh.vertices_zyx, vertices[:, [2, 1, 0]])
    assert np.array_equal(mesh.faces, faces)


def test_identical_surface_distance_is_zero() -> None:
    vertices, faces = tetrahedron()
    metrics = sampled_surface_distances(
        vertices,
        faces,
        vertices.copy(),
        faces.copy(),
        sample_count=1000,
        seed=7,
    )
    assert metrics == {"MeanSurfaceDistanceMm": 0.0, "HD95Mm": 0.0}


def test_volume_centroid_tracks_translation() -> None:
    vertices, faces = tetrahedron()
    translation = np.array([3.0, -2.0, 5.0])
    original = mesh_volume_centroid(vertices, faces)
    moved = mesh_volume_centroid(vertices + translation, faces)
    assert np.allclose(moved - original, translation)
    assert triangle_degenerate_count(vertices, faces) == 0


def cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [4.0, 4.0, 4.0],
            [8.0, 4.0, 4.0],
            [8.0, 8.0, 4.0],
            [4.0, 8.0, 4.0],
            [4.0, 4.0, 8.0],
            [8.0, 4.0, 8.0],
            [8.0, 8.0, 8.0],
            [4.0, 8.0, 8.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_identical_mesh_evaluation_has_perfect_overlap() -> None:
    from medgs4d.mesh_transfer import evaluate_mesh_against_reference
    from medgs4d.meshes import mesh_to_mask_roundtrip

    geometry = simple_geometry()
    vertices, faces = cube_mesh()
    reference = mesh_in_geometry(vertices, faces, geometry)
    mask = mesh_to_mask_roundtrip(reference.vertices_zyx, faces, geometry.shape_zyx)
    metrics = evaluate_mesh_against_reference(
        reference,
        reference,
        mask,
        geometry,
        surface_samples=1000,
        seed=11,
    )
    assert metrics["Dice"] == 1.0
    assert metrics["IoU"] == 1.0
    assert metrics["MeanSurfaceDistanceMm"] == 0.0
    assert metrics["HD95Mm"] == 0.0
    assert metrics["CentroidErrorMm"] == 0.0
    assert metrics["VolumeErrorPercent"] == 0.0
