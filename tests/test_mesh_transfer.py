import numpy as np

from medgs4d.mesh_transfer import (
    evaluate_mesh_against_reference,
    limit_displacement_magnitude,
    mesh_in_geometry,
    mesh_volume_centroid,
    robust_global_gaussian_inliers,
    robust_weighted_median_transfer,
    sampled_surface_distances,
    smooth_vertex_displacements,
    triangle_degenerate_count,
)
from medgs4d.meshes import binary_dice, mask_to_mesh, mesh_to_mask_roundtrip
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


def oblique_geometry() -> VolumeGeometry:
    return VolumeGeometry(
        shape_zyx=(24, 24, 24),
        origin_xyz=(-10.3, 5.2, 7.7),
        column_direction_xyz=(0.0, 1.0, 0.0),
        row_direction_xyz=(-1.0, 0.0, 0.0),
        slice_direction_xyz=(0.0, 0.0, 1.0),
        spacing_zyx=(2.5, 0.8, 0.8),
        slice_coordinates=tuple(7.7 + 2.5 * i for i in range(24)),
        sop_instance_uids=tuple(str(i) for i in range(24)),
        series_instance_uid="oblique",
        series_path="/tmp/oblique",
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


def test_global_gaussian_filter_rejects_large_motion_mode() -> None:
    coherent = np.tile([2.0, -1.0, 0.5], (20, 1))
    coherent += np.linspace(-0.2, 0.2, 20)[:, None]
    outliers = np.array(
        [[80.0, 0.0, 0.0], [0.0, -90.0, 0.0], [0.0, 0.0, 100.0]]
    )
    motion = np.concatenate([coherent, outliers], axis=0)
    inliers, translation = robust_global_gaussian_inliers(motion)
    assert inliers.sum() >= 20
    assert not inliers[-3:].any()
    assert np.allclose(translation, [2.0, -1.0, 0.5], atol=0.25)


def test_weighted_median_transfer_ignores_neighbor_outlier() -> None:
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
    result = robust_weighted_median_transfer(
        displacement,
        distances,
        global_translation_xyz_mm=np.array([1.0, 0.0, 0.0]),
        robust_z=3.5,
        min_inliers=3,
        local_detail_weight=1.0,
    )
    assert not result.inlier_mask[0, -1]
    assert np.allclose(result.displacement_xyz_mm[0], [1.0, 0.0, 0.0], atol=0.11)
    assert np.allclose(result.normalized_weights.sum(axis=1), 1.0)


def test_smoothing_preserves_constant_translation() -> None:
    vertices, faces = cube_mesh()
    displacement = np.tile([3.0, -2.0, 1.0], (len(vertices), 1))
    smoothed = smooth_vertex_displacements(
        displacement,
        faces,
        iterations=10,
        alpha=0.5,
    )
    assert np.allclose(smoothed, displacement)


def test_displacement_limit_applies_hard_safety_bound() -> None:
    displacement = np.array([[30.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    limited = limit_displacement_magnitude(displacement, 10.0)
    assert np.allclose(np.linalg.norm(limited, axis=1), [10.0, 5.0])


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


def test_mask_mesh_patient_roundtrip_is_exact_for_oblique_geometry() -> None:
    geometry = oblique_geometry()
    mask = np.zeros(geometry.shape_zyx, dtype=bool)
    mask[7:13, 8:16, 6:15] = True
    mesh = mask_to_mesh(mask, geometry)
    remapped = mesh_in_geometry(mesh.vertices_xyz, mesh.faces, geometry)
    roundtrip = mesh_to_mask_roundtrip(
        remapped.vertices_zyx,
        remapped.faces,
        geometry.shape_zyx,
    )
    assert binary_dice(mask, roundtrip) == 1.0


def test_identical_mesh_evaluation_has_perfect_overlap() -> None:
    geometry = simple_geometry()
    mask = np.zeros(geometry.shape_zyx, dtype=bool)
    mask[4:9, 4:9, 4:9] = True
    reference = mask_to_mesh(mask, geometry)
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
