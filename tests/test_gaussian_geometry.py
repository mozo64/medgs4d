from types import SimpleNamespace

import numpy as np

from medgs4d.gaussian_geometry import (
    GaussianDicomTransform,
    camera_time_boundaries,
    medgs_m_to_slice_index,
    pixel_to_ndc,
    project_medgs_xz_to_pixel,
    slice_index_to_medgs_m,
    unproject_pixel_to_medgs_xz,
)
from medgs4d.rtstruct import VolumeGeometry


def projection_matrix() -> np.ndarray:
    fov = 0.6911112070083618
    distance = -1.0
    sign = np.sign(distance)
    c2w = np.array(
        [
            [-sign, 0.0, 0.0, 0.0],
            [0.0, 0.0, sign, distance],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    c2w[:3, 1:3] *= -1.0
    world_view = np.linalg.inv(c2w).T

    near = 0.01
    far = 100.0
    tangent = np.tan(fov / 2.0)
    projection = np.zeros((4, 4))
    projection[0, 0] = 1.0 / tangent
    projection[1, 1] = 1.0 / tangent
    projection[3, 2] = 1.0
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -(far * near) / (far - near)
    return world_view @ projection.T


def test_projection_roundtrip() -> None:
    matrix = projection_matrix()
    points = np.array([[0.0, 0.0], [0.1, 0.05], [-0.07, -0.02]])
    pixels = project_medgs_xz_to_pixel(points, matrix, 512, 512)
    restored = unproject_pixel_to_medgs_xz(pixels, matrix, 512, 512)
    assert np.allclose(restored, points, atol=1e-10)


def test_ndc_center_is_image_center() -> None:
    center = pixel_to_ndc(np.array([[255.5, 255.5]]), 512, 512)
    assert np.allclose(center, [[0.0, 0.0]])


def test_time_mapping_roundtrip() -> None:
    steps = np.array([0.1, 0.2, 0.3, 0.4])
    boundaries = camera_time_boundaries(steps)
    assert np.allclose(boundaries, [0.0, 0.1, 0.3, 0.6, 1.0])

    indices = np.array([0.0, 0.5, 1.0, 2.5, 4.0])
    m = slice_index_to_medgs_m(indices, steps)
    restored = medgs_m_to_slice_index(m, steps)
    assert np.allclose(restored, indices)


def test_patient_latent_roundtrip() -> None:
    geometry = VolumeGeometry(
        shape_zyx=(147, 512, 512),
        origin_xyz=(-259.2, -237.619, -266.2),
        column_direction_xyz=(1.0, 0.0, 0.0),
        row_direction_xyz=(0.0, 1.0, 0.0),
        slice_direction_xyz=(0.0, 0.0, 1.0),
        spacing_zyx=(3.0, 0.9766, 0.9766),
        slice_coordinates=tuple(-266.2 + 3.0 * i for i in range(147)),
        sop_instance_uids=tuple(str(i) for i in range(147)),
        series_instance_uid="series",
        series_path="/tmp/series",
    )
    transform = GaussianDicomTransform(
        geometry=geometry,
        full_projection=projection_matrix(),
        image_width=512,
        image_height=512,
        time_steps=np.ones(147) / 147.0,
    )
    points_xyz = np.array(
        [
            [-50.0, 10.0, 0.0],
            [-30.0, 25.0, 12.0],
            [0.0, 0.0, -100.0],
        ]
    )
    restored = transform.patient_roundtrip(points_xyz)
    assert np.allclose(restored, points_xyz, atol=1e-8)
