from __future__ import annotations

import numpy as np

from medgs4d.meshes import (
    binary_dice,
    mask_to_mesh,
    mesh_is_watertight,
    mesh_to_mask_roundtrip,
)
from medgs4d.rtstruct import RoiContour, VolumeGeometry, rasterize_roi


def geometry(shape=(24, 32, 40)) -> VolumeGeometry:
    return VolumeGeometry(
        shape_zyx=shape,
        origin_xyz=(10.0, -20.0, 30.0),
        column_direction_xyz=(1.0, 0.0, 0.0),
        row_direction_xyz=(0.0, 1.0, 0.0),
        slice_direction_xyz=(0.0, 0.0, 1.0),
        spacing_zyx=(2.5, 1.5, 1.0),
        slice_coordinates=tuple(30.0 + 2.5 * index for index in range(shape[0])),
        sop_instance_uids=tuple(f"slice-{index}" for index in range(shape[0])),
        series_instance_uid="series",
        series_path="/synthetic",
    )


def test_coordinate_transform_roundtrip():
    geo = geometry()
    points_zyx = np.asarray([[0.0, 0.0, 0.0], [3.25, 8.5, 12.75]])
    points_xyz = geo.index_zyx_to_patient_xyz(points_zyx)
    reconstructed = geo.patient_xyz_to_index_zyx(points_xyz)
    assert np.allclose(reconstructed, points_zyx)


def test_rasterize_square_contour():
    geo = geometry()
    points_zyx = np.asarray(
        [
            [5.0, 8.0, 10.0],
            [5.0, 8.0, 20.0],
            [5.0, 18.0, 20.0],
            [5.0, 18.0, 10.0],
        ]
    )
    contour = RoiContour(
        points_xyz=geo.index_zyx_to_patient_xyz(points_zyx),
        referenced_sop_instance_uid="slice-5",
        geometric_type="CLOSED_PLANAR",
    )
    result = rasterize_roi([contour], geo)
    assert result.mask[5].sum() > 0
    assert result.mask[:5].sum() == 0
    assert result.mask[6:].sum() == 0
    assert result.contour_table.iloc[0]["SliceIndex"] == 5


def test_mask_mesh_roundtrip():
    geo = geometry(shape=(28, 36, 44))
    z, y, x = np.indices(geo.shape_zyx)
    mask = (
        ((z - 14.0) / 8.0) ** 2
        + ((y - 18.0) / 11.0) ** 2
        + ((x - 22.0) / 14.0) ** 2
        <= 1.0
    )
    mesh = mask_to_mesh(mask, geo)
    reconstructed = mesh_to_mask_roundtrip(mesh.vertices_zyx, mesh.faces, mask.shape)
    assert mesh_is_watertight(mesh.faces)
    assert np.isfinite(mesh.vertices_xyz).all()
    assert binary_dice(mask, reconstructed) > 0.97
