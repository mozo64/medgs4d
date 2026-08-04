from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medgs4d.mesh_series import (
    build_rtstruct_mesh_series,
    downsample_ct_for_visualization,
    format_phase_roi_name,
    phase_directory_name,
)


@dataclass(frozen=True)
class FakeGeometry:
    shape_zyx: tuple[int, int, int]
    origin_xyz: tuple[float, float, float]
    column_direction_xyz: tuple[float, float, float]
    row_direction_xyz: tuple[float, float, float]
    slice_direction_xyz: tuple[float, float, float]
    spacing_zyx: tuple[float, float, float]
    slice_coordinates: tuple[float, ...]
    sop_instance_uids: tuple[str, ...]
    series_instance_uid: str
    series_path: str

    def index_zyx_to_patient_xyz(self, points_zyx: np.ndarray) -> np.ndarray:
        points = np.asarray(points_zyx, dtype=np.float64)
        z = points[..., 0, None]
        y = points[..., 1, None]
        x = points[..., 2, None]
        return (
            np.asarray(self.origin_xyz)
            + x * self.spacing_zyx[2] * np.asarray(self.column_direction_xyz)
            + y * self.spacing_zyx[1] * np.asarray(self.row_direction_xyz)
            + z * self.spacing_zyx[0] * np.asarray(self.slice_direction_xyz)
        )


def make_geometry() -> FakeGeometry:
    return FakeGeometry(
        shape_zyx=(6, 8, 10),
        origin_xyz=(10.0, 20.0, 30.0),
        column_direction_xyz=(1.0, 0.0, 0.0),
        row_direction_xyz=(0.0, 1.0, 0.0),
        slice_direction_xyz=(0.0, 0.0, 1.0),
        spacing_zyx=(2.0, 1.5, 1.0),
        slice_coordinates=tuple(30.0 + 2.0 * index for index in range(6)),
        sop_instance_uids=tuple(f"sop-{index}" for index in range(6)),
        series_instance_uid="series",
        series_path="/tmp/series",
    )


def test_phase_names():
    assert phase_directory_name(0) == "phase_00"
    assert phase_directory_name(90) == "phase_90"
    assert format_phase_roi_name("Tumor_c{phase:02d}", 10) == "Tumor_c10"


def test_downsample_ct_preserves_grid_coordinates():
    geometry = make_geometry()
    ct = np.arange(np.prod(geometry.shape_zyx), dtype=np.float32).reshape(
        geometry.shape_zyx
    )
    volume, reduced = downsample_ct_for_visualization(
        ct,
        geometry,
        stride_zyx=(2, 2, 5),
        hu_window=(-1000, 1000),
    )

    assert volume.shape == (3, 4, 2)
    assert reduced.shape_zyx == volume.shape
    assert reduced.spacing_zyx == (4.0, 3.0, 5.0)

    reduced_index = np.array([[1.0, 2.0, 1.0]])
    original_index = reduced_index * np.array([2.0, 2.0, 5.0])
    np.testing.assert_allclose(
        reduced.index_zyx_to_patient_xyz(reduced_index),
        geometry.index_zyx_to_patient_xyz(original_index),
    )


def test_series_writes_summary_and_manifest(tmp_path, monkeypatch):
    def fake_build_case(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        phase = float(kwargs["phase"])
        return {
            "manifest": {
                "ct_series_instance_uid": f"ct-{int(phase):02d}",
                "ct_visualization_shape_zyx": [3, 4, 5],
                "ct_visualization_stride_zyx": [2, 4, 4],
            },
            "report": {
                "shape_zyx": [6, 16, 20],
                "mask_voxels": 100,
                "mask_volume_mm3": 200.0,
                "mesh_vertices": 50,
                "mesh_faces": 96,
                "mesh_volume_mm3": 198.0,
                "roundtrip_dice": 1.0,
                "mesh_watertight": True,
            },
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr(
        "medgs4d.mesh_series.build_rtstruct_mesh_case",
        fake_build_case,
    )

    output_dir = tmp_path / "series"
    summary = build_rtstruct_mesh_series(
        dicom_dir=tmp_path,
        patient_id="patient",
        study_instance_uid="study",
        phases=[0, 10],
        roi_template="Tumor_c{phase:02d}",
        output_dir=output_dir,
    )

    assert list(summary["ROIName"]) == ["Tumor_c00", "Tumor_c10"]
    assert (output_dir / "series_summary.csv").exists()
    assert (output_dir / "series_manifest.json").exists()
