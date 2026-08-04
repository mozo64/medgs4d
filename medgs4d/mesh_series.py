from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence
import json
import shutil

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .rtstruct import VolumeGeometry


def phase_directory_name(phase: float) -> str:
    """Return a stable directory name such as phase_00 or phase_10."""

    rounded = int(round(float(phase)))
    if not np.isclose(float(phase), rounded):
        raise ValueError(f"Expected an integer respiratory phase, got {phase}")
    return f"phase_{rounded:02d}"


def format_phase_roi_name(template: str, phase: float) -> str:
    """Format a phase-specific ROI name, for example Tumor_c00."""

    rounded = int(round(float(phase)))
    if not np.isclose(float(phase), rounded):
        raise ValueError(f"Expected an integer respiratory phase, got {phase}")
    return template.format(phase=rounded)


def downsample_ct_for_visualization(
    ct_volume: np.ndarray,
    geometry: VolumeGeometry,
    stride_zyx: Sequence[int] = (2, 4, 4),
    hu_window: Sequence[float] = (-1000.0, 400.0),
) -> tuple[np.ndarray, VolumeGeometry]:
    """Prepare a compact HU volume and matching geometry for notebook rendering."""

    stride = tuple(int(value) for value in stride_zyx)
    if len(stride) != 3 or any(value <= 0 for value in stride):
        raise ValueError(f"stride_zyx must contain three positive integers, got {stride}")

    low, high = (float(value) for value in hu_window)
    if low >= high:
        raise ValueError(f"Invalid HU window: {(low, high)}")

    volume = np.asarray(ct_volume)
    if tuple(volume.shape) != tuple(geometry.shape_zyx):
        raise ValueError(
            f"CT shape {volume.shape} does not match geometry {geometry.shape_zyx}"
        )

    clipped = np.clip(volume, low, high)
    downsampled = np.rint(
        clipped[:: stride[0], :: stride[1], :: stride[2]]
    ).astype(np.int16)

    spacing = tuple(
        float(original) * factor
        for original, factor in zip(geometry.spacing_zyx, stride)
    )

    downsampled_geometry = replace(
        geometry,
        shape_zyx=tuple(int(value) for value in downsampled.shape),
        spacing_zyx=spacing,
        slice_coordinates=tuple(geometry.slice_coordinates[:: stride[0]]),
        sop_instance_uids=tuple(geometry.sop_instance_uids[:: stride[0]]),
    )
    return downsampled, downsampled_geometry


def build_rtstruct_mesh_case(
    *,
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
    phase: float,
    roi_name: str,
    output_dir: Path,
    rtstruct_file: Path | None = None,
    hu_window: Sequence[float] = (-1000.0, 400.0),
    roundtrip_dice_min: float = 0.95,
    ct_vis_stride_zyx: Sequence[int] = (2, 4, 4),
    force: bool = False,
) -> dict:
    """Build one phase mask, mesh, validation report, and compact CT volume."""

    from .mesh_validation import save_validation_overview
    from .meshes import (
        mask_to_mesh,
        mesh_to_mask_roundtrip,
        mesh_validation_report,
        save_mesh_npz,
        save_report,
        write_ply,
    )
    from .rtstruct import (
        contours_to_json,
        find_rtstruct_and_ct_series,
        load_ct_geometry,
        load_ct_volume,
        rasterize_roi,
        read_roi_contours,
        save_geometry,
    )

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\nUse --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rtstruct_path, ct_series = find_rtstruct_and_ct_series(
        Path(dicom_dir),
        patient_id,
        study_instance_uid,
        float(phase),
        roi_name=roi_name,
        rtstruct_file=rtstruct_file,
    )
    geometry = load_ct_geometry(
        Path(ct_series["SeriesPath"]),
        str(ct_series["SeriesInstanceUID"]),
    )
    _, contours = read_roi_contours(rtstruct_path, roi_name)
    rasterized = rasterize_roi(contours, geometry)
    mesh = mask_to_mesh(rasterized.mask, geometry)
    roundtrip_mask = mesh_to_mask_roundtrip(
        mesh.vertices_zyx,
        mesh.faces,
        geometry.shape_zyx,
    )
    report = mesh_validation_report(
        rasterized.mask,
        roundtrip_mask,
        mesh,
        geometry,
    )
    report.update(
        {
            "patient_id": patient_id,
            "study_instance_uid": study_instance_uid,
            "phase_percent": float(phase),
            "roi_name": roi_name,
            "rtstruct_file": str(rtstruct_path),
            "ct_series_instance_uid": str(ct_series["SeriesInstanceUID"]),
            "ct_series_path": str(ct_series["SeriesPath"]),
            "contour_count": len(contours),
            "roundtrip_dice_minimum": float(roundtrip_dice_min),
        }
    )

    np.save(output_dir / "mask.npy", rasterized.mask)
    np.save(output_dir / "roundtrip_mask.npy", roundtrip_mask)
    save_geometry(output_dir / "geometry.json", geometry)
    rasterized.contour_table.to_csv(output_dir / "contour_report.csv", index=False)
    (output_dir / "contours.json").write_text(
        json.dumps(contours_to_json(contours, geometry), indent=2),
        encoding="utf-8",
    )
    save_mesh_npz(output_dir / "mesh_raw.npz", mesh)
    write_ply(output_dir / "mesh_raw.ply", mesh.vertices_xyz, mesh.faces)
    save_report(output_dir / "validation_report.json", report)

    ct_volume = load_ct_volume(geometry)
    ct_vis, ct_vis_geometry = downsample_ct_for_visualization(
        ct_volume,
        geometry,
        stride_zyx=ct_vis_stride_zyx,
        hu_window=hu_window,
    )
    np.save(output_dir / "ct_vis_hu.npy", ct_vis)
    save_geometry(output_dir / "ct_vis_geometry.json", ct_vis_geometry)

    save_validation_overview(
        output_dir / "validation_overview.png",
        ct_volume,
        rasterized.mask,
        roundtrip_mask,
        contours,
        geometry,
        tuple(float(value) for value in hu_window),
    )

    manifest = {
        "patient_id": patient_id,
        "study_instance_uid": study_instance_uid,
        "phase_percent": float(phase),
        "roi_name": roi_name,
        "rtstruct_file": str(rtstruct_path),
        "ct_series_instance_uid": str(ct_series["SeriesInstanceUID"]),
        "geometry_file": "geometry.json",
        "mask_file": "mask.npy",
        "roundtrip_mask_file": "roundtrip_mask.npy",
        "contours_file": "contours.json",
        "contour_report_file": "contour_report.csv",
        "mesh_npz_file": "mesh_raw.npz",
        "mesh_ply_file": "mesh_raw.ply",
        "validation_report_file": "validation_report.json",
        "validation_overview_file": "validation_overview.png",
        "ct_visualization_file": "ct_vis_hu.npy",
        "ct_visualization_geometry_file": "ct_vis_geometry.json",
        "ct_visualization_shape_zyx": list(ct_vis.shape),
        "ct_visualization_stride_zyx": [int(value) for value in ct_vis_stride_zyx],
        "ct_visualization_hu_window": [float(value) for value in hu_window],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not report["finite_vertices"]:
        raise AssertionError("Mesh contains non-finite vertices")
    if not report["mesh_watertight"]:
        raise AssertionError("Generated mesh is not watertight")
    if report["roundtrip_dice"] < roundtrip_dice_min:
        raise AssertionError(
            f"Round-trip Dice {report['roundtrip_dice']:.6f} is below "
            f"{roundtrip_dice_min:.6f}"
        )

    return {
        "manifest": manifest,
        "report": report,
        "output_dir": str(output_dir),
    }


def build_rtstruct_mesh_series(
    *,
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
    phases: Iterable[float],
    roi_template: str,
    output_dir: Path,
    hu_window: Sequence[float] = (-1000.0, 400.0),
    roundtrip_dice_min: float = 0.95,
    ct_vis_stride_zyx: Sequence[int] = (2, 4, 4),
    force: bool = False,
) -> pd.DataFrame:
    """Build reference masks, meshes, and visualization CT for all requested phases."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\nUse --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    phase_values = sorted({float(value) for value in phases})
    if not phase_values:
        raise ValueError("No respiratory phases were provided")

    rows = []
    phase_manifests = []
    for phase in phase_values:
        roi_name = format_phase_roi_name(roi_template, phase)
        phase_dir = output_dir / phase_directory_name(phase)
        result = build_rtstruct_mesh_case(
            dicom_dir=Path(dicom_dir),
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            phase=phase,
            roi_name=roi_name,
            output_dir=phase_dir,
            hu_window=hu_window,
            roundtrip_dice_min=roundtrip_dice_min,
            ct_vis_stride_zyx=ct_vis_stride_zyx,
        )
        manifest = result["manifest"]
        report = result["report"]
        relative_phase_dir = phase_dir.relative_to(output_dir)
        phase_manifests.append(
            {
                "phase_percent": phase,
                "roi_name": roi_name,
                "directory": str(relative_phase_dir),
                "manifest_file": str(relative_phase_dir / "manifest.json"),
            }
        )
        rows.append(
            {
                "PhasePercent": phase,
                "ROIName": roi_name,
                "PhaseDirectory": str(relative_phase_dir),
                "CTSeriesInstanceUID": manifest["ct_series_instance_uid"],
                "CTShapeZYX": "x".join(str(value) for value in report["shape_zyx"]),
                "CTVisualizationShapeZYX": "x".join(
                    str(value) for value in manifest["ct_visualization_shape_zyx"]
                ),
                "CTVisualizationStrideZYX": "x".join(
                    str(value) for value in manifest["ct_visualization_stride_zyx"]
                ),
                "MaskVoxels": report["mask_voxels"],
                "MaskVolumeMm3": report["mask_volume_mm3"],
                "MeshVertices": report["mesh_vertices"],
                "MeshFaces": report["mesh_faces"],
                "MeshVolumeMm3": report["mesh_volume_mm3"],
                "RoundtripDice": report["roundtrip_dice"],
                "MeshWatertight": report["mesh_watertight"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("PhasePercent").reset_index(drop=True)
    summary.to_csv(output_dir / "series_summary.csv", index=False)

    series_manifest = {
        "patient_id": patient_id,
        "study_instance_uid": study_instance_uid,
        "roi_template": roi_template,
        "phases": phase_values,
        "ct_visualization_stride_zyx": [int(value) for value in ct_vis_stride_zyx],
        "ct_visualization_hu_window": [float(value) for value in hu_window],
        "series_summary_file": "series_summary.csv",
        "phase_cases": phase_manifests,
    }
    (output_dir / "series_manifest.json").write_text(
        json.dumps(series_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
