from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math

import numpy as np
import pandas as pd

from .data import (
    list_dicom_files,
    parse_phase_percent,
    read_dicom_header,
    read_hu_image,
    scan_patient_series,
)


@dataclass(frozen=True)
class VolumeGeometry:
    """Describe one regular CT volume in DICOM patient coordinates."""

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
        """Map continuous array indices [z, y, x] to patient coordinates [x, y, z]."""

        points = np.asarray(points_zyx, dtype=np.float64)
        z = points[..., 0, None]
        y = points[..., 1, None]
        x = points[..., 2, None]
        origin = np.asarray(self.origin_xyz, dtype=np.float64)
        column_direction = np.asarray(self.column_direction_xyz, dtype=np.float64)
        row_direction = np.asarray(self.row_direction_xyz, dtype=np.float64)
        slice_direction = np.asarray(self.slice_direction_xyz, dtype=np.float64)
        spacing_z, spacing_y, spacing_x = self.spacing_zyx
        return (
            origin
            + x * spacing_x * column_direction
            + y * spacing_y * row_direction
            + z * spacing_z * slice_direction
        )

    def patient_xyz_to_index_zyx(self, points_xyz: np.ndarray) -> np.ndarray:
        """Map patient coordinates [x, y, z] to continuous array indices [z, y, x]."""

        points = np.asarray(points_xyz, dtype=np.float64)
        delta = points - np.asarray(self.origin_xyz, dtype=np.float64)
        column_direction = np.asarray(self.column_direction_xyz, dtype=np.float64)
        row_direction = np.asarray(self.row_direction_xyz, dtype=np.float64)
        slice_direction = np.asarray(self.slice_direction_xyz, dtype=np.float64)
        spacing_z, spacing_y, spacing_x = self.spacing_zyx
        x = np.einsum("...i,i->...", delta, column_direction) / spacing_x
        y = np.einsum("...i,i->...", delta, row_direction) / spacing_y
        z = np.einsum("...i,i->...", delta, slice_direction) / spacing_z
        return np.stack([z, y, x], axis=-1)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "VolumeGeometry":
        converted = dict(data)
        for key in (
            "shape_zyx",
            "origin_xyz",
            "column_direction_xyz",
            "row_direction_xyz",
            "slice_direction_xyz",
            "spacing_zyx",
            "slice_coordinates",
            "sop_instance_uids",
        ):
            converted[key] = tuple(converted[key])
        return cls(**converted)


@dataclass(frozen=True)
class RoiContour:
    """One planar contour belonging to a named RTSTRUCT ROI."""

    points_xyz: np.ndarray
    referenced_sop_instance_uid: str
    geometric_type: str


@dataclass(frozen=True)
class RasterizedRoi:
    """Rasterized ROI together with contour-to-slice diagnostics."""

    mask: np.ndarray
    contours: tuple[RoiContour, ...]
    contour_table: pd.DataFrame


def _normal_from_orientation(orientation: Sequence[float]) -> np.ndarray:
    values = np.asarray([float(value) for value in orientation], dtype=np.float64)
    column_direction = values[:3]
    row_direction = values[3:]
    normal = np.cross(column_direction, row_direction)
    return normal / np.linalg.norm(normal)


def load_ct_geometry(series_dir: Path, expected_series_uid: str | None = None) -> VolumeGeometry:
    """Read ordered CT headers and construct a regular patient-coordinate geometry."""

    rows = []
    for path in list_dicom_files(series_dir):
        dataset = read_dicom_header(path)
        if str(getattr(dataset, "Modality", "")).upper() != "CT":
            continue
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if expected_series_uid is not None and series_uid != expected_series_uid:
            raise ValueError(f"Unexpected SeriesInstanceUID in {path}")
        position = np.asarray(
            [float(value) for value in dataset.ImagePositionPatient], dtype=np.float64
        )
        orientation = np.asarray(
            [float(value) for value in dataset.ImageOrientationPatient], dtype=np.float64
        )
        normal = _normal_from_orientation(orientation)
        rows.append(
            {
                "Path": str(path),
                "SOPInstanceUID": str(getattr(dataset, "SOPInstanceUID", "")),
                "SeriesInstanceUID": series_uid,
                "Position": position,
                "Orientation": orientation,
                "SliceCoordinate": float(np.dot(position, normal)),
                "Rows": int(dataset.Rows),
                "Columns": int(dataset.Columns),
                "PixelSpacing": tuple(float(value) for value in dataset.PixelSpacing),
            }
        )

    if not rows:
        raise ValueError(f"No CT slices found in {series_dir}")

    rows.sort(key=lambda row: row["SliceCoordinate"])
    first = rows[0]
    coordinates = np.asarray([row["SliceCoordinate"] for row in rows], dtype=np.float64)
    if len(coordinates) < 2:
        raise ValueError("At least two CT slices are required")
    spacings = np.diff(coordinates)
    spacing_z = float(np.median(spacings))
    if not np.allclose(spacings, spacing_z, atol=1e-3):
        raise ValueError("CT slice spacing is not regular within 1e-3 mm")

    orientation = first["Orientation"]
    column_direction = orientation[:3]
    row_direction = orientation[3:]
    normal = _normal_from_orientation(orientation)
    spacing_y, spacing_x = first["PixelSpacing"]

    for row in rows:
        if (row["Rows"], row["Columns"]) != (first["Rows"], first["Columns"]):
            raise ValueError("CT slice dimensions differ within the series")
        if not np.allclose(row["Orientation"], orientation, atol=1e-6):
            raise ValueError("CT slice orientation differs within the series")
        if not np.allclose(row["PixelSpacing"], first["PixelSpacing"], atol=1e-6):
            raise ValueError("CT pixel spacing differs within the series")

    return VolumeGeometry(
        shape_zyx=(len(rows), first["Rows"], first["Columns"]),
        origin_xyz=tuple(float(value) for value in first["Position"]),
        column_direction_xyz=tuple(float(value) for value in column_direction),
        row_direction_xyz=tuple(float(value) for value in row_direction),
        slice_direction_xyz=tuple(float(value) for value in normal),
        spacing_zyx=(spacing_z, spacing_y, spacing_x),
        slice_coordinates=tuple(float(value) for value in coordinates),
        sop_instance_uids=tuple(row["SOPInstanceUID"] for row in rows),
        series_instance_uid=first["SeriesInstanceUID"],
        series_path=str(series_dir),
    )


def load_ct_volume(geometry: VolumeGeometry) -> np.ndarray:
    """Load the CT series represented by a VolumeGeometry in HU."""

    series_dir = Path(geometry.series_path)
    by_sop = {}
    for path in list_dicom_files(series_dir):
        dataset = read_dicom_header(path)
        if str(getattr(dataset, "Modality", "")).upper() == "CT":
            by_sop[str(getattr(dataset, "SOPInstanceUID", ""))] = path
    return np.stack(
        [read_hu_image(by_sop[sop_uid]) for sop_uid in geometry.sop_instance_uids],
        axis=0,
    ).astype(np.float32)


def _roi_number_by_name(dataset: Any, roi_name: str) -> int:
    matches = [
        int(roi.ROINumber)
        for roi in getattr(dataset, "StructureSetROISequence", ()) or ()
        if str(getattr(roi, "ROIName", "")) == roi_name
    ]
    if len(matches) != 1:
        available = [
            str(getattr(roi, "ROIName", ""))
            for roi in getattr(dataset, "StructureSetROISequence", ()) or ()
        ]
        raise ValueError(
            f"ROI {roi_name!r} matched {len(matches)} entries. Available ROI names: {available}"
        )
    return matches[0]


def read_roi_contours(rtstruct_path: Path, roi_name: str) -> tuple[Any, tuple[RoiContour, ...]]:
    """Read one named ROI from a DICOM RTSTRUCT file."""

    import pydicom

    dataset = pydicom.dcmread(str(rtstruct_path), force=True)
    if str(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT":
        raise ValueError(f"Not an RTSTRUCT object: {rtstruct_path}")
    roi_number = _roi_number_by_name(dataset, roi_name)

    contours = []
    for roi_contour in getattr(dataset, "ROIContourSequence", ()) or ():
        if int(getattr(roi_contour, "ReferencedROINumber", -1)) != roi_number:
            continue
        for contour in getattr(roi_contour, "ContourSequence", ()) or ():
            values = np.asarray(contour.ContourData, dtype=np.float64)
            points_xyz = values.reshape(-1, 3)
            image_sequence = getattr(contour, "ContourImageSequence", ()) or ()
            referenced_sop_uid = (
                str(getattr(image_sequence[0], "ReferencedSOPInstanceUID", ""))
                if image_sequence
                else ""
            )
            contours.append(
                RoiContour(
                    points_xyz=points_xyz,
                    referenced_sop_instance_uid=referenced_sop_uid,
                    geometric_type=str(getattr(contour, "ContourGeometricType", "")),
                )
            )
    if not contours:
        raise ValueError(f"ROI {roi_name!r} contains no contours in {rtstruct_path}")
    return dataset, tuple(contours)


def rtstruct_roi_names(rtstruct_path: Path) -> list[str]:
    """List ROI names stored in one RTSTRUCT object."""

    import pydicom

    dataset = pydicom.dcmread(str(rtstruct_path), stop_before_pixels=True, force=True)
    return [
        str(getattr(roi, "ROIName", ""))
        for roi in getattr(dataset, "StructureSetROISequence", ()) or ()
    ]


ROI_INSPECTION_COLUMNS = [
    "PatientID",
    "StudyInstanceUID",
    "PhasePercent",
    "StructureSetLabel",
    "ROIName",
    "ROINumber",
    "ContourCount",
    "GeometricTypes",
    "PointCount",
    "DistinctContourSlices",
    "ContoursPerSliceMax",
    "ZMinMm",
    "ZMaxMm",
    "ZExtentMm",
    "ClosedPlanar",
    "VolumeCandidate",
    "RTSTRUCTFile",
]


def summarize_rtstruct_rois(
    dataset: Any,
    rtstruct_path: Path | str = "",
    phase_percent: float | None = None,
) -> pd.DataFrame:
    """Summarize contour geometry for every ROI in one RTSTRUCT dataset."""

    if phase_percent is None:
        phase_percent = parse_phase_percent(
            str(getattr(dataset, "SeriesDescription", ""))
        )

    contours_by_roi: dict[int, list[Any]] = {}
    for roi_contour in getattr(dataset, "ROIContourSequence", ()) or ():
        roi_number = int(getattr(roi_contour, "ReferencedROINumber", -1))
        contours_by_roi.setdefault(roi_number, []).extend(
            list(getattr(roi_contour, "ContourSequence", ()) or ())
        )

    rows = []
    for roi in getattr(dataset, "StructureSetROISequence", ()) or ():
        roi_number = int(roi.ROINumber)
        contours = contours_by_roi.get(roi_number, [])
        geometric_types = sorted(
            {
                str(getattr(contour, "ContourGeometricType", ""))
                for contour in contours
            }
        )

        point_count = 0
        contour_z_values = []
        valid_polygon_points = True
        for contour in contours:
            values = np.asarray(
                getattr(contour, "ContourData", ()) or (),
                dtype=np.float64,
            )
            points = values.reshape(-1, 3)
            point_count += len(points)
            valid_polygon_points &= len(points) >= 3
            if len(points):
                contour_z_values.append(float(points[:, 2].mean()))

        rounded_z = [round(value, 3) for value in contour_z_values]
        distinct_z = sorted(set(rounded_z))
        counts_by_z = {
            value: rounded_z.count(value)
            for value in distinct_z
        }
        closed_planar = bool(contours) and geometric_types == ["CLOSED_PLANAR"]
        volume_candidate = (
            closed_planar
            and valid_polygon_points
            and len(distinct_z) >= 2
        )

        z_min = min(contour_z_values) if contour_z_values else math.nan
        z_max = max(contour_z_values) if contour_z_values else math.nan
        rows.append(
            {
                "PatientID": str(getattr(dataset, "PatientID", "")),
                "StudyInstanceUID": str(
                    getattr(dataset, "StudyInstanceUID", "")
                ),
                "PhasePercent": phase_percent,
                "StructureSetLabel": str(
                    getattr(dataset, "StructureSetLabel", "")
                ),
                "ROIName": str(getattr(roi, "ROIName", "")),
                "ROINumber": roi_number,
                "ContourCount": len(contours),
                "GeometricTypes": " | ".join(geometric_types),
                "PointCount": point_count,
                "DistinctContourSlices": len(distinct_z),
                "ContoursPerSliceMax": max(counts_by_z.values(), default=0),
                "ZMinMm": z_min,
                "ZMaxMm": z_max,
                "ZExtentMm": z_max - z_min if contour_z_values else math.nan,
                "ClosedPlanar": closed_planar,
                "VolumeCandidate": volume_candidate,
                "RTSTRUCTFile": str(rtstruct_path),
            }
        )

    return pd.DataFrame(rows, columns=ROI_INSPECTION_COLUMNS)


def inspect_rtstruct_rois(
    rtstruct_path: Path,
    phase_percent: float | None = None,
) -> pd.DataFrame:
    """Read one RTSTRUCT file and summarize all ROI contour geometries."""

    import pydicom

    dataset = pydicom.dcmread(
        str(rtstruct_path),
        stop_before_pixels=True,
        force=True,
    )
    if str(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT":
        raise ValueError(f"Not an RTSTRUCT object: {rtstruct_path}")
    return summarize_rtstruct_rois(dataset, rtstruct_path, phase_percent)


def find_rtstruct_file(
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
    phase: float,
    roi_name: str | None = None,
    rtstruct_file: Path | None = None,
) -> Path:
    """Resolve exactly one local RTSTRUCT object for one respiratory phase."""

    if rtstruct_file is not None:
        return Path(rtstruct_file)

    inventory = scan_patient_series(dicom_dir, patient_id)
    candidates = inventory.loc[
        (inventory["StudyInstanceUID"].astype(str) == str(study_instance_uid))
        & (inventory["Modality"].astype(str).str.upper() == "RTSTRUCT")
        & np.isclose(
            inventory["PhasePercent"].astype(float),
            float(phase),
            equal_nan=False,
        )
    ]

    candidate_files = []
    for row in candidates.itertuples():
        for path in list_dicom_files(Path(row.SeriesPath)):
            dataset = read_dicom_header(path)
            if str(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT":
                continue
            if roi_name is None or roi_name in rtstruct_roi_names(path):
                candidate_files.append(path)

    candidate_files = sorted(set(candidate_files))
    if len(candidate_files) != 1:
        raise ValueError(
            f"Expected exactly one RTSTRUCT file for phase {phase:g}% and ROI "
            f"{roi_name!r}; found {len(candidate_files)}: {candidate_files}"
        )
    return candidate_files[0]


def inspect_phase_rois(
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
    phase: float,
    rtstruct_file: Path | None = None,
) -> pd.DataFrame:
    """Summarize all ROI geometries for one study phase."""

    path = find_rtstruct_file(
        dicom_dir,
        patient_id,
        study_instance_uid,
        phase,
        rtstruct_file=rtstruct_file,
    )
    return inspect_rtstruct_rois(path, phase_percent=phase)


def _referenced_series_uids(dataset: Any) -> set[str]:
    referenced = set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", ()) or ():
        for study in getattr(frame, "RTReferencedStudySequence", ()) or ():
            for series in getattr(study, "RTReferencedSeriesSequence", ()) or ():
                uid = str(getattr(series, "SeriesInstanceUID", ""))
                if uid:
                    referenced.add(uid)
    return referenced


def _build_sop_to_series_index(series: pd.DataFrame) -> dict[str, str]:
    index = {}
    ct = series.loc[series["Modality"].astype(str).str.upper() == "CT"]
    for row in ct.itertuples():
        for path in list_dicom_files(Path(row.SeriesPath)):
            dataset = read_dicom_header(path)
            sop_uid = str(getattr(dataset, "SOPInstanceUID", ""))
            if sop_uid:
                index[sop_uid] = str(row.SeriesInstanceUID)
    return index


def find_rtstruct_and_ct_series(
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
    phase: float,
    roi_name: str | None = None,
    rtstruct_file: Path | None = None,
) -> tuple[Path, pd.Series]:
    """Resolve one RTSTRUCT object and its referenced local CT series."""

    inventory = scan_patient_series(dicom_dir, patient_id)
    rtstruct_file = find_rtstruct_file(
        dicom_dir,
        patient_id,
        study_instance_uid,
        phase,
        roi_name=roi_name,
        rtstruct_file=rtstruct_file,
    )

    import pydicom

    dataset = pydicom.dcmread(str(rtstruct_file), stop_before_pixels=True, force=True)
    referenced_series = _referenced_series_uids(dataset)

    if not referenced_series:
        _, contours = read_roi_contours(rtstruct_file, roi_name) if roi_name else (dataset, ())
        contour_sops = {
            contour.referenced_sop_instance_uid
            for contour in contours
            if contour.referenced_sop_instance_uid
        }
        sop_to_series = _build_sop_to_series_index(inventory)
        referenced_series = {sop_to_series[uid] for uid in contour_sops if uid in sop_to_series}

    ct_candidates = inventory.loc[
        inventory["SeriesInstanceUID"].astype(str).isin(referenced_series)
        & (inventory["Modality"].astype(str).str.upper() == "CT")
    ]
    if len(ct_candidates) != 1:
        raise ValueError(
            f"Expected exactly one referenced local CT series; found {len(ct_candidates)} "
            f"for UIDs {sorted(referenced_series)}"
        )
    return Path(rtstruct_file), ct_candidates.iloc[0]


def rasterize_roi(
    contours: Sequence[RoiContour],
    geometry: VolumeGeometry,
) -> RasterizedRoi:
    """Rasterize planar RTSTRUCT contours on the exact referenced CT grid."""

    from skimage.draw import polygon

    mask = np.zeros(geometry.shape_zyx, dtype=bool)
    sop_to_slice = {
        sop_uid: index for index, sop_uid in enumerate(geometry.sop_instance_uids)
    }
    normal = np.asarray(geometry.slice_direction_xyz, dtype=np.float64)
    slice_coordinates = np.asarray(geometry.slice_coordinates, dtype=np.float64)
    rows = []

    for contour_index, contour in enumerate(contours):
        points_zyx = geometry.patient_xyz_to_index_zyx(contour.points_xyz)
        if contour.referenced_sop_instance_uid in sop_to_slice:
            slice_index = sop_to_slice[contour.referenced_sop_instance_uid]
        else:
            contour_coordinate = float(np.mean(contour.points_xyz @ normal))
            slice_index = int(np.argmin(np.abs(slice_coordinates - contour_coordinate)))

        plane_distances = np.abs(
            contour.points_xyz @ normal - slice_coordinates[slice_index]
        )
        max_plane_distance = float(plane_distances.max())
        if max_plane_distance > 1e-2:
            raise ValueError(
                f"Contour {contour_index} is {max_plane_distance:.6f} mm away from "
                f"the resolved CT slice plane"
            )

        row_coordinates = points_zyx[:, 1]
        column_coordinates = points_zyx[:, 2]
        rr, cc = polygon(
            row_coordinates,
            column_coordinates,
            shape=geometry.shape_zyx[1:],
        )
        # XOR implements the even-odd fill rule and therefore preserves nested holes.
        mask[slice_index, rr, cc] ^= True

        rows.append(
            {
                "ContourIndex": contour_index,
                "SliceIndex": slice_index,
                "ReferencedSOPInstanceUID": contour.referenced_sop_instance_uid,
                "GeometricType": contour.geometric_type,
                "PointCount": len(contour.points_xyz),
                "MaxPlaneDistanceMm": max_plane_distance,
                "MinRow": float(row_coordinates.min()),
                "MaxRow": float(row_coordinates.max()),
                "MinColumn": float(column_coordinates.min()),
                "MaxColumn": float(column_coordinates.max()),
            }
        )

    if not mask.any():
        raise ValueError("Rasterized ROI mask is empty")
    return RasterizedRoi(
        mask=mask,
        contours=tuple(contours),
        contour_table=pd.DataFrame(rows).sort_values(
            ["SliceIndex", "ContourIndex"]
        ).reset_index(drop=True),
    )


def contours_to_json(
    contours: Sequence[RoiContour],
    geometry: VolumeGeometry,
) -> list[dict[str, Any]]:
    """Serialize patient-space and voxel-space contour coordinates for validation."""

    sop_to_slice = {
        sop_uid: index for index, sop_uid in enumerate(geometry.sop_instance_uids)
    }
    normal = np.asarray(geometry.slice_direction_xyz, dtype=np.float64)
    slice_coordinates = np.asarray(geometry.slice_coordinates, dtype=np.float64)
    result = []
    for contour in contours:
        points_zyx = geometry.patient_xyz_to_index_zyx(contour.points_xyz)
        if contour.referenced_sop_instance_uid in sop_to_slice:
            slice_index = sop_to_slice[contour.referenced_sop_instance_uid]
        else:
            coordinate = float(np.mean(contour.points_xyz @ normal))
            slice_index = int(np.argmin(np.abs(slice_coordinates - coordinate)))
        result.append(
            {
                "slice_index": slice_index,
                "referenced_sop_instance_uid": contour.referenced_sop_instance_uid,
                "geometric_type": contour.geometric_type,
                "points_xyz": contour.points_xyz.tolist(),
                "points_zyx": points_zyx.tolist(),
            }
        )
    return result


def inventory_rtstruct_objects(
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
) -> pd.DataFrame:
    """Return one row per RTSTRUCT DICOM object with phase and ROI names."""

    series = scan_patient_series(dicom_dir, patient_id)
    selected = series.loc[
        (series["StudyInstanceUID"].astype(str) == str(study_instance_uid))
        & (series["Modality"].astype(str).str.upper() == "RTSTRUCT")
    ]
    rows = []
    for row in selected.itertuples():
        for path in list_dicom_files(Path(row.SeriesPath)):
            dataset = read_dicom_header(path)
            if str(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT":
                continue
            names = rtstruct_roi_names(path)
            rows.append(
                {
                    "PatientID": patient_id,
                    "StudyInstanceUID": study_instance_uid,
                    "PhasePercent": parse_phase_percent(
                        str(getattr(dataset, "SeriesDescription", row.SeriesDescription))
                    ),
                    "RTSeriesInstanceUID": str(
                        getattr(dataset, "SeriesInstanceUID", row.SeriesInstanceUID)
                    ),
                    "SOPInstanceUID": str(getattr(dataset, "SOPInstanceUID", "")),
                    "StructureSetLabel": str(
                        getattr(dataset, "StructureSetLabel", "")
                    ),
                    "ROICount": len(names),
                    "ROINames": " | ".join(names),
                    "RTSTRUCTFile": str(path),
                }
            )
    columns = [
        "PatientID",
        "StudyInstanceUID",
        "PhasePercent",
        "RTSeriesInstanceUID",
        "SOPInstanceUID",
        "StructureSetLabel",
        "ROICount",
        "ROINames",
        "RTSTRUCTFile",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["PhasePercent", "RTSeriesInstanceUID", "RTSTRUCTFile"]
    ).reset_index(drop=True)


def save_geometry(path: Path, geometry: VolumeGeometry) -> None:
    path.write_text(
        json.dumps(geometry.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_geometry(path: Path) -> VolumeGeometry:
    return VolumeGeometry.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
