from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Sequence
import json
import os
import re
import shutil
import time
import zipfile

import numpy as np
import pandas as pd

from .runs import build_study_dir, validate_name, write_dataframe, write_json


PHASE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
UNPACK_MARKER = ".unpack_complete.json"
Representation = Literal["raw", "denoised"]


@dataclass(frozen=True)
class StudyManifest:
    """Describe one prepared 4D-CT study and its reusable artifacts."""

    study_name: str
    patient_id: str
    study_instance_uid: str
    phases: tuple[float, ...]
    slice_count: int
    volume_shape: tuple[int, int, int]
    hu_window: tuple[float, float]
    denoise_sigma: tuple[float, float, float]
    raw_volume_paths: dict[float, str]
    denoised_volume_paths: dict[float, str]
    phase_slice_manifest_path: str
    phase_summary_path: str

    @property
    def root(self) -> Path:
        return Path(self.phase_slice_manifest_path).parent

    def volume_path(self, phase: float, representation: Representation) -> Path:
        """Return the prepared volume path for one phase and representation."""

        paths = (
            self.raw_volume_paths
            if representation == "raw"
            else self.denoised_volume_paths
        )
        matching = [key for key in paths if np.isclose(key, phase)]
        if not matching:
            raise KeyError(f"Phase {phase:g}% is not available")
        return Path(paths[matching[0]])


def phase_tag(phase: float) -> str:
    """Convert a respiratory phase into a stable filename component."""

    value = float(phase)
    if value.is_integer():
        return f"{int(value):02d}"
    return f"{value:g}".replace(".", "p")


def parse_phase_percent(description: str) -> float:
    """Extract a respiratory phase percentage from a series description."""

    match = PHASE_PATTERN.search(description)
    return float(match.group(1)) if match else float("nan")


# ---------------------------------------------------------------------------
# Archive discovery and extraction
# ---------------------------------------------------------------------------


def list_archive_patients(archives_dir: Path) -> pd.DataFrame:
    """List patients and archive counts available in the ZIP collection."""

    rows = []
    for patient_dir in sorted(path for path in archives_dir.iterdir() if path.is_dir()):
        archives = sorted(patient_dir.glob("*.zip"))
        if archives:
            rows.append(
                {
                    "PatientID": patient_dir.name,
                    "ArchiveCount": len(archives),
                    "CompressedBytes": sum(path.stat().st_size for path in archives),
                    "ArchiveDirectory": str(patient_dir),
                }
            )
    return pd.DataFrame(rows)


def discover_patient_archives(archives_dir: Path, patient_id: str) -> list[Path]:
    """Find all series archives belonging to one patient."""

    archives = sorted((archives_dir / patient_id).glob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            f"No ZIP archives found for patient {patient_id} under {archives_dir}"
        )
    return archives


def archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Return safe non-directory members stored in a ZIP archive."""

    members = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member path: {info.filename}")
        if not info.is_dir():
            members.append(info)
    if not members:
        raise ValueError("Archive contains no files")
    return members


def _member_path(root: Path, member_name: str) -> Path:
    return root.joinpath(*PurePosixPath(member_name).parts)


def _existing_tree_matches(
    destination: Path,
    members: Sequence[zipfile.ZipInfo],
) -> bool:
    if not destination.is_dir():
        return False
    return all(
        (path := _member_path(destination, info.filename)).is_file()
        and path.stat().st_size == info.file_size
        for info in members
    )


def _marker_matches(marker_path: Path, archive_path: Path) -> bool:
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
        stat = archive_path.stat()
        return (
            data.get("archive_size") == stat.st_size
            and data.get("archive_mtime_ns") == stat.st_mtime_ns
        )
    except (OSError, ValueError, TypeError):
        return False


def _write_unpack_marker(
    destination: Path,
    archive_path: Path,
    members: Sequence[zipfile.ZipInfo],
) -> None:
    stat = archive_path.stat()
    write_json(
        destination / UNPACK_MARKER,
        {
            "archive": str(archive_path),
            "archive_size": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "member_count": len(members),
            "uncompressed_bytes": sum(info.file_size for info in members),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _extract_one_archive(
    archive_path: Path,
    archives_dir: Path,
    dicom_dir: Path,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    relative = archive_path.relative_to(archives_dir)
    patient_id = relative.parent.name
    series_uid = archive_path.stem
    destination = dicom_dir / patient_id / series_uid
    started = time.perf_counter()
    row: dict[str, Any] = {
        "PatientID": patient_id,
        "SeriesInstanceUID": series_uid,
        "Archive": str(archive_path),
        "Destination": str(destination),
        "Status": "",
        "Files": 0,
        "UncompressedBytes": 0,
        "Seconds": 0.0,
        "Message": "",
    }

    try:
        if dry_run:
            row["Status"] = "dry-run"
            return row

        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive_members(archive)
            row["Files"] = len(members)
            row["UncompressedBytes"] = sum(info.file_size for info in members)

            marker = destination / UNPACK_MARKER
            if not force and _marker_matches(marker, archive_path):
                row["Status"] = "skipped"
                row["Message"] = "completion marker matches archive"
                return row
            if not force and _existing_tree_matches(destination, members):
                _write_unpack_marker(destination, archive_path, members)
                row["Status"] = "skipped"
                row["Message"] = "existing files verified"
                return row

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / f".{series_uid}.extracting"
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True)

            for info in members:
                target = _member_path(temporary, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

            if not _existing_tree_matches(temporary, members):
                raise RuntimeError("extracted file verification failed")

            shutil.rmtree(destination, ignore_errors=True)
            os.replace(temporary, destination)
            _write_unpack_marker(destination, archive_path, members)
            row["Status"] = "extracted"
            row["Message"] = "archive extracted and verified"
    except Exception as error:  # Keep a complete extraction log.
        row["Status"] = "failed"
        row["Message"] = f"{type(error).__name__}: {error}"
    finally:
        row["Seconds"] = round(time.perf_counter() - started, 3)
    return row


def extract_patient_archives(
    archives_dir: Path,
    dicom_dir: Path,
    patient_id: str,
    *,
    workers: int = 4,
    force: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Extract and verify all DICOM series archives for one patient."""

    archives = discover_patient_archives(archives_dir, patient_id)
    worker_count = max(1, min(workers, len(archives)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        rows = list(
            executor.map(
                lambda path: _extract_one_archive(
                    path, archives_dir, dicom_dir, force, dry_run
                ),
                archives,
            )
        )
    frame = pd.DataFrame(rows)
    failures = frame.loc[frame["Status"] == "failed"]
    if not failures.empty:
        preview = "\n".join(
            f"{row.Archive}: {row.Message}"
            for row in failures.head(10).itertuples()
        )
        raise RuntimeError(f"Some archives could not be extracted:\n{preview}")
    return frame


# ---------------------------------------------------------------------------
# DICOM inspection
# ---------------------------------------------------------------------------


def list_extracted_patients(dicom_dir: Path) -> pd.DataFrame:
    """List patients currently available in the extracted DICOM directory."""

    rows = []
    for patient_dir in sorted(path for path in dicom_dir.iterdir() if path.is_dir()):
        series = [path for path in patient_dir.iterdir() if path.is_dir()]
        rows.append(
            {
                "PatientID": patient_dir.name,
                "ExtractedSeries": len(series),
                "PatientDirectory": str(patient_dir),
            }
        )
    return pd.DataFrame(rows)


def list_dicom_files(series_dir: Path) -> list[Path]:
    """List DICOM-like files recursively inside one extracted series."""

    files = [path for path in series_dir.rglob("*") if path.is_file()]
    return sorted(path for path in files if path.name != UNPACK_MARKER)


def read_dicom_header(path: Path) -> Any:
    """Read one DICOM file without loading its pixel array."""

    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def _float_vector(dataset: Any, attribute: str, length: int) -> np.ndarray:
    values = np.asarray(
        [float(value) for value in getattr(dataset, attribute)], dtype=np.float64
    )
    if values.size != length:
        raise ValueError(f"{attribute} has length {values.size}; expected {length}")
    return values


def slice_coordinate(dataset: Any) -> float:
    """Calculate physical position along the DICOM slice normal."""

    position = _float_vector(dataset, "ImagePositionPatient", 3)
    orientation = _float_vector(dataset, "ImageOrientationPatient", 6)
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    return float(np.dot(position, normal))


def scan_patient_series(dicom_dir: Path, patient_id: str) -> pd.DataFrame:
    """Collect study, series, phase, and geometry metadata for one patient."""

    patient_dir = dicom_dir / patient_id
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Extracted patient directory not found: {patient_dir}")
    rows = []
    for series_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
        files = list_dicom_files(series_dir)
        if not files:
            continue
        dataset = read_dicom_header(files[0])
        description = str(getattr(dataset, "SeriesDescription", ""))
        rows.append(
            {
                "PatientID": str(getattr(dataset, "PatientID", patient_id)),
                "StudyDate": str(getattr(dataset, "StudyDate", "")),
                "StudyDescription": str(getattr(dataset, "StudyDescription", "")),
                "StudyInstanceUID": str(getattr(dataset, "StudyInstanceUID", "")),
                "Modality": str(getattr(dataset, "Modality", "")),
                "PhasePercent": parse_phase_percent(description),
                "SeriesNumber": getattr(dataset, "SeriesNumber", ""),
                "SeriesDescription": description,
                "SeriesInstanceUID": str(getattr(dataset, "SeriesInstanceUID", "")),
                "DICOMFiles": len(files),
                "SeriesPath": str(series_dir),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No readable DICOM series found in {patient_dir}")
    return frame


def list_patient_series(
    dicom_dir: Path,
    patient_id: str,
    *,
    study_instance_uid: str | None = None,
    modality: str | None = None,
) -> pd.DataFrame:
    """List complete series metadata without selecting respiratory CT phases."""

    series = scan_patient_series(dicom_dir, patient_id)
    if study_instance_uid is not None:
        series = series.loc[
            series["StudyInstanceUID"].astype(str) == str(study_instance_uid)
        ]
    if modality is not None:
        series = series.loc[
            series["Modality"].astype(str).str.upper() == modality.upper()
        ]
    return series.sort_values(
        ["StudyDate", "StudyInstanceUID", "Modality", "PhasePercent", "SeriesInstanceUID"],
        na_position="last",
    ).reset_index(drop=True)


def list_patient_studies(dicom_dir: Path, patient_id: str) -> pd.DataFrame:
    """Summarize studies and expose modality-specific series counts."""

    series = scan_patient_series(dicom_dir, patient_id)
    rows = []
    for uid, group in series.groupby("StudyInstanceUID", sort=False):
        modality = group["Modality"].astype(str).str.upper()
        ct = group.loc[modality == "CT"]
        rtstruct = group.loc[modality == "RTSTRUCT"]
        phases = sorted(ct["PhasePercent"].dropna().astype(float).unique())
        modalities = sorted(value for value in modality.unique() if value)
        rows.append(
            {
                "PatientID": patient_id,
                "StudyInstanceUID": uid,
                "StudyDate": group["StudyDate"].iloc[0],
                "StudyDescription": group["StudyDescription"].iloc[0],
                "Modalities": ", ".join(modalities),
                "SeriesCount": len(group),
                "CTSeriesCount": len(ct),
                "RTSTRUCTSeriesCount": len(rtstruct),
                "OtherSeriesCount": len(group) - len(ct) - len(rtstruct),
                "PhaseCount": len(phases),
                "Phases": ", ".join(f"{phase:g}" for phase in phases),
                "MaxSlicesPerSeries": int(ct["DICOMFiles"].max()) if not ct.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["PhaseCount", "MaxSlicesPerSeries", "RTSTRUCTSeriesCount"],
        ascending=False,
    ).reset_index(drop=True)


def inspect_study(
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
) -> pd.DataFrame:
    """List respiratory phases, series, slice counts, and geometry for one study."""

    series = scan_patient_series(dicom_dir, patient_id)
    selected = series.loc[
        (series["StudyInstanceUID"].astype(str) == str(study_instance_uid))
        & (series["Modality"] == "CT")
        & series["PhasePercent"].notna()
    ].copy()
    if selected.empty:
        raise ValueError(f"No respiratory CT series found for study {study_instance_uid}")
    selected = (
        selected.sort_values(
            ["PhasePercent", "DICOMFiles", "SeriesInstanceUID"],
            ascending=[True, False, True],
        )
        .drop_duplicates("PhasePercent", keep="first")
        .sort_values("PhasePercent")
        .reset_index(drop=True)
    )
    return selected



def _rtstruct_referenced_series_uids(dataset: Any) -> set[str]:
    """Collect CT SeriesInstanceUID values referenced by one RTSTRUCT object."""

    referenced: set[str] = set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", ()) or ():
        for study in getattr(frame, "RTReferencedStudySequence", ()) or ():
            for series in getattr(study, "RTReferencedSeriesSequence", ()) or ():
                uid = str(getattr(series, "SeriesInstanceUID", ""))
                if uid:
                    referenced.add(uid)
    return referenced


def _rtstruct_roi_names(dataset: Any) -> list[str]:
    """Return ordered unique ROI names declared by one RTSTRUCT object."""

    names = []
    seen = set()
    for roi in getattr(dataset, "StructureSetROISequence", ()) or ():
        name = str(getattr(roi, "ROIName", "")).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _rtstruct_contour_summary(dataset: Any) -> tuple[int, set[str]]:
    """Count contour records and referenced contour-image SOP instances."""

    contour_count = 0
    image_uids: set[str] = set()
    for roi_contour in getattr(dataset, "ROIContourSequence", ()) or ():
        for contour in getattr(roi_contour, "ContourSequence", ()) or ():
            contour_count += 1
            for image in getattr(contour, "ContourImageSequence", ()) or ():
                uid = str(getattr(image, "ReferencedSOPInstanceUID", ""))
                if uid:
                    image_uids.add(uid)
    return contour_count, image_uids


def _build_ct_sop_series_index(series: pd.DataFrame) -> dict[str, str]:
    """Map local CT SOPInstanceUID values to their parent series."""

    index: dict[str, str] = {}
    ct = series.loc[series["Modality"].astype(str).str.upper() == "CT"]
    for row in ct.itertuples():
        series_uid = str(row.SeriesInstanceUID)
        for path in list_dicom_files(Path(row.SeriesPath)):
            dataset = read_dicom_header(path)
            sop_uid = str(getattr(dataset, "SOPInstanceUID", ""))
            if sop_uid:
                index[sop_uid] = series_uid
    return index


def list_rtstructs(
    dicom_dir: Path,
    patient_id: str,
    *,
    referenced_study_uid: str | None = None,
) -> pd.DataFrame:
    """Inventory RTSTRUCT objects and resolve their referenced local CT series."""

    series = scan_patient_series(dicom_dir, patient_id)
    rt_series = series.loc[
        series["Modality"].astype(str).str.upper() == "RTSTRUCT"
    ]
    if rt_series.empty:
        return pd.DataFrame(
            columns=[
                "PatientID",
                "RTStudyInstanceUID",
                "RTSeriesInstanceUID",
                "StudyDate",
                "SeriesDescription",
                "StructureSetLabel",
                "ROICount",
                "ROINames",
                "ContourCount",
                "ContourImageCount",
                "ReferencedCTSeriesCount",
                "ReferencedCTStudyUIDs",
                "ReferencedCTSeriesUIDs",
                "ReferencedPhases",
                "ReferencedSeriesDescriptions",
                "DICOMFile",
                "SeriesPath",
            ]
        )

    series_by_uid = {
        str(row.SeriesInstanceUID): row
        for row in series.itertuples()
    }
    sop_to_series: dict[str, str] | None = None
    rows = []

    for rt_row in rt_series.itertuples():
        for path in list_dicom_files(Path(rt_row.SeriesPath)):
            dataset = read_dicom_header(path)
            if str(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT":
                continue

            referenced_series = _rtstruct_referenced_series_uids(dataset)
            contour_count, contour_image_uids = _rtstruct_contour_summary(dataset)

            if not referenced_series and contour_image_uids:
                if sop_to_series is None:
                    sop_to_series = _build_ct_sop_series_index(series)
                referenced_series.update(
                    sop_to_series[uid]
                    for uid in contour_image_uids
                    if uid in sop_to_series
                )

            referenced_rows = [
                series_by_uid[uid]
                for uid in sorted(referenced_series)
                if uid in series_by_uid
            ]
            referenced_studies = sorted(
                {
                    str(row.StudyInstanceUID)
                    for row in referenced_rows
                    if str(row.StudyInstanceUID)
                }
            )
            referenced_phases = sorted(
                {
                    float(row.PhasePercent)
                    for row in referenced_rows
                    if pd.notna(row.PhasePercent)
                }
            )
            referenced_descriptions = sorted(
                {
                    str(row.SeriesDescription)
                    for row in referenced_rows
                    if str(row.SeriesDescription)
                }
            )
            roi_names = _rtstruct_roi_names(dataset)

            row = {
                "PatientID": str(getattr(dataset, "PatientID", patient_id)),
                "RTStudyInstanceUID": str(
                    getattr(dataset, "StudyInstanceUID", rt_row.StudyInstanceUID)
                ),
                "RTSeriesInstanceUID": str(
                    getattr(dataset, "SeriesInstanceUID", rt_row.SeriesInstanceUID)
                ),
                "StudyDate": str(getattr(dataset, "StudyDate", rt_row.StudyDate)),
                "SeriesDescription": str(
                    getattr(dataset, "SeriesDescription", rt_row.SeriesDescription)
                ),
                "StructureSetLabel": str(
                    getattr(dataset, "StructureSetLabel", "")
                ),
                "ROICount": len(roi_names),
                "ROINames": " | ".join(roi_names),
                "ContourCount": contour_count,
                "ContourImageCount": len(contour_image_uids),
                "ReferencedCTSeriesCount": len(referenced_series),
                "ReferencedCTStudyUIDs": ", ".join(referenced_studies),
                "ReferencedCTSeriesUIDs": ", ".join(sorted(referenced_series)),
                "ReferencedPhases": ", ".join(
                    f"{phase:g}" for phase in referenced_phases
                ),
                "ReferencedSeriesDescriptions": " | ".join(
                    referenced_descriptions
                ),
                "DICOMFile": str(path),
                "SeriesPath": str(rt_row.SeriesPath),
            }
            if (
                referenced_study_uid is None
                or str(referenced_study_uid) in referenced_studies
            ):
                rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "ReferencedCTStudyUIDs",
            "ReferencedPhases",
            "RTStudyInstanceUID",
            "RTSeriesInstanceUID",
            "DICOMFile",
        ],
        na_position="last",
    ).reset_index(drop=True)


def collect_phase_series(
    dicom_dir: Path,
    patient_id: str,
    study_instance_uid: str,
) -> dict[float, tuple[Path, str]]:
    """Resolve one selected CT series for each respiratory phase."""

    frame = inspect_study(dicom_dir, patient_id, study_instance_uid)
    return {
        float(row.PhasePercent): (Path(row.SeriesPath), str(row.SeriesInstanceUID))
        for row in frame.itertuples()
    }


def read_slice_table(series_dir: Path, expected_series_uid: str) -> pd.DataFrame:
    """Read and physically order all CT slices from one DICOM series."""

    rows = []
    for path in list_dicom_files(series_dir):
        dataset = read_dicom_header(path)
        if str(getattr(dataset, "Modality", "")) != "CT":
            continue
        if str(getattr(dataset, "SeriesInstanceUID", "")) != expected_series_uid:
            raise ValueError(f"Unexpected SeriesInstanceUID in {path}")
        position = _float_vector(dataset, "ImagePositionPatient", 3)
        orientation = _float_vector(dataset, "ImageOrientationPatient", 6)
        pixel_spacing = _float_vector(dataset, "PixelSpacing", 2)
        rows.append(
            {
                "File": path.name,
                "Path": str(path),
                "SOPInstanceUID": str(getattr(dataset, "SOPInstanceUID", "")),
                "InstanceNumber": int(getattr(dataset, "InstanceNumber", -1)),
                "SliceCoordinate": slice_coordinate(dataset),
                "PositionX": float(position[0]),
                "PositionY": float(position[1]),
                "PositionZ": float(position[2]),
                "Rows": int(dataset.Rows),
                "Columns": int(dataset.Columns),
                "PixelSpacingRow": float(pixel_spacing[0]),
                "PixelSpacingColumn": float(pixel_spacing[1]),
                "SliceThickness": float(getattr(dataset, "SliceThickness", np.nan)),
                "Orientation": json.dumps([float(value) for value in orientation]),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["SliceCoordinate", "InstanceNumber", "File"]
    ).reset_index(drop=True)
    frame.insert(0, "SliceIndex", np.arange(len(frame), dtype=int))
    return frame


def validate_slice_geometry(
    phase_tables: dict[float, pd.DataFrame],
    *,
    spacing_tolerance: float = 1e-3,
) -> None:
    """Validate shared shape, orientation, and slice positions across phases."""

    if not phase_tables:
        raise ValueError("No respiratory phases were resolved")
    phases = sorted(phase_tables)
    reference = phase_tables[phases[0]]
    reference_coordinates = reference["SliceCoordinate"].to_numpy(float)
    reference_shape = (int(reference.iloc[0]["Rows"]), int(reference.iloc[0]["Columns"]))
    reference_orientation = reference.iloc[0]["Orientation"]
    if len(reference_coordinates) < 2 or not np.all(np.diff(reference_coordinates) > 0):
        raise ValueError("Reference slice coordinates are not strictly increasing")

    for phase in phases:
        table = phase_tables[phase]
        shape = (int(table.iloc[0]["Rows"]), int(table.iloc[0]["Columns"]))
        if shape != reference_shape:
            raise ValueError(f"Image shape differs at phase {phase:g}%")
        if len(table) != len(reference):
            raise ValueError(f"Slice count differs at phase {phase:g}%")
        if table.iloc[0]["Orientation"] != reference_orientation:
            raise ValueError(f"Orientation differs at phase {phase:g}%")
        coordinates = table["SliceCoordinate"].to_numpy(float)
        if not np.allclose(coordinates, reference_coordinates, atol=spacing_tolerance):
            raise ValueError(f"Slice positions differ at phase {phase:g}%")


def read_hu_image(path: Path) -> np.ndarray:
    """Read one DICOM CT slice and convert pixels to Hounsfield units."""

    import pydicom

    dataset = pydicom.dcmread(str(path), force=True)
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return pixels * slope + intercept


def reconstruct_hu_volume(paths: Sequence[Path]) -> np.ndarray:
    """Reconstruct one ordered CT volume in Hounsfield units."""

    return np.stack([read_hu_image(path) for path in paths], axis=0).astype(np.float32)


def denoise_hu_volume(
    volume: np.ndarray,
    sigma: tuple[float, float, float],
) -> np.ndarray:
    """Apply the configured mild three-dimensional Gaussian filter."""

    from scipy.ndimage import gaussian_filter

    return gaussian_filter(volume, sigma=sigma, mode="nearest").astype(np.float32)


def prepare_study(
    dicom_dir: Path,
    prepared_root: Path,
    *,
    patient_id: str,
    study_instance_uid: str,
    study_name: str,
    hu_window: tuple[float, float] = (-1000.0, 400.0),
    denoise_sigma: tuple[float, float, float] = (0.20, 0.40, 0.40),
    force: bool = False,
) -> StudyManifest:
    """Prepare all respiratory phases and save reusable HU volumes and manifests."""

    validate_name(study_name, "study name")
    study_dir = build_study_dir(prepared_root, study_name)
    if study_dir.exists():
        if not force:
            raise FileExistsError(
                f"Prepared study already exists: {study_dir}\nUse --force to replace it."
            )
        shutil.rmtree(study_dir)

    resolved = collect_phase_series(dicom_dir, patient_id, study_instance_uid)
    phase_tables = {
        phase: read_slice_table(series_dir, series_uid)
        for phase, (series_dir, series_uid) in resolved.items()
    }
    validate_slice_geometry(phase_tables)

    raw_dir = study_dir / "volumes" / "raw"
    denoised_dir = study_dir / "volumes" / "denoised"
    raw_dir.mkdir(parents=True)
    denoised_dir.mkdir(parents=True)
    manifest_parts = []
    summary_rows = []
    raw_paths: dict[float, str] = {}
    denoised_paths: dict[float, str] = {}

    for phase in sorted(phase_tables):
        table = phase_tables[phase].copy()
        volume = reconstruct_hu_volume([Path(path) for path in table["Path"]])
        raw_path = raw_dir / f"phase_{phase_tag(phase)}.npy"
        denoised_path = denoised_dir / f"phase_{phase_tag(phase)}.npy"
        np.save(raw_path, volume)
        np.save(denoised_path, denoise_hu_volume(volume, denoise_sigma))
        raw_paths[phase] = str(raw_path.resolve())
        denoised_paths[phase] = str(denoised_path.resolve())

        table.insert(0, "PhasePercent", phase)
        table.insert(0, "PatientID", patient_id)
        table.insert(1, "StudyInstanceUID", study_instance_uid)
        table["RawVolumePath"] = str(raw_path.resolve())
        table["DenoisedVolumePath"] = str(denoised_path.resolve())
        manifest_parts.append(table)
        coordinates = table["SliceCoordinate"].to_numpy(float)
        summary_rows.append(
            {
                "PhasePercent": phase,
                "Slices": len(table),
                "Rows": volume.shape[1],
                "Columns": volume.shape[2],
                "FirstCoordinate": float(coordinates[0]),
                "LastCoordinate": float(coordinates[-1]),
                "MedianSpacing": float(np.median(np.diff(coordinates))),
                "RawVolumePath": str(raw_path.resolve()),
                "DenoisedVolumePath": str(denoised_path.resolve()),
            }
        )

    phase_slice_path = study_dir / "phase_slice_manifest.csv"
    phase_summary_path = study_dir / "phase_summary.csv"
    write_dataframe(phase_slice_path, pd.concat(manifest_parts, ignore_index=True))
    write_dataframe(phase_summary_path, pd.DataFrame(summary_rows))

    first_volume = np.load(next(iter(raw_paths.values())), mmap_mode="r")
    manifest = StudyManifest(
        study_name=study_name,
        patient_id=patient_id,
        study_instance_uid=study_instance_uid,
        phases=tuple(sorted(raw_paths)),
        slice_count=int(first_volume.shape[0]),
        volume_shape=tuple(int(value) for value in first_volume.shape),
        hu_window=tuple(float(value) for value in hu_window),
        denoise_sigma=tuple(float(value) for value in denoise_sigma),
        raw_volume_paths=raw_paths,
        denoised_volume_paths=denoised_paths,
        phase_slice_manifest_path=str(phase_slice_path.resolve()),
        phase_summary_path=str(phase_summary_path.resolve()),
    )
    save_study_manifest(manifest, study_dir)
    return manifest


def save_study_manifest(manifest: StudyManifest, study_dir: Path) -> None:
    """Save prepared-study metadata and artifact paths."""

    payload = asdict(manifest)
    payload["raw_volume_paths"] = {
        f"{float(key):g}": value for key, value in manifest.raw_volume_paths.items()
    }
    payload["denoised_volume_paths"] = {
        f"{float(key):g}": value
        for key, value in manifest.denoised_volume_paths.items()
    }
    write_json(study_dir / "manifest.json", payload)


def load_study_manifest(study_dir: Path) -> StudyManifest:
    """Load and validate metadata for a prepared study."""

    path = study_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["phases"] = tuple(float(value) for value in data["phases"])
    data["volume_shape"] = tuple(int(value) for value in data["volume_shape"])
    data["hu_window"] = tuple(float(value) for value in data["hu_window"])
    data["denoise_sigma"] = tuple(float(value) for value in data["denoise_sigma"])
    data["raw_volume_paths"] = {
        float(key): value for key, value in data["raw_volume_paths"].items()
    }
    data["denoised_volume_paths"] = {
        float(key): value for key, value in data["denoised_volume_paths"].items()
    }
    manifest = StudyManifest(**data)
    if not Path(manifest.phase_slice_manifest_path).is_file():
        raise FileNotFoundError(manifest.phase_slice_manifest_path)
    return manifest


@lru_cache(maxsize=32)
def _load_cached_volume(path: str, mmap_mode: str) -> np.ndarray:
    return np.load(path, mmap_mode=mmap_mode)


def load_phase_volume(
    study: StudyManifest,
    phase: float,
    *,
    representation: Representation = "raw",
    mmap_mode: str | None = "r",
) -> np.ndarray:
    """Load one prepared HU volume for the requested phase."""

    path = str(study.volume_path(phase, representation))
    if mmap_mode is None:
        return np.load(path)
    return _load_cached_volume(path, mmap_mode)


def build_phase_slice_index(study: StudyManifest) -> pd.DataFrame:
    """Build the complete phase-by-slice index for training and evaluation."""

    frame = pd.read_csv(study.phase_slice_manifest_path)
    return (
        frame[["PhasePercent", "SliceIndex", "SliceCoordinate"]]
        .sort_values(["PhasePercent", "SliceIndex"])
        .reset_index(drop=True)
    )


def window_hu(image_hu: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    """Window HU values to the normalized range zero to one."""

    low, high = hu_window
    if high <= low:
        raise ValueError("HU window high must exceed low")
    clipped = np.clip(image_hu, low, high)
    return (clipped - low) / (high - low)


def window_hu_to_uint8(
    image_hu: np.ndarray,
    hu_window: tuple[float, float],
) -> np.ndarray:
    """Window HU values and reproduce MedGS-compatible uint8 rounding."""

    return np.round(window_hu(image_hu, hu_window) * 255.0).astype(np.uint8)


def grayscale_to_rgb_uint8(image_uint8: np.ndarray) -> np.ndarray:
    """Repeat one uint8 grayscale image into three identical channels."""

    return np.repeat(image_uint8[..., None], 3, axis=2)


def load_target_tensor(
    study: StudyManifest,
    phase: float,
    slice_index: int,
    *,
    representation: Representation = "raw",
    device: str = "cuda",
):
    """Load one CT target as a three-channel float tensor in zero to one."""

    import torch

    volume = load_phase_volume(study, phase, representation=representation)
    image = window_hu_to_uint8(np.asarray(volume[int(slice_index)]), study.hu_window)
    tensor = torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).repeat(3, 1, 1)
