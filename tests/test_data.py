from pathlib import Path
import zipfile

import numpy as np

from medgs4d.data import (
    list_archive_patients,
    phase_tag,
    window_hu_to_uint8,
)


def test_archive_patient_listing(tmp_path: Path) -> None:
    patient = tmp_path / "117_HM10395"
    patient.mkdir()
    for index in range(2):
        with zipfile.ZipFile(patient / f"series{index}.zip", "w") as archive:
            archive.writestr("image.dcm", b"dicom")
    frame = list_archive_patients(tmp_path)
    assert frame.loc[0, "PatientID"] == "117_HM10395"
    assert frame.loc[0, "ArchiveCount"] == 2


def test_uint8_window_matches_notebook_contract() -> None:
    values = np.asarray([[-1000.0, -300.0, 400.0]], dtype=np.float32)
    result = window_hu_to_uint8(values, (-1000.0, 400.0))
    assert result.tolist() == [[0, 128, 255]]


def test_phase_tag_is_stable() -> None:
    assert phase_tag(0) == "00"
    assert phase_tag(20) == "20"
    assert phase_tag(12.5) == "12p5"


def test_archive_extraction_is_resumable(tmp_path: Path) -> None:
    from medgs4d.data import extract_patient_archives

    archives = tmp_path / "archives"
    dicom = tmp_path / "dicom"
    patient = archives / "p1"
    patient.mkdir(parents=True)
    with zipfile.ZipFile(patient / "series1.zip", "w") as archive:
        archive.writestr("0001.dcm", b"first")
        archive.writestr("0002.dcm", b"second")

    first = extract_patient_archives(archives, dicom, "p1", workers=1)
    second = extract_patient_archives(archives, dicom, "p1", workers=1)
    assert first.loc[0, "Status"] == "extracted"
    assert second.loc[0, "Status"] == "skipped"
    assert (dicom / "p1" / "series1" / "0002.dcm").read_bytes() == b"second"


def test_slice_coordinate_uses_physical_normal() -> None:
    from types import SimpleNamespace
    from medgs4d.data import slice_coordinate

    dataset = SimpleNamespace(
        ImagePositionPatient=[0.0, 0.0, 12.5],
        ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    assert slice_coordinate(dataset) == 12.5
