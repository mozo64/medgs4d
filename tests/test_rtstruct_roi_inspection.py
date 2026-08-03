from __future__ import annotations

from types import SimpleNamespace

from medgs4d.rtstruct import summarize_rtstruct_rois


def item(**values):
    return SimpleNamespace(**values)


def contour(z: float, geometric_type: str = "CLOSED_PLANAR", points: int = 4):
    coordinates = []
    for index in range(points):
        coordinates.extend([float(index), float(index % 2), z])
    return item(
        ContourData=coordinates,
        ContourGeometricType=geometric_type,
    )


def dataset():
    return item(
        PatientID="117_HM10395",
        StudyInstanceUID="study",
        SeriesDescription="Gated, 0.0%",
        StructureSetLabel="Plan_0",
        StructureSetROISequence=[
            item(ROINumber=1, ROIName="Tumor_c00"),
            item(ROINumber=2, ROIName="Carina_c00"),
            item(ROINumber=3, ROIName="Empty_c00"),
            item(ROINumber=4, ROIName="TwoParts_c00"),
        ],
        ROIContourSequence=[
            item(
                ReferencedROINumber=1,
                ContourSequence=[contour(-3.0), contour(0.0), contour(3.0)],
            ),
            item(
                ReferencedROINumber=2,
                ContourSequence=[contour(0.0)],
            ),
            item(
                ReferencedROINumber=4,
                ContourSequence=[
                    contour(0.0),
                    contour(0.0),
                    contour(3.0),
                ],
            ),
        ],
    )


def test_summarize_rtstruct_rois_marks_volume_candidates():
    frame = summarize_rtstruct_rois(dataset(), "/tmp/rtstruct.dcm", 0.0)
    rows = frame.set_index("ROIName")

    tumor = rows.loc["Tumor_c00"]
    assert tumor["ContourCount"] == 3
    assert tumor["DistinctContourSlices"] == 3
    assert tumor["ClosedPlanar"]
    assert tumor["VolumeCandidate"]
    assert tumor["ZExtentMm"] == 6.0

    carina = rows.loc["Carina_c00"]
    assert carina["DistinctContourSlices"] == 1
    assert carina["ClosedPlanar"]
    assert not carina["VolumeCandidate"]


def test_summarize_rtstruct_rois_keeps_empty_and_multi_contour_rois():
    frame = summarize_rtstruct_rois(dataset(), "/tmp/rtstruct.dcm", 0.0)
    rows = frame.set_index("ROIName")

    empty = rows.loc["Empty_c00"]
    assert empty["ContourCount"] == 0
    assert empty["PointCount"] == 0
    assert not empty["ClosedPlanar"]
    assert not empty["VolumeCandidate"]

    two_parts = rows.loc["TwoParts_c00"]
    assert two_parts["ContourCount"] == 3
    assert two_parts["DistinctContourSlices"] == 2
    assert two_parts["ContoursPerSliceMax"] == 2
    assert two_parts["VolumeCandidate"]
