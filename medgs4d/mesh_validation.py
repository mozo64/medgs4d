from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

from .meshes import TriangleMesh
from .rtstruct import RoiContour, VolumeGeometry


def window_hu(image: np.ndarray, low: float = -1000.0, high: float = 400.0) -> np.ndarray:
    clipped = np.clip(np.asarray(image, dtype=np.float32), low, high)
    return (clipped - low) / (high - low)


def mask_boundary(mask_2d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    binary = np.asarray(mask_2d, dtype=bool)
    return binary & ~binary_erosion(binary)


def save_validation_overview(
    path: Path,
    ct_volume: np.ndarray,
    mask: np.ndarray,
    roundtrip_mask: np.ndarray,
    contours: Sequence[RoiContour],
    geometry: VolumeGeometry,
    hu_window: tuple[float, float] = (-1000.0, 400.0),
) -> None:
    """Save first, largest, and last annotated axial slices."""

    import matplotlib.pyplot as plt

    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    nonempty = np.flatnonzero(areas)
    slices = [int(nonempty[0]), int(np.argmax(areas)), int(nonempty[-1])]
    contour_indices = []
    sop_to_slice = {uid: i for i, uid in enumerate(geometry.sop_instance_uids)}
    for contour in contours:
        points_zyx = geometry.patient_xyz_to_index_zyx(contour.points_xyz)
        slice_index = sop_to_slice.get(
            contour.referenced_sop_instance_uid,
            int(round(float(points_zyx[:, 0].mean()))),
        )
        contour_indices.append((slice_index, points_zyx))

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, slice_index in zip(axes, slices):
        axis.imshow(window_hu(ct_volume[slice_index], *hu_window), cmap="gray")
        axis.contour(mask[slice_index], levels=[0.5], linewidths=1.5)
        axis.contour(roundtrip_mask[slice_index], levels=[0.5], linewidths=1.0, linestyles="--")
        for contour_slice, points_zyx in contour_indices:
            if contour_slice == slice_index:
                closed = np.vstack([points_zyx[:, [2, 1]], points_zyx[0, [2, 1]]])
                axis.plot(closed[:, 0], closed[:, 1], linewidth=1.0)
        axis.set_title(f"Axial slice {slice_index}")
        axis.set_axis_off()
    figure.suptitle("CT + RTSTRUCT contour + rasterized mask + mesh round-trip")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_case(case_dir: Path) -> dict[str, Any]:
    """Load artifacts produced by scripts/rtstruct_mesh.py build."""

    from .meshes import load_mesh_npz
    from .rtstruct import load_ct_volume, load_geometry

    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    geometry = load_geometry(case_dir / manifest["geometry_file"])
    mask = np.load(case_dir / manifest["mask_file"])
    roundtrip_mask = np.load(case_dir / manifest["roundtrip_mask_file"])
    mesh = load_mesh_npz(case_dir / manifest["mesh_npz_file"])
    ct_volume = load_ct_volume(geometry)
    contours = json.loads((case_dir / manifest["contours_file"]).read_text(encoding="utf-8"))
    report = json.loads((case_dir / manifest["validation_report_file"]).read_text(encoding="utf-8"))
    return {
        "manifest": manifest,
        "geometry": geometry,
        "ct_volume": ct_volume,
        "mask": mask,
        "roundtrip_mask": roundtrip_mask,
        "mesh": mesh,
        "contours": contours,
        "report": report,
    }
