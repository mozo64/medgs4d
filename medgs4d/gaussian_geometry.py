from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .rtstruct import VolumeGeometry


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def ndc_to_pixel(ndc_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert rasterizer NDC coordinates [x, y] to continuous [row, col]."""

    ndc = np.asarray(ndc_xy, dtype=np.float64)
    col = ((ndc[..., 0] + 1.0) * float(width) - 1.0) * 0.5
    row = ((ndc[..., 1] + 1.0) * float(height) - 1.0) * 0.5
    return np.stack([row, col], axis=-1)


def pixel_to_ndc(points_rc: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert continuous image coordinates [row, col] to rasterizer NDC."""

    points = np.asarray(points_rc, dtype=np.float64)
    row = points[..., 0]
    col = points[..., 1]
    ndc_x = (2.0 * col + 1.0) / float(width) - 1.0
    ndc_y = (2.0 * row + 1.0) / float(height) - 1.0
    return np.stack([ndc_x, ndc_y], axis=-1)


def project_medgs_xz_to_pixel(
    points_xz: np.ndarray,
    full_projection: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Project MedGS plane coordinates [x, z] onto the original image."""

    xz = np.asarray(points_xz, dtype=np.float64)
    world = np.zeros((*xz.shape[:-1], 4), dtype=np.float64)
    world[..., 0] = xz[..., 0]
    world[..., 2] = xz[..., 1]
    world[..., 3] = 1.0
    clip = world @ np.asarray(full_projection, dtype=np.float64)
    ndc = clip[..., :2] / clip[..., 3:4]
    return ndc_to_pixel(ndc, width, height)


def unproject_pixel_to_medgs_xz(
    points_rc: np.ndarray,
    full_projection: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Unproject image [row, col] rays onto the MedGS plane y=0."""

    points = np.asarray(points_rc, dtype=np.float64)
    original_shape = points.shape[:-1]
    ndc = pixel_to_ndc(points.reshape(-1, 2), width, height)
    inverse = np.linalg.inv(np.asarray(full_projection, dtype=np.float64))

    clip_near = np.column_stack(
        [ndc[:, 0], ndc[:, 1], np.zeros(len(ndc)), np.ones(len(ndc))]
    )
    clip_far = np.column_stack(
        [ndc[:, 0], ndc[:, 1], np.ones(len(ndc)), np.ones(len(ndc))]
    )
    world_near = clip_near @ inverse
    world_far = clip_far @ inverse
    world_near = world_near[:, :3] / world_near[:, 3:4]
    world_far = world_far[:, :3] / world_far[:, 3:4]

    direction = world_far - world_near
    alpha = -world_near[:, 1] / direction[:, 1]
    intersection = world_near + alpha[:, None] * direction
    xz = intersection[:, [0, 2]]
    return xz.reshape(*original_shape, 2)


def camera_time_boundaries(time_steps: np.ndarray) -> np.ndarray:
    """Return cumulative renderer times for camera indices 0..N."""

    steps = np.asarray(time_steps, dtype=np.float64).reshape(-1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def medgs_m_to_slice_index(m: np.ndarray, time_steps: np.ndarray) -> np.ndarray:
    """Map continuous MedGS temporal center m to continuous slice index."""

    boundaries = camera_time_boundaries(time_steps)
    return np.interp(
        np.asarray(m, dtype=np.float64),
        boundaries,
        np.arange(len(boundaries), dtype=np.float64),
    )


def slice_index_to_medgs_m(
    slice_indices: np.ndarray,
    time_steps: np.ndarray,
) -> np.ndarray:
    """Map a continuous slice index to the renderer's continuous time."""

    boundaries = camera_time_boundaries(time_steps)
    return np.interp(
        np.asarray(slice_indices, dtype=np.float64),
        np.arange(len(boundaries), dtype=np.float64),
        boundaries,
    )


@dataclass(frozen=True)
class GaussianDicomTransform:
    """Map latent MedGS coordinates [x, z, m] to DICOM patient xyz."""

    geometry: VolumeGeometry
    full_projection: np.ndarray
    image_width: int
    image_height: int
    time_steps: np.ndarray

    @classmethod
    def from_canonical(
        cls,
        canonical: Any,
        geometry: VolumeGeometry,
    ) -> "GaussianDicomTransform":
        camera = canonical.cameras[0]
        return cls(
            geometry=geometry,
            full_projection=_as_numpy(camera.full_proj_transform),
            image_width=int(camera.image_width),
            image_height=int(camera.image_height),
            time_steps=_as_numpy(canonical.gaussians.get_time).reshape(-1),
        )

    @property
    def frame_count(self) -> int:
        return int(len(self.time_steps))

    def latent_to_index_zyx(self, points_xzm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xzm, dtype=np.float64)
        rc = project_medgs_xz_to_pixel(
            points[..., :2],
            self.full_projection,
            self.image_width,
            self.image_height,
        )
        slice_index = medgs_m_to_slice_index(points[..., 2], self.time_steps)
        return np.concatenate([slice_index[..., None], rc], axis=-1)

    def latent_to_patient_xyz(self, points_xzm: np.ndarray) -> np.ndarray:
        indices = self.latent_to_index_zyx(points_xzm)
        return self.geometry.index_zyx_to_patient_xyz(indices)

    def patient_xyz_to_latent(self, points_xyz: np.ndarray) -> np.ndarray:
        indices = self.geometry.patient_xyz_to_index_zyx(points_xyz)
        xz = unproject_pixel_to_medgs_xz(
            indices[..., 1:3],
            self.full_projection,
            self.image_width,
            self.image_height,
        )
        m = slice_index_to_medgs_m(indices[..., 0], self.time_steps)
        return np.concatenate([xz, m[..., None]], axis=-1)

    def patient_roundtrip(self, points_xyz: np.ndarray) -> np.ndarray:
        return self.latent_to_patient_xyz(self.patient_xyz_to_latent(points_xyz))

    def orientation_dict(self) -> dict[str, list[float] | float]:
        center = np.array([[0.0, 0.0, 0.5]], dtype=np.float64)
        base = self.latent_to_patient_xyz(center)[0]
        perturbations = {
            "patient_mm_per_medgs_x": np.array([[1.0, 0.0, 0.5]]),
            "patient_mm_per_medgs_z": np.array([[0.0, 1.0, 0.5]]),
            "patient_mm_per_medgs_m": np.array([[0.0, 0.0, 0.5 + 1e-3]]),
        }
        result: dict[str, list[float] | float] = {}
        for name, point in perturbations.items():
            delta = self.latent_to_patient_xyz(point)[0] - base
            if name.endswith("_m"):
                delta = delta / 1e-3
            result[name] = delta.tolist()
            result[name + "_norm"] = float(np.linalg.norm(delta))
        return result


def renderer_time_for_slice(slice_index: int, time_steps: np.ndarray) -> float:
    boundaries = camera_time_boundaries(time_steps)
    return float(boundaries[int(slice_index)])


def effective_gaussian_xz_at_slice(
    base_xz: np.ndarray,
    m: np.ndarray,
    sigma: np.ndarray,
    polynomial_weights: np.ndarray,
    polynomial_degree: int,
    slice_index: int,
    time_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce renderer x-z centers and temporal visibility for one slice."""

    xz = np.asarray(base_xz, dtype=np.float64).copy()
    centers = np.asarray(m, dtype=np.float64).reshape(-1)
    sigmas = np.asarray(sigma, dtype=np.float64).reshape(-1)
    weights = np.asarray(polynomial_weights, dtype=np.float64)
    time = renderer_time_for_slice(slice_index, time_steps)
    offset = centers - time

    chunks = np.split(weights, polynomial_degree, axis=1)
    for degree, chunk in enumerate(chunks, start=1):
        xz += chunk * offset[:, None] ** degree

    visibility = np.exp(-0.5 * (offset / sigmas) ** 2)
    return xz, visibility


def latent_centers_from_state(
    dynamic_xyz: Any,
    dynamic_m: Any,
) -> np.ndarray:
    xyz = _as_numpy(dynamic_xyz)
    m = _as_numpy(dynamic_m).reshape(-1)
    return np.column_stack([xyz[:, 0], xyz[:, 2], m])


def canonical_latent_centers(canonical: Any) -> np.ndarray:
    return latent_centers_from_state(canonical.xyz, canonical.m)


def inside_volume_mask(indices_zyx: np.ndarray, shape_zyx: Sequence[int]) -> np.ndarray:
    indices = np.asarray(indices_zyx, dtype=np.float64)
    shape = np.asarray(shape_zyx, dtype=np.float64)
    return np.logical_and(indices >= 0.0, indices <= shape - 1.0).all(axis=-1)
