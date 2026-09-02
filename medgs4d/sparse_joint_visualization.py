from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import Any, Sequence

from .meshes import TriangleMesh, load_mesh_npz
from .rtstruct import load_geometry

REFERENCE_COLOR = "limegreen"
PREDICTION_COLOR = "crimson"


def _window_hu(ct_hu: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    low, high = (float(value) for value in hu_window)
    clipped = np.clip(np.asarray(ct_hu, dtype=np.float32), low, high)
    return (clipped - low) / (high - low)


def show_sparse_joint_slice_browser(
        ground_truth_ct: np.ndarray,
        predicted_ct: np.ndarray,
        reference_mask: np.ndarray,
        predicted_mask: np.ndarray,
        observed_slices: Sequence[int],
        *,
        slice_start: int = 0,
) -> Any:
    """Interactive GT-vs-MedGS browser using original CT slice indices."""

    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    gt = np.asarray(ground_truth_ct)
    pred = np.asarray(predicted_ct)
    ref_mask = np.asarray(reference_mask, dtype=bool)
    pred_mask = np.asarray(predicted_mask, dtype=bool)
    observed = {int(value) for value in observed_slices}

    if not (gt.shape == pred.shape == ref_mask.shape == pred_mask.shape):
        raise ValueError("CT and mask volumes must have identical shapes")

    slice_count = gt.shape[0]
    slice_end = slice_start + slice_count - 1

    slider = widgets.IntSlider(
        min=slice_start,
        max=slice_end,
        value=slice_start + slice_count // 2,
        description="CT slice",
        continuous_update=False,
        layout=widgets.Layout(width="650px"),
    )

    output = widgets.Output()

    def update(*_: object) -> None:
        slice_index = int(slider.value)
        local_index = slice_index - slice_start

        status = (
            "observed"
            if slice_index in observed
            else "missing"
        )

        with output:
            output.clear_output(wait=True)

            figure, axes = plt.subplots(
                2,
                2,
                figsize=(9, 8),
                constrained_layout=True,
            )

            axes[0, 0].imshow(
                gt[local_index],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[0, 0].set_title("GT CT")

            axes[0, 1].imshow(
                pred[local_index],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[0, 1].set_title("MedGS joint CT")

            axes[1, 0].imshow(
                ref_mask[local_index],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[1, 0].set_title("GT mask")

            axes[1, 1].imshow(
                pred_mask[local_index],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[1, 1].set_title("MedGS joint mask")

            for axis in axes.ravel():
                axis.axis("off")

            figure.suptitle(
                f"CT slice {slice_index} — {status}"
            )

            plt.show()

    slider.observe(update, names="value")
    update()

    box = widgets.VBox(
        [
            slider,
            output,
        ]
    )

    display(box)
    return box


def show_sparse_joint_orthogonal_views(
        ct_hu: np.ndarray,
        reference_mask: np.ndarray,
        predicted_mask: np.ndarray,
        geometry,
) -> Any:
    """Show axial/coronal/sagittal CT with GT and predicted tumor contours."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ct = np.asarray(ct_hu)
    reference = np.asarray(reference_mask, dtype=bool)
    prediction = np.asarray(predicted_mask, dtype=bool)
    combined = reference | prediction
    occupied = np.argwhere(combined)
    if not len(occupied):
        raise ValueError("Reference and predicted masks are both empty")

    z_mid, y_mid, x_mid = np.round(occupied.mean(axis=0)).astype(int)
    spacing_z, spacing_y, spacing_x = geometry.spacing_zyx

    views = [
        (
            ct[z_mid],
            reference[z_mid],
            prediction[z_mid],
            spacing_y / spacing_x,
            f"Axial z={z_mid}",
        ),
        (
            np.flipud(ct[:, y_mid, :]),
            np.flipud(reference[:, y_mid, :]),
            np.flipud(prediction[:, y_mid, :]),
            spacing_z / spacing_x,
            f"Coronal y={y_mid}",
        ),
        (
            np.flipud(ct[:, :, x_mid]),
            np.flipud(reference[:, :, x_mid]),
            np.flipud(prediction[:, :, x_mid]),
            spacing_z / spacing_y,
            f"Sagittal x={x_mid}",
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.86, wspace=0.04)
    for axis, (ct_view, ref_view, pred_view, aspect, title) in zip(axes, views):
        axis.imshow(
            _window_hu(ct_view, (-1000.0, 400.0)),
            cmap="gray",
            aspect=aspect,
            origin="upper",
        )
        if ref_view.any():
            axis.contour(ref_view, levels=[0.5], colors=[REFERENCE_COLOR], linewidths=2.0)
        if pred_view.any():
            axis.contour(pred_view, levels=[0.5], colors=[PREDICTION_COLOR], linewidths=2.0)
        axis.set_title(title)
        axis.set_axis_off()

    figure.legend(
        handles=[
            Line2D([0], [0], color=REFERENCE_COLOR, linewidth=2, label="RTSTRUCT reference"),
            Line2D([0], [0], color=PREDICTION_COLOR, linewidth=2, label="MedGS reconstruction"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Sparse-joint tumor reconstruction", fontsize=15)
    plt.show()
    return figure


def _keep_largest_components(mask: np.ndarray, count: int, min_size: int) -> np.ndarray:
    from scipy import ndimage as ndi

    labels, component_count = ndi.label(mask)
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = [
        int(label)
        for label in np.argsort(sizes)[::-1]
        if sizes[label] >= int(min_size)
    ][: int(count)]
    return np.isin(labels, selected)


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    from scipy import ndimage as ndi

    labels, component_count = ndi.label(mask)
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels]


def _build_body_mask(ct_hu: np.ndarray, threshold_hu: float = -700.0) -> np.ndarray:
    from scipy import ndimage as ndi

    body = np.zeros_like(ct_hu, dtype=bool)
    for z_index, ct_slice in enumerate(ct_hu):
        labels, component_count = ndi.label(ct_slice > float(threshold_hu))
        if component_count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        selected = labels == int(np.argmax(sizes))
        selected = ndi.binary_closing(selected, structure=np.ones((3, 3), dtype=bool))
        body[z_index] = ndi.binary_fill_holes(selected)
    return ndi.binary_closing(body, structure=np.ones((3, 3, 3), dtype=bool))


def _mask_to_surface(mask: np.ndarray, geometry, step_size: int) -> tuple[np.ndarray, np.ndarray]:
    from skimage.measure import marching_cubes

    if not np.asarray(mask, dtype=bool).any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode="constant")
    vertices_zyx, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        step_size=int(step_size),
        allow_degenerate=False,
    )
    vertices_zyx -= 1.0
    vertices_xyz = geometry.index_zyx_to_patient_xyz(vertices_zyx)
    return vertices_xyz.astype(np.float32), faces.astype(np.int32)


def _add_mesh_trace(
        figure,
        vertices_xyz: np.ndarray,
        faces: np.ndarray,
        *,
        name: str,
        color: str,
        opacity: float,
) -> None:
    import plotly.graph_objects as go

    vertices = np.asarray(vertices_xyz)
    triangles = np.asarray(faces)
    if len(vertices) == 0 or len(triangles) == 0:
        return
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            name=name,
            color=color,
            opacity=float(opacity),
            flatshading=False,
        )
    )


def show_sparse_joint_anatomy_3d(
        phase_dir: Path,
        predicted_mesh: TriangleMesh,
) -> Any:
    """Interactive body/lung/bone context with reference and MedGS tumor meshes."""

    import plotly.graph_objects as go
    from scipy import ndimage as ndi

    phase_dir = Path(phase_dir)
    ct_hu = np.load(phase_dir / "ct_vis_hu.npy", allow_pickle=False)
    geometry = load_geometry(phase_dir / "ct_vis_geometry.json")
    reference_mesh = load_mesh_npz(phase_dir / "mesh_raw.npz")

    body_mask = _build_body_mask(ct_hu)
    lung_mask = _keep_largest_components((ct_hu < -500.0) & body_mask, count=3, min_size=150)
    lung_mask = ndi.binary_closing(lung_mask, structure=np.ones((3, 3, 3), dtype=bool))
    bone_mask = _remove_small_components((ct_hu > 200.0) & body_mask, min_size=8)

    body_vertices, body_faces = _mask_to_surface(body_mask, geometry, step_size=4)
    lung_vertices, lung_faces = _mask_to_surface(lung_mask, geometry, step_size=2)
    bone_vertices, bone_faces = _mask_to_surface(bone_mask, geometry, step_size=2)

    figure = go.Figure()
    _add_mesh_trace(figure, body_vertices, body_faces, name="Body", color="lightgray", opacity=0.07)
    _add_mesh_trace(figure, lung_vertices, lung_faces, name="Lungs", color="lightskyblue", opacity=0.17)
    _add_mesh_trace(figure, bone_vertices, bone_faces, name="Bones", color="wheat", opacity=0.22)
    _add_mesh_trace(
        figure,
        reference_mesh.vertices_xyz,
        reference_mesh.faces,
        name="RTSTRUCT tumor",
        color=REFERENCE_COLOR,
        opacity=0.78,
    )
    _add_mesh_trace(
        figure,
        predicted_mesh.vertices_xyz,
        predicted_mesh.faces,
        name="MedGS tumor",
        color=PREDICTION_COLOR,
        opacity=0.62,
    )

    figure.update_layout(
        title="Sparse-joint MedGS tumor mesh in anatomical context",
        scene=dict(
            xaxis_title="Patient x [mm]",
            yaxis_title="Patient y [mm]",
            zaxis_title="Patient z [mm]",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.55, y=-1.35, z=0.95),
                up=dict(x=0.0, y=0.0, z=1.0),
                center=dict(x=0.0, y=0.0, z=0.0),
            ),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1000,
        height=820,
    )
    figure.show()
    return figure
