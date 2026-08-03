from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .canonical import CanonicalAssets, get_camera_for_slice
from .data import load_phase_volume, window_hu_to_uint8
from .deformation import DeformationField
from .results import RunResults, load_run_models


def _to_gray(image) -> np.ndarray:
    array = image.detach().clamp(0, 1).mean(dim=0).cpu().numpy()
    return array


def render_run_slice(
    run: RunResults,
    canonical: CanonicalAssets,
    field: DeformationField,
    *,
    phase: float,
    slice_index: int,
) -> dict[str, np.ndarray]:
    """Render ground truth, canonical baseline, and dynamic reconstruction."""

    import torch

    volume = load_phase_volume(
        run.study,
        phase,
        representation=run.config.target_representation,
    )
    ground_truth = window_hu_to_uint8(
        np.asarray(volume[int(slice_index)]), run.study.hu_window
    )
    camera = get_camera_for_slice(canonical, slice_index)
    with torch.no_grad():
        baseline = canonical.runtime.render(
            camera,
            canonical.gaussians,
            canonical.pipeline,
            canonical.background,
        )["render"]
        view, _ = field.build_phase_state(
            float(phase) / 100.0, use_checkpointing=False
        )
        dynamic = canonical.runtime.render(
            camera,
            view,
            canonical.pipeline,
            canonical.background,
        )["render"]
    return {
        "ground_truth": ground_truth,
        "baseline": _to_gray(baseline),
        "dynamic": _to_gray(dynamic),
    }


def show_training_curves(run: RunResults) -> Any:
    """Display L1, PSNR, and SSIM training and validation curves."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for axis, metric in zip(axes, ["L1", "PSNR", "SSIM"]):
        history = run.training_history
        rolling = history.set_index("Iteration")[metric].rolling(50, min_periods=1).mean()
        axis.plot(rolling.index, rolling.values, label="train")
        if run.validation_history is not None and metric in run.validation_history:
            axis.plot(
                run.validation_history["Iteration"],
                run.validation_history[metric],
                marker="o",
                label="validation",
            )
            axis.legend(frameon=False)
        axis.set_xlabel("Iteration")
        axis.set_title(metric)
    return figure


def show_slice_browser(
    run: RunResults,
    *,
    device: str = "cuda",
) -> Any:
    """Create an interactive phase-and-slice reconstruction browser."""

    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    canonical, field = load_run_models(run, device=device)
    phase_widget = widgets.SelectionSlider(
        options=[float(value) for value in run.study.phases],
        value=float(run.study.phases[0]),
        description="Phase",
        continuous_update=False,
    )
    slice_widget = widgets.IntSlider(
        min=0,
        max=run.study.slice_count - 1,
        value=run.study.slice_count // 2,
        description="Slice",
        continuous_update=False,
    )
    output = widgets.Output()

    def update(*_: object) -> None:
        images = render_run_slice(
            run,
            canonical,
            field,
            phase=float(phase_widget.value),
            slice_index=int(slice_widget.value),
        )
        with output:
            output.clear_output(wait=True)
            figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            for axis, (name, image) in zip(axes, images.items()):
                axis.imshow(image, cmap="gray", vmin=0, vmax=255 if name == "ground_truth" else 1)
                axis.set_title(name.replace("_", " ").title())
                axis.axis("off")
            plt.show()

    phase_widget.observe(update, names="value")
    slice_widget.observe(update, names="value")
    update()
    box = widgets.VBox([widgets.HBox([phase_widget, slice_widget]), output])
    display(box)
    return box


def show_breathing_cycle_browser(
    run: RunResults,
    *,
    device: str = "cuda",
) -> Any:
    """Create an interactive full-cycle browser for one selected slice."""

    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    canonical, field = load_run_models(run, device=device)
    slice_widget = widgets.IntSlider(
        min=0,
        max=run.study.slice_count - 1,
        value=run.study.slice_count // 2,
        description="Slice",
        continuous_update=False,
    )
    output = widgets.Output()

    def update(*_: object) -> None:
        slice_index = int(slice_widget.value)
        phase_images = [
            (
                phase,
                render_run_slice(
                    run,
                    canonical,
                    field,
                    phase=phase,
                    slice_index=slice_index,
                ),
            )
            for phase in run.study.phases
        ]
        with output:
            output.clear_output(wait=True)
            figure, axes = plt.subplots(
                2,
                len(phase_images),
                figsize=(2.1 * len(phase_images), 4.5),
                constrained_layout=True,
            )
            for column, (phase, images) in enumerate(phase_images):
                axes[0, column].imshow(images["ground_truth"], cmap="gray", vmin=0, vmax=255)
                axes[1, column].imshow(images["dynamic"], cmap="gray", vmin=0, vmax=1)
                axes[0, column].set_title(f"{phase:g}%")
                axes[0, column].axis("off")
                axes[1, column].axis("off")
            axes[0, 0].set_ylabel("Ground truth")
            axes[1, 0].set_ylabel("MedGS4D")
            plt.show()

    slice_widget.observe(update, names="value")
    update()
    box = widgets.VBox([slice_widget, output])
    display(box)
    return box


def save_slice_comparison(
    run: RunResults,
    output_path: Path,
    *,
    phase: float,
    slice_index: int,
    device: str = "cuda",
) -> None:
    """Save a static ground-truth, baseline, and dynamic comparison figure."""

    import matplotlib.pyplot as plt

    canonical, field = load_run_models(run, device=device)
    images = render_run_slice(
        run, canonical, field, phase=phase, slice_index=slice_index
    )
    figure, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
    for axis, (name, image) in zip(axes, images.items()):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255 if name == "ground_truth" else 1)
        axis.set_title(name.replace("_", " ").title())
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_breathing_cycle_grid(
    run: RunResults,
    output_path: Path,
    *,
    slice_index: int,
    device: str = "cuda",
) -> None:
    """Save all respiratory phases and dynamic reconstructions for one slice."""

    import matplotlib.pyplot as plt

    canonical, field = load_run_models(run, device=device)
    images = [
        (
            phase,
            render_run_slice(
                run,
                canonical,
                field,
                phase=phase,
                slice_index=slice_index,
            ),
        )
        for phase in run.study.phases
    ]
    figure, axes = plt.subplots(
        2,
        len(images),
        figsize=(2.0 * len(images), 4.4),
        constrained_layout=True,
    )
    for column, (phase, item) in enumerate(images):
        axes[0, column].imshow(item["ground_truth"], cmap="gray", vmin=0, vmax=255)
        axes[1, column].imshow(item["dynamic"], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"{phase:g}%")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def render_run_error_maps(
    run: RunResults,
    canonical: CanonicalAssets,
    field: DeformationField,
    *,
    phase: float,
    slice_index: int,
) -> dict[str, np.ndarray]:
    """Render reconstructions and directly comparable absolute error maps."""

    images = render_run_slice(
        run,
        canonical,
        field,
        phase=phase,
        slice_index=slice_index,
    )
    ground_truth = images["ground_truth"].astype(np.float32) / 255.0
    baseline = images["baseline"].astype(np.float32)
    dynamic = images["dynamic"].astype(np.float32)
    baseline_error = np.abs(ground_truth - baseline)
    dynamic_error = np.abs(ground_truth - dynamic)

    return {
        "ground_truth": ground_truth,
        "baseline": baseline,
        "dynamic": dynamic,
        "baseline_error": baseline_error,
        "dynamic_error": dynamic_error,
        "error_reduction": baseline_error - dynamic_error,
    }


def _error_map_figure(
    images: dict[str, np.ndarray],
    *,
    phase: float,
    slice_index: int,
    error_max: float | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    if error_max is None:
        error_values = np.concatenate(
            [
                images["baseline_error"].ravel(),
                images["dynamic_error"].ravel(),
            ]
        )
        error_max = float(np.quantile(error_values, 0.99))
    error_max = max(float(error_max), 1e-6)

    reduction = images["error_reduction"]
    reduction_max = max(
        float(np.quantile(np.abs(reduction), 0.99)),
        1e-6,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(11.2, 7.0),
        constrained_layout=True,
    )
    image_specs = (
        ("ground_truth", "Ground truth"),
        ("baseline", "Canonical baseline"),
        ("dynamic", "Dynamic reconstruction"),
    )
    for axis, (key, title) in zip(axes[0], image_specs):
        axis.imshow(images[key], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    baseline_map = axes[1, 0].imshow(
        images["baseline_error"],
        cmap="magma",
        vmin=0.0,
        vmax=error_max,
    )
    axes[1, 0].set_title("|GT − baseline|")
    axes[1, 0].axis("off")
    figure.colorbar(
        baseline_map,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )

    dynamic_map = axes[1, 1].imshow(
        images["dynamic_error"],
        cmap="magma",
        vmin=0.0,
        vmax=error_max,
    )
    axes[1, 1].set_title("|GT − dynamic|")
    axes[1, 1].axis("off")
    figure.colorbar(
        dynamic_map,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
    )

    reduction_map = axes[1, 2].imshow(
        reduction,
        cmap="RdBu",
        vmin=-reduction_max,
        vmax=reduction_max,
    )
    axes[1, 2].set_title("Error reduction\npositive = dynamic better")
    axes[1, 2].axis("off")
    figure.colorbar(
        reduction_map,
        ax=axes[1, 2],
        fraction=0.046,
        pad=0.04,
    )

    figure.suptitle(
        f"Phase {phase:g}% | slice {slice_index}",
        fontsize=12,
    )
    return figure


def show_error_map_browser(
    run: RunResults,
    *,
    checkpoint: Path | None = None,
    device: str = "cuda",
    error_max: float | None = None,
) -> Any:
    """Create an interactive browser with reconstruction and error maps."""

    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    canonical, field = load_run_models(
        run,
        checkpoint=checkpoint,
        device=device,
    )
    phase_widget = widgets.SelectionSlider(
        options=[float(value) for value in run.study.phases],
        value=float(run.study.phases[0]),
        description="Phase",
        continuous_update=False,
    )
    slice_widget = widgets.IntSlider(
        min=0,
        max=run.study.slice_count - 1,
        value=run.study.slice_count // 2,
        description="Slice",
        continuous_update=False,
    )
    output = widgets.Output()

    def update(*_: object) -> None:
        phase = float(phase_widget.value)
        slice_index = int(slice_widget.value)
        images = render_run_error_maps(
            run,
            canonical,
            field,
            phase=phase,
            slice_index=slice_index,
        )
        with output:
            output.clear_output(wait=True)
            figure = _error_map_figure(
                images,
                phase=phase,
                slice_index=slice_index,
                error_max=error_max,
            )
            plt.show()

    phase_widget.observe(update, names="value")
    slice_widget.observe(update, names="value")
    update()
    box = widgets.VBox(
        [widgets.HBox([phase_widget, slice_widget]), output]
    )
    display(box)
    return box


def save_error_map_grid(
    run: RunResults,
    output_path: Path,
    *,
    phase: float,
    slice_index: int,
    checkpoint: Path | None = None,
    device: str = "cuda",
    error_max: float | None = None,
) -> Path:
    """Save one ground-truth, reconstruction, and error-map grid."""

    import matplotlib.pyplot as plt

    canonical, field = load_run_models(
        run,
        checkpoint=checkpoint,
        device=device,
    )
    images = render_run_error_maps(
        run,
        canonical,
        field,
        phase=phase,
        slice_index=slice_index,
    )
    figure = _error_map_figure(
        images,
        phase=phase,
        slice_index=slice_index,
        error_max=error_max,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_run_error_maps(
    run: RunResults,
    *,
    phases: list[float] | None = None,
    slice_indices: list[int] | None = None,
    checkpoint: Path | None = None,
    device: str = "cuda",
    error_max: float | None = None,
) -> Path:
    """Save selected error-map grids under visualizations/error_maps."""

    import matplotlib.pyplot as plt

    selected_phases = (
        [float(value) for value in run.study.phases]
        if phases is None
        else [float(value) for value in phases]
    )
    selected_slices = (
        [run.study.slice_count // 2]
        if slice_indices is None
        else [int(value) for value in slice_indices]
    )
    output_dir = run.run_dir / "visualizations" / "error_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical, field = load_run_models(
        run,
        checkpoint=checkpoint,
        device=device,
    )
    for phase in selected_phases:
        phase_label = f"{phase:g}".replace(".", "p")
        for slice_index in selected_slices:
            images = render_run_error_maps(
                run,
                canonical,
                field,
                phase=phase,
                slice_index=slice_index,
            )
            figure = _error_map_figure(
                images,
                phase=phase,
                slice_index=slice_index,
                error_max=error_max,
            )
            path = (
                output_dir
                / f"phase_{phase_label}_slice_{slice_index:03d}.png"
            )
            figure.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(figure)

    return output_dir
