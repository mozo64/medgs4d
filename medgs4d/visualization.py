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
