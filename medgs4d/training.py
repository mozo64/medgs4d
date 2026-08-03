from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import math
import time

import numpy as np
import pandas as pd
import torch

from .canonical import CanonicalAssets, get_camera_for_slice
from .config import MedGS4DConfig, save_config
from .data import StudyManifest, load_target_tensor
from .deformation import DeformationField
from .runs import RunPaths, find_latest_checkpoint, write_dataframe, write_json
from .splits import create_split_manifest, get_split_rows, save_split_manifest


@dataclass
class TrainingState:
    """Store the current iteration and persisted training histories."""

    iteration: int
    history: pd.DataFrame
    validation_history: pd.DataFrame


def set_deterministic_seed(seed: int) -> None:
    """Seed Python-independent numerical generators used by the training code."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _phase_neighbors(phase: float, phase_grid: list[float]) -> tuple[float, float]:
    ordered = sorted(float(value) for value in phase_grid)
    index = ordered.index(float(phase))
    return ordered[(index - 1) % len(ordered)], ordered[(index + 1) % len(ordered)]


def build_sampling_plan(
    train_rows: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
    canonical_phase: float | None = None,
    phase_jitter_initial_std: float = 0.0,
) -> pd.DataFrame:
    """Create a deterministic balanced phase-and-slice sampling sequence."""

    if train_rows.empty:
        raise ValueError("Training split is empty")
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1000)
    phases = sorted(train_rows["PhasePercent"].astype(float).unique())
    phase_grid = sorted(
        set(phases + ([] if canonical_phase is None else [float(canonical_phase)]))
    )
    rows_by_phase = {
        phase: train_rows.loc[
            np.isclose(train_rows["PhasePercent"].astype(float), phase)
        ].reset_index(drop=True)
        for phase in phases
    }
    phase_order: list[float] = []
    while len(phase_order) < iterations:
        phase_order.extend(rng.permutation(phases).tolist())
    phase_order = phase_order[:iterations]

    shuffled_indices: dict[float, list[int]] = {}
    positions = {phase: 0 for phase in phases}
    records = []
    for iteration, phase in enumerate(phase_order, start=1):
        if phase not in shuffled_indices or positions[phase] >= len(shuffled_indices[phase]):
            shuffled_indices[phase] = rng.permutation(len(rows_by_phase[phase])).tolist()
            positions[phase] = 0
        row_index = shuffled_indices[phase][positions[phase]]
        positions[phase] += 1
        row = rows_by_phase[phase].iloc[row_index]

        left, right = _phase_neighbors(phase, phase_grid)
        neighbor = left if int(rng.integers(0, 2)) == 0 else right
        progress = (iteration - 1) / max(iterations - 1, 1)
        jitter_std = phase_jitter_initial_std * (1.0 - progress)
        jitter = (
            float(noise_rng.normal(0.0, jitter_std)) if jitter_std > 0 else 0.0
        )
        respiratory_time = phase / 100.0
        augmented_time = float(np.remainder(respiratory_time + jitter, 1.0))
        records.append(
            {
                "Iteration": iteration,
                "PhasePercent": phase,
                "SliceIndex": int(row["SliceIndex"]),
                "NeighborPhasePercent": neighbor,
                "RespiratoryTime": respiratory_time,
                "PhaseJitterStd": jitter_std,
                "PhaseJitter": jitter,
                "AugmentedRespiratoryTime": augmented_time,
            }
        )
    return pd.DataFrame(records)


def build_smoothness_indices(
    gaussian_count: int,
    *,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Select a fixed reproducible Gaussian subset for smoothness regularization."""

    rng = np.random.default_rng(seed)
    count = min(int(sample_size), int(gaussian_count))
    return np.sort(rng.choice(gaussian_count, size=count, replace=False).astype(np.int64))


def load_or_create_sampling_plan(
    path: Path,
    train_rows: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
    canonical_phase: float,
    phase_jitter_initial_std: float,
    resume: bool,
    checkpoint_iteration: int = 0,
) -> pd.DataFrame:
    """Create, reuse, trim, or deterministically extend a sampling plan."""

    if not resume:
        plan = build_sampling_plan(
            train_rows,
            iterations=iterations,
            seed=seed,
            canonical_phase=canonical_phase,
            phase_jitter_initial_std=phase_jitter_initial_std,
        )
        write_dataframe(path, plan)
        return plan

    if not path.is_file():
        raise FileNotFoundError(f"Saved sampling plan does not exist: {path}")

    plan = pd.read_csv(path)
    if len(plan) < checkpoint_iteration:
        raise ValueError(
            f"Sampling plan has {len(plan)} rows but checkpoint is at "
            f"iteration {checkpoint_iteration}"
        )
    if iterations < checkpoint_iteration:
        raise ValueError(
            f"Requested target iteration {iterations} is earlier than "
            f"resume checkpoint {checkpoint_iteration}"
        )

    plan = (
        plan.sort_values("Iteration")
        .drop_duplicates("Iteration", keep="last")
        .reset_index(drop=True)
    )

    if len(plan) > iterations:
        plan = plan.iloc[:iterations].copy()

    if len(plan) < iterations:
        existing_count = len(plan)
        additional_count = iterations - existing_count
        continuing_jitter_std = (
            float(plan.iloc[-1]["PhaseJitterStd"])
            if existing_count and "PhaseJitterStd" in plan
            else phase_jitter_initial_std
        )
        extension = build_sampling_plan(
            train_rows,
            iterations=additional_count,
            seed=seed + existing_count + 1,
            canonical_phase=canonical_phase,
            phase_jitter_initial_std=continuing_jitter_std,
        )
        extension = extension.copy()
        extension["Iteration"] = np.arange(
            existing_count + 1,
            iterations + 1,
            dtype=np.int64,
        )
        plan = pd.concat([plan, extension], ignore_index=True)

    write_dataframe(path, plan)
    return plan


def create_optimizer(
    field: DeformationField,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Create the optimizer for deformation-network parameters only."""

    return torch.optim.Adam(field.model.parameters(), lr=learning_rate)


def compute_reconstruction_loss(
    canonical: CanonicalAssets,
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float,
    ssim_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted L1 and SSIM reconstruction loss components."""

    rendered_batch = rendered.unsqueeze(0) if rendered.ndim == 3 else rendered
    target_batch = target.unsqueeze(0) if target.ndim == 3 else target
    l1_value = canonical.runtime.l1_loss(rendered_batch, target_batch)
    ssim_value = canonical.runtime.ssim(rendered_batch, target_batch)
    image_loss = l1_weight * l1_value + ssim_weight * (1.0 - ssim_value)
    mse = torch.mean((rendered_batch - target_batch) ** 2)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return image_loss, {"l1": l1_value, "ssim": ssim_value, "psnr": psnr}


def normalized_deformation(
    field: DeformationField,
    state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Express x, z, and effective m displacement in normalized units."""

    return torch.cat(
        [
            state["delta_xz"] / field.xz_scale,
            state["delta_m"] / field.m_scale,
        ],
        dim=-1,
    )


def compute_magnitude_loss(
    field: DeformationField,
    state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize excessive displacement from canonical Gaussian geometry."""

    return normalized_deformation(field, state).square().mean()


def compute_temporal_smoothness_loss(
    field: DeformationField,
    first_state: Mapping[str, torch.Tensor],
    second_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize abrupt deformation changes between neighboring phases."""

    first = normalized_deformation(field, first_state)
    second = normalized_deformation(field, second_state)
    return (first - second).square().mean()


def train_step(
    canonical: CanonicalAssets,
    field: DeformationField,
    optimizer: torch.optim.Optimizer,
    study: StudyManifest,
    sample: Mapping[str, Any],
    smoothness_indices: torch.Tensor,
    config: MedGS4DConfig,
    *,
    iteration: int,
) -> dict[str, float]:
    """Run one render, backward pass, and optimizer update."""

    phase = float(sample["PhasePercent"])
    slice_index = int(sample["SliceIndex"])
    respiratory_time = float(sample["AugmentedRespiratoryTime"])
    neighbor_time = float(sample["NeighborPhasePercent"]) / 100.0
    target = load_target_tensor(
        study,
        phase,
        slice_index,
        representation=config.target_representation,
        device=str(field.xyz.device),
    )
    camera = get_camera_for_slice(canonical, slice_index)

    optimizer.zero_grad(set_to_none=True)
    gaussian_view, state = field.build_phase_state(
        respiratory_time, use_checkpointing=True
    )
    render_result = canonical.runtime.render(
        camera,
        gaussian_view,
        canonical.pipeline,
        canonical.background,
        train=True,
        iter=iteration,
    )
    rendered = render_result["render"]
    image_loss, metrics = compute_reconstruction_loss(
        canonical,
        rendered,
        target,
        l1_weight=config.training.l1_weight,
        ssim_weight=config.training.ssim_weight,
    )
    magnitude = compute_magnitude_loss(field, state)
    current_subset = field.build_subset_state(respiratory_time, smoothness_indices)
    neighbor_subset = field.build_subset_state(neighbor_time, smoothness_indices)
    smoothness = compute_temporal_smoothness_loss(
        field, current_subset, neighbor_subset
    )
    total = (
        image_loss
        + config.training.magnitude_weight * magnitude
        + config.training.smoothness_weight * smoothness
    )
    total.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            field.model.parameters(), config.training.max_gradient_norm
        ).item()
    )
    optimizer.step()
    return {
        "TotalLoss": float(total.detach().item()),
        "ImageLoss": float(image_loss.detach().item()),
        "L1": float(metrics["l1"].detach().item()),
        "SSIM": float(metrics["ssim"].detach().item()),
        "PSNR": float(metrics["psnr"].detach().item()),
        "DeformationMagnitudeLoss": float(magnitude.detach().item()),
        "TemporalSmoothnessLoss": float(smoothness.detach().item()),
        "GradientNorm": gradient_norm,
    }


def save_checkpoint(
    path: Path,
    *,
    iteration: int,
    field: DeformationField,
    optimizer: torch.optim.Optimizer,
    config: MedGS4DConfig,
) -> None:
    """Save a restartable deformation checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": int(iteration),
        "deformation_mlp_state_dict": field.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "study_name": config.study_name,
            "run_name": config.run_name,
            "canonical_phase": config.canonical_phase,
            "canonical_checkpoint": config.canonical_checkpoint,
            "canonical_checkpoint_iteration": config.canonical_checkpoint_iteration,
            "parameter_count": field.parameter_count,
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    field: DeformationField,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str,
    config: MedGS4DConfig | None = None,
) -> int:
    """Restore a deformation checkpoint and optionally validate its run identity."""

    payload = torch.load(path, map_location=device, weights_only=False)
    checkpoint_config = payload.get("config", {})

    if config is not None:
        expected = {
            "study_name": config.study_name,
            "run_name": config.run_name,
            "canonical_phase": config.canonical_phase,
            "parameter_count": field.parameter_count,
            "canonical_checkpoint": config.canonical_checkpoint,
            "canonical_checkpoint_iteration": config.canonical_checkpoint_iteration,
        }
        mismatches = []
        for key, expected_value in expected.items():
            if key not in checkpoint_config:
                continue
            actual_value = checkpoint_config[key]
            if isinstance(expected_value, float):
                matches = abs(float(actual_value) - expected_value) <= 1e-6
            else:
                matches = actual_value == expected_value
            if not matches:
                mismatches.append(
                    f"{key}: {actual_value!r} != {expected_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Resume checkpoint is incompatible with this run: "
                + "; ".join(mismatches)
            )

    field.model.load_state_dict(
        payload["deformation_mlp_state_dict"],
        strict=True,
    )
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if optimizer is not None and "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if (
        optimizer is not None
        and torch.cuda.is_available()
        and payload.get("cuda_rng_state_all")
    ):
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload["iteration"])


def load_training_history(path: Path) -> pd.DataFrame:
    """Load training history and tolerate an empty pre-training CSV."""

    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def reconcile_training_history(
    history: pd.DataFrame,
    checkpoint_iteration: int,
) -> pd.DataFrame:
    """Trim and deduplicate history so it matches a resumed checkpoint."""

    if history.empty:
        return history
    return (
        history.loc[history["Iteration"].astype(int) <= checkpoint_iteration]
        .sort_values("Iteration")
        .drop_duplicates("Iteration", keep="last")
        .reset_index(drop=True)
    )


def _validation_rows(
    split_manifest: pd.DataFrame,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    validation = get_split_rows(split_manifest, "validation")
    if validation.empty or sample_count == 0:
        return validation.iloc[:0]
    if sample_count >= len(validation):
        return validation
    rng = np.random.default_rng(seed + 2001)
    selected = []
    for _, group in validation.groupby("PhasePercent", sort=True):
        count = max(1, round(sample_count * len(group) / len(validation)))
        indices = rng.choice(len(group), size=min(count, len(group)), replace=False)
        selected.append(group.iloc[np.sort(indices)])
    result = pd.concat(selected, ignore_index=True)
    return result.head(sample_count)


def validate_during_training(
    canonical: CanonicalAssets,
    field: DeformationField,
    study: StudyManifest,
    rows: pd.DataFrame,
    config: MedGS4DConfig,
) -> dict[str, float] | None:
    """Evaluate a fixed compact validation subset without updating parameters."""

    if rows.empty:
        return None
    values = []
    field.model.eval()
    with torch.no_grad():
        for row in rows.itertuples():
            phase = float(row.PhasePercent)
            slice_index = int(row.SliceIndex)
            target = load_target_tensor(
                study,
                phase,
                slice_index,
                representation=config.target_representation,
                device=str(field.xyz.device),
            )
            view, _ = field.build_phase_state(
                phase / 100.0, use_checkpointing=False
            )
            rendered = canonical.runtime.render(
                get_camera_for_slice(canonical, slice_index),
                view,
                canonical.pipeline,
                canonical.background,
            )["render"]
            _, metrics = compute_reconstruction_loss(
                canonical,
                rendered,
                target,
                l1_weight=config.training.l1_weight,
                ssim_weight=config.training.ssim_weight,
            )
            values.append(
                [
                    float(metrics["l1"].item()),
                    float(metrics["ssim"].item()),
                    float(metrics["psnr"].item()),
                ]
            )
    field.model.train()
    array = np.asarray(values)
    return {
        "Samples": len(values),
        "L1": float(array[:, 0].mean()),
        "SSIM": float(array[:, 1].mean()),
        "PSNR": float(array[:, 2].mean()),
    }


def _training_summary(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    rows = []
    for label, frame in [("all", history), *[
        (f"phase_{phase:g}", group)
        for phase, group in history.groupby("PhasePercent", sort=True)
    ]]:
        final = frame.tail(min(25, len(frame)))
        rows.append(
            {
                "Label": label,
                "Iterations": len(frame),
                "MeanTotalLoss": final["TotalLoss"].mean(),
                "MeanL1": final["L1"].mean(),
                "MeanSSIM": final["SSIM"].mean(),
                "MeanPSNR": final["PSNR"].mean(),
            }
        )
    return pd.DataFrame(rows)


def train_medgs4d(
    study: StudyManifest,
    canonical: CanonicalAssets,
    run_paths: RunPaths,
    config: MedGS4DConfig,
    *,
    resume: bool = False,
    resume_checkpoint: Path | None = None,
    final_evaluation: bool = True,
) -> Path:
    """Train or resume one complete phase-conditioned MedGS4D run."""

    set_deterministic_seed(config.training.seed)
    phase_slice_index = pd.read_csv(study.phase_slice_manifest_path)[
        ["PhasePercent", "SliceIndex", "SliceCoordinate"]
    ]

    if resume:
        split_manifest = pd.read_csv(run_paths.split_manifest)
    else:
        split_manifest = create_split_manifest(
            phase_slice_index,
            mode=config.split.mode,
            canonical_phase=config.canonical_phase,
            validation_phases=config.split.validation_phases,
        )
        save_split_manifest(split_manifest, run_paths.split_manifest)

    field = DeformationField(
        canonical,
        config.deformation,
        config.canonical_phase,
        seed=config.training.seed,
    )
    optimizer = create_optimizer(field, config.training.learning_rate)
    run_paths.checkpoints.mkdir(parents=True, exist_ok=True)
    run_paths.evaluation.mkdir(parents=True, exist_ok=True)
    run_paths.visualizations.mkdir(parents=True, exist_ok=True)
    write_json(
        run_paths.root / "deformation_normalization.json",
        field.normalization_dict(),
    )

    if resume:
        checkpoint_path = (
            resume_checkpoint.resolve()
            if resume_checkpoint is not None
            else find_latest_checkpoint(run_paths.checkpoints)
        )
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint is available for resume: {checkpoint_path}"
            )
        start_iteration = load_checkpoint(
            checkpoint_path,
            field,
            optimizer,
            device=str(field.xyz.device),
            config=config,
        )
        if config.training.iterations < start_iteration:
            raise ValueError(
                f"Requested target iteration {config.training.iterations} "
                f"is earlier than resume checkpoint {start_iteration}"
            )

        history = reconcile_training_history(
            load_training_history(run_paths.training_history),
            start_iteration,
        )
        validation_history = reconcile_training_history(
            load_training_history(run_paths.validation_history),
            start_iteration,
        )

        evaluation_history_path = run_paths.evaluation / "history.csv"
        evaluation_history = load_training_history(evaluation_history_path)
        if not evaluation_history.empty:
            iteration_column = (
                "CheckpointIteration"
                if "CheckpointIteration" in evaluation_history.columns
                else "Iteration"
            )
            evaluation_history = (
                evaluation_history.loc[
                    evaluation_history[iteration_column].astype(int)
                    <= start_iteration
                ]
                .sort_values(iteration_column)
                .drop_duplicates(iteration_column, keep="last")
                .reset_index(drop=True)
            )
            write_dataframe(evaluation_history_path, evaluation_history)

        # Make default future resume follow the explicitly selected branch.
        save_checkpoint(
            run_paths.checkpoints / "deformation_latest.pth",
            iteration=start_iteration,
            field=field,
            optimizer=optimizer,
            config=config,
        )

        stale_outputs = (
            run_paths.completion,
            run_paths.report,
            run_paths.report_metrics,
            run_paths.root / "training_history.pdf",
            run_paths.evaluation / "per_slice.csv",
            run_paths.evaluation / "per_phase.csv",
            run_paths.evaluation / "overall.json",
            run_paths.evaluation / "metrics.pdf",
            run_paths.evaluation / "history.pdf",
        )
        for path in stale_outputs:
            path.unlink(missing_ok=True)

        print(
            f"Resuming MedGS4D from iteration {start_iteration}: "
            f"{checkpoint_path}",
            flush=True,
        )
    else:
        start_iteration = 0
        history = pd.DataFrame()
        validation_history = pd.DataFrame()
        initial = run_paths.checkpoints / "deformation_iter_000000.pth"
        save_checkpoint(
            initial,
            iteration=0,
            field=field,
            optimizer=optimizer,
            config=config,
        )
        save_checkpoint(
            run_paths.checkpoints / "deformation_latest.pth",
            iteration=0,
            field=field,
            optimizer=optimizer,
            config=config,
        )

    save_config(config, run_paths.config)

    train_rows = get_split_rows(split_manifest, "train")
    plan = load_or_create_sampling_plan(
        run_paths.sampling_plan,
        train_rows,
        iterations=config.training.iterations,
        seed=config.training.seed,
        canonical_phase=config.canonical_phase,
        phase_jitter_initial_std=config.training.phase_jitter_initial_std,
        resume=resume,
        checkpoint_iteration=start_iteration,
    )

    smoothness_np = (
        np.load(run_paths.smoothness_indices)
        if resume and run_paths.smoothness_indices.is_file()
        else build_smoothness_indices(
            field.xyz.shape[0],
            sample_size=config.training.smoothness_gaussians,
            seed=config.training.seed + 1001,
        )
    )
    if not run_paths.smoothness_indices.is_file():
        np.save(run_paths.smoothness_indices, smoothness_np)
    smoothness_indices = torch.from_numpy(smoothness_np).to(
        device=field.xyz.device,
        dtype=torch.long,
    )
    validation_rows = _validation_rows(
        split_manifest,
        config.training.validation_samples,
        config.training.seed,
    )

    history_rows = history.to_dict("records") if not history.empty else []
    validation_rows_history = (
        validation_history.to_dict("records")
        if not validation_history.empty
        else []
    )
    elapsed_before = (
        float(history_rows[-1]["ElapsedSeconds"]) if history_rows else 0.0
    )
    segment_start = time.perf_counter()
    field.model.train()

    for iteration in range(
        start_iteration + 1,
        config.training.iterations + 1,
    ):
        sample = plan.iloc[iteration - 1].to_dict()
        metrics = train_step(
            canonical,
            field,
            optimizer,
            study,
            sample,
            smoothness_indices,
            config,
            iteration=iteration,
        )
        elapsed = elapsed_before + time.perf_counter() - segment_start
        history_rows.append(
            {
                "Iteration": iteration,
                "PhasePercent": float(sample["PhasePercent"]),
                "SliceIndex": int(sample["SliceIndex"]),
                "NeighborPhasePercent": float(
                    sample["NeighborPhasePercent"]
                ),
                "PhaseJitterStd": float(sample["PhaseJitterStd"]),
                "PhaseJitter": float(sample["PhaseJitter"]),
                "LearningRate": float(optimizer.param_groups[0]["lr"]),
                "ElapsedSeconds": elapsed,
                "PeakGpuMemoryGB": (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                ),
                **metrics,
            }
        )

        if (
            config.training.validate_every > 0
            and iteration % config.training.validate_every == 0
            and not validation_rows.empty
        ):
            result = validate_during_training(
                canonical,
                field,
                study,
                validation_rows,
                config,
            )
            if result is not None:
                validation_rows_history.append(
                    {"Iteration": iteration, **result}
                )

        if (
            iteration % config.training.log_every == 0
            or iteration == config.training.iterations
        ):
            current = pd.DataFrame(history_rows)
            write_dataframe(run_paths.training_history, current)
            if validation_rows_history:
                write_dataframe(
                    run_paths.validation_history,
                    pd.DataFrame(validation_rows_history),
                )
            window = current.tail(config.training.log_every)
            print(
                f"Iteration {iteration:06d}/"
                f"{config.training.iterations:06d} "
                f"L1={window['L1'].mean():.6f} "
                f"SSIM={window['SSIM'].mean():.6f} "
                f"PSNR={window['PSNR'].mean():.3f}",
                flush=True,
            )

        if (
            iteration % config.training.checkpoint_every == 0
            or iteration == config.training.iterations
        ):
            checkpoint = (
                run_paths.checkpoints
                / f"deformation_iter_{iteration:06d}.pth"
            )
            save_checkpoint(
                checkpoint,
                iteration=iteration,
                field=field,
                optimizer=optimizer,
                config=config,
            )
            save_checkpoint(
                run_paths.checkpoints / "deformation_latest.pth",
                iteration=iteration,
                field=field,
                optimizer=optimizer,
                config=config,
            )

    history = pd.DataFrame(history_rows)
    write_dataframe(run_paths.training_history, history)
    if validation_rows_history:
        write_dataframe(
            run_paths.validation_history,
            pd.DataFrame(validation_rows_history),
        )
    summary = _training_summary(history)
    write_dataframe(run_paths.training_summary, summary)

    elapsed_seconds = (
        float(history.iloc[-1]["ElapsedSeconds"])
        if not history.empty
        else 0.0
    )
    write_json(
        run_paths.completion,
        {
            "status": "complete",
            "iteration": config.training.iterations,
            "parameter_count": field.parameter_count,
            "training_samples": len(train_rows),
            "validation_samples": int(
                (split_manifest["Split"] == "validation").sum()
            ),
            "elapsed_seconds": elapsed_seconds,
            "canonical_checkpoint": config.canonical_checkpoint,
            "canonical_checkpoint_iteration": (
                config.canonical_checkpoint_iteration
            ),
        },
    )

    from .reporting import generate_training_history_pdf

    generate_training_history_pdf(run_paths.root)

    if final_evaluation:
        from .evaluation import (
            aggregate_metrics_overall,
            aggregate_metrics_per_phase,
            evaluate_medgs4d_model,
            save_evaluation_results,
        )
        from .reporting import generate_run_report

        per_slice = evaluate_medgs4d_model(
            study,
            canonical,
            field,
            split_manifest,
            split="all",
            target_representation=config.target_representation,
        )
        per_phase = aggregate_metrics_per_phase(per_slice)
        overall = aggregate_metrics_overall(per_slice)
        overall["Checkpoint"] = str(
            run_paths.checkpoints / "deformation_latest.pth"
        )
        overall["CheckpointIteration"] = config.training.iterations
        overall["CanonicalCheckpoint"] = config.canonical_checkpoint
        overall["CanonicalCheckpointIteration"] = (
            config.canonical_checkpoint_iteration
        )
        save_evaluation_results(
            run_paths.evaluation,
            per_slice=per_slice,
            per_phase=per_phase,
            overall=overall,
        )
        generate_run_report(run_paths.root)
    return run_paths.root

