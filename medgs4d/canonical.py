from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CanonicalConfig, load_canonical_config, save_config
from .data import (
    StudyManifest,
    grayscale_to_rgb_uint8,
    load_phase_volume,
    window_hu_to_uint8,
)
from .runs import prepare_output_directory, write_dataframe, write_json


@dataclass(frozen=True)
class CanonicalPaths:
    """Collect the standard files produced by one canonical MedGS run."""

    root: Path
    dataset: Path
    model: Path
    config: Path
    frame_manifest: Path
    metadata: Path
    training_history: Path
    training_script: Path


@dataclass
class MedGSRuntime:
    """Hold imported upstream MedGS classes and functions."""

    render: Any
    gaussian_model_factory: Any
    scene_class: Any
    l1_loss: Any
    ssim: Any


@dataclass
class CanonicalAssets:
    """Bundle one frozen canonical MedGS model and renderer state."""

    run_dir: Path
    config: CanonicalConfig
    gaussians: Any
    scene: Any
    cameras: list[Any]
    xyz: torch.Tensor
    xz: torch.Tensor
    m_logits: torch.Tensor
    m: torch.Tensor
    pipeline: Namespace
    background: torch.Tensor
    dataset_args: Namespace
    loaded_iteration: int
    runtime: MedGSRuntime


def build_canonical_paths(
    output_root: Path,
    study_name: str,
    run_name: str,
) -> CanonicalPaths:
    """Build the standard directory layout for one canonical run."""

    root = output_root / study_name / run_name
    return CanonicalPaths(
        root=root,
        dataset=root / "dataset",
        model=root / "model",
        config=root / "config.json",
        frame_manifest=root / "frame_manifest.csv",
        metadata=root / "canonical_run.json",
        training_history=root / "canonical_training_history.csv",
        training_script=root / "tools" / "train_with_history.py",
    )


def add_medgs_repository(medgs_repository: Path) -> None:
    """Add the separately cloned upstream MedGS repository to sys.path."""

    repository = str(medgs_repository.resolve())
    if repository not in sys.path:
        sys.path.insert(0, repository)


def load_medgs_runtime(medgs_repository: Path) -> MedGSRuntime:
    """Import the renderer, scene, Gaussian model, and loss functions from MedGS."""

    add_medgs_repository(medgs_repository)
    from gaussian_renderer import render
    from models import gaussianModel
    from scene import Scene
    from utils.loss_utils import l1_loss, ssim

    return MedGSRuntime(
        render=render,
        gaussian_model_factory=gaussianModel,
        scene_class=Scene,
        l1_loss=l1_loss,
        ssim=ssim,
    )


def build_canonical_dataset(
    study: StudyManifest,
    output_dir: Path,
    *,
    canonical_phase: float,
    representation: str,
) -> pd.DataFrame:
    """Create MedGS original and mirror image folders for one respiratory phase."""

    volume = load_phase_volume(
        study, canonical_phase, representation=representation, mmap_mode="r"
    )
    study_manifest = pd.read_csv(study.phase_slice_manifest_path)
    phase_rows = (
        study_manifest.loc[
            np.isclose(
                study_manifest["PhasePercent"].astype(float), canonical_phase
            )
        ]
        .sort_values("SliceIndex")
        .reset_index(drop=True)
    )
    if phase_rows.empty:
        raise ValueError(f"Canonical phase {canonical_phase:g}% is unavailable")
    if len(phase_rows) != volume.shape[0]:
        raise ValueError("Canonical volume and slice manifest have different lengths")

    original_dir = output_dir / "original"
    mirror_dir = output_dir / "mirror"
    original_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    for frame_index, row in phase_rows.iterrows():
        gray = window_hu_to_uint8(
            np.asarray(volume[frame_index]), study.hu_window
        )
        rgb = grayscale_to_rgb_uint8(gray)
        filename = f"{frame_index:04d}.png"
        original_path = original_dir / filename
        mirror_path = mirror_dir / filename
        Image.fromarray(rgb).save(original_path)
        Image.fromarray(np.fliplr(rgb).copy()).save(mirror_path)
        frame_rows.append(
            {
                "PatientID": study.patient_id,
                "StudyInstanceUID": study.study_instance_uid,
                "StudyName": study.study_name,
                "PhasePercent": float(canonical_phase),
                "FrameIndex": int(frame_index),
                "SliceIndex": int(row["SliceIndex"]),
                "SliceCoordinate": float(row["SliceCoordinate"]),
                "DICOMPath": str(row.get("Path", "")),
                "SourceVolumePath": str(
                    study.volume_path(canonical_phase, representation)
                ),
                "OriginalPNG": str(original_path.resolve()),
                "MirrorPNG": str(mirror_path.resolve()),
            }
        )
    return pd.DataFrame(frame_rows)



def reconcile_canonical_training_history(
    path: Path,
    checkpoint_iteration: int,
) -> None:
    """Trim history to the checkpoint used for a resumed run."""

    if not path.is_file():
        return
    try:
        history = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return
    if history.empty:
        return

    history = (
        history.loc[
            history["Iteration"].astype(int) <= int(checkpoint_iteration)
        ]
        .sort_values("Iteration")
        .drop_duplicates("Iteration", keep="last")
        .reset_index(drop=True)
    )
    write_dataframe(path, history)


def build_medgs_training_history_script(
    medgs_repository: Path,
    destination: Path,
) -> Path:
    """Create a private MedGS train.py that appends canonical metrics to CSV."""

    source_path = medgs_repository / "train.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"MedGS train.py not found: {source_path}")

    source = source_path.read_text(encoding="utf-8")

    import_anchor = "import copy\n"
    if import_anchor not in source:
        raise RuntimeError(
            "Cannot patch MedGS train.py: import anchor was not found"
        )
    source = source.replace(
        import_anchor,
        import_anchor + "import csv\n",
        1,
    )

    helper_anchor = "\n\ntry:\n    from torch.utils.tensorboard import SummaryWriter\n"
    if helper_anchor not in source:
        raise RuntimeError(
            "Cannot patch MedGS train.py: helper anchor was not found"
        )

    helper_code = r"""
_MEDGS_HISTORY_PATH = os.environ.get("MEDGS_TRAINING_HISTORY_PATH")
_MEDGS_HISTORY_LOG_EVERY = int(
    os.environ.get("MEDGS_TRAINING_LOG_EVERY", "100")
)
_MEDGS_HISTORY_COLUMNS = [
    "Iteration",
    "TotalLoss",
    "L1",
    "InterpolationL1",
    "SSIM",
    "SSIMLoss",
    "SigmaLoss",
    "PSNR",
    "EMALoss",
    "EMAPSNR",
    "GaussianCount",
    "IterationTimeMs",
    "ElapsedSeconds",
]


def _history_float(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _append_training_history(row):
    if not _MEDGS_HISTORY_PATH:
        return

    directory = os.path.dirname(_MEDGS_HISTORY_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)

    write_header = (
        not os.path.isfile(_MEDGS_HISTORY_PATH)
        or os.path.getsize(_MEDGS_HISTORY_PATH) == 0
    )
    with open(
        _MEDGS_HISTORY_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=_MEDGS_HISTORY_COLUMNS,
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
"""
    source = source.replace(
        helper_anchor,
        "\n\n" + helper_code + helper_anchor,
        1,
    )

    loop_anchor = """            total_point = gaussians._xyz.shape[0]
            if iteration % 100 == 0:
"""
    if loop_anchor not in source:
        raise RuntimeError(
            "Cannot patch MedGS train.py: canonical logging anchor was not found"
        )

    logging_code = """            total_point = gaussians._xyz.shape[0]
            if (
                _MEDGS_HISTORY_PATH
                and (
                    iteration % _MEDGS_HISTORY_LOG_EVERY == 0
                    or iteration == opt.iterations
                )
            ):
                _append_training_history(
                    {
                        "Iteration": int(iteration),
                        "TotalLoss": _history_float(loss),
                        "L1": _history_float(Ll1),
                        "InterpolationL1": _history_float(Ll1_inter),
                        "SSIM": 1.0 - _history_float(ssim_loss),
                        "SSIMLoss": _history_float(ssim_loss),
                        "SigmaLoss": _history_float(sigma_loss),
                        "PSNR": _history_float(psnr_),
                        "EMALoss": _history_float(ema_loss_for_log),
                        "EMAPSNR": _history_float(ema_psnr_for_log),
                        "GaussianCount": int(total_point),
                        "IterationTimeMs": float(
                            iter_start.elapsed_time(iter_end)
                        ),
                        "ElapsedSeconds": float(time.time() - init_time),
                    }
                )
            if iteration % 100 == 0:
"""
    source = source.replace(loop_anchor, logging_code, 1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination


def run_medgs_training(
    medgs_repository: Path,
    dataset_dir: Path,
    model_dir: Path,
    config: CanonicalConfig,
    *,
    training_script: Path,
    training_history: Path,
    log_every: int,
    start_checkpoint: Path | None = None,
) -> None:
    """Run upstream MedGS with persistent canonical training history."""

    patched_script = build_medgs_training_history_script(
        medgs_repository,
        training_script,
    )
    command = [
        sys.executable,
        str(patched_script),
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--pipeline",
        "img",
        "--iterations",
        str(config.iterations),
        "--save_iterations",
        str(config.iterations),
        "--test_iterations",
        str(config.iterations),
        "--checkpoint_iterations",
        str(config.iterations),
        "--poly_degree",
        str(config.poly_degree),
        "--batch_size",
        str(config.batch_size),
        "--camera",
        config.camera,
    ]
    if start_checkpoint is not None:
        command.extend(["--start_checkpoint", str(start_checkpoint)])

    environment = os.environ.copy()
    environment["MEDGS_TRAINING_HISTORY_PATH"] = str(
        training_history.resolve()
    )
    environment["MEDGS_TRAINING_LOG_EVERY"] = str(int(log_every))
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(medgs_repository.resolve())
        if not existing_pythonpath
        else str(medgs_repository.resolve())
        + os.pathsep
        + existing_pythonpath
    )

    print("Running:", shlex.join(command), flush=True)
    subprocess.run(
        command,
        cwd=medgs_repository,
        env=environment,
        check=True,
    )


def _checkpoint_iteration(path: Path) -> int:
    """Return the iteration encoded in a chkpnt<iteration>.pth filename."""

    match = re.fullmatch(r"chkpnt(\d+)\.pth", path.name)
    if match is None:
        raise ValueError(f"Invalid canonical checkpoint name: {path.name}")
    return int(match.group(1))


def latest_canonical_checkpoint(model_dir: Path) -> tuple[Path, int]:
    """Find the newest canonical checkpoint saved by upstream MedGS."""

    checkpoints = []
    for path in model_dir.glob("chkpnt*.pth"):
        match = re.fullmatch(r"chkpnt(\d+)\.pth", path.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        raise FileNotFoundError(f"No canonical checkpoint found in: {model_dir}")

    iteration, path = max(checkpoints, key=lambda item: item[0])
    return path, iteration


def _validate_resume_config(
    existing: CanonicalConfig,
    requested: CanonicalConfig,
) -> None:
    """Allow only the target iteration to change when resuming a run."""

    fields = (
        "study_name",
        "run_name",
        "canonical_phase",
        "representation",
        "poly_degree",
        "batch_size",
        "camera",
        "seed",
    )
    mismatches = [
        name
        for name in fields
        if getattr(existing, name) != getattr(requested, name)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: {getattr(existing, name)!r} != "
            f"{getattr(requested, name)!r}"
            for name in mismatches
        )
        raise ValueError(
            "Cannot resume with changed canonical configuration: " + details
        )


def train_canonical_model(
    study: StudyManifest,
    medgs_repository: Path,
    output_root: Path,
    config: CanonicalConfig,
    *,
    resume: bool = False,
    force: bool = False,
    log_every: int = 100,
) -> Path:
    """Prepare, train, or resume one static upstream MedGS model."""

    if resume and force:
        raise ValueError("--resume and --force are mutually exclusive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    paths = build_canonical_paths(
        output_root, config.study_name, config.run_name
    )

    start_checkpoint = None
    start_iteration = None

    if resume:
        if not paths.root.is_dir():
            raise FileNotFoundError(
                f"Canonical run does not exist: {paths.root}"
            )
        if not paths.config.is_file():
            raise FileNotFoundError(
                f"Canonical config does not exist: {paths.config}"
            )
        if not paths.frame_manifest.is_file():
            raise FileNotFoundError(
                f"Canonical frame manifest does not exist: "
                f"{paths.frame_manifest}"
            )
        if not (paths.dataset / "original").is_dir():
            raise FileNotFoundError(
                f"Canonical dataset does not exist: {paths.dataset}"
            )

        existing_config = load_canonical_config(paths.config)
        _validate_resume_config(existing_config, config)
        start_checkpoint, start_iteration = latest_canonical_checkpoint(
            paths.model
        )
        if config.iterations <= start_iteration:
            raise ValueError(
                f"Requested target iteration {config.iterations} must exceed "
                f"the latest checkpoint iteration {start_iteration}"
            )
        reconcile_canonical_training_history(
            paths.training_history,
            start_iteration,
        )

        print(
            f"Resuming canonical run from iteration {start_iteration}: "
            f"{start_checkpoint}",
            flush=True,
        )
    else:
        prepare_output_directory(paths.root, force=force, resume=False)
        save_config(config, paths.config)
        frame_manifest = build_canonical_dataset(
            study,
            paths.dataset,
            canonical_phase=config.canonical_phase,
            representation=config.representation,
        )
        write_dataframe(paths.frame_manifest, frame_manifest)

    run_medgs_training(
        medgs_repository,
        paths.dataset,
        paths.model,
        config,
        training_script=paths.training_script,
        training_history=paths.training_history,
        log_every=log_every,
        start_checkpoint=start_checkpoint,
    )

    checkpoint = paths.model / f"chkpnt{config.iterations}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Expected canonical checkpoint was not saved: {checkpoint}"
        )

    # On resume, update the public metadata only after successful training.
    save_config(config, paths.config)
    metadata = {
        "study_dir": str(study.root.resolve()),
        "medgs_repository": str(medgs_repository.resolve()),
        "dataset_dir": str(paths.dataset.resolve()),
        "model_dir": str(paths.model.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "training_history": str(paths.training_history.resolve()),
        "training_script": str(paths.training_script.resolve()),
        "training_log_every": int(log_every),
    }
    if start_checkpoint is not None:
        metadata["resumed_from_checkpoint"] = str(
            start_checkpoint.resolve()
        )
        metadata["resumed_from_iteration"] = int(start_iteration)

    write_json(paths.metadata, metadata)
    return paths.root


def _read_canonical_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "canonical_run.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_canonical(
    canonical_run_dir: Path,
    medgs_repository: Path | None = None,
    *,
    device: str = "cuda",
) -> CanonicalAssets:
    """Load one trained canonical MedGS model and freeze all Gaussian parameters."""

    canonical_run_dir = canonical_run_dir.resolve()
    config = load_canonical_config(canonical_run_dir / "config.json")
    metadata = _read_canonical_metadata(canonical_run_dir)
    repository = (
        medgs_repository.resolve()
        if medgs_repository is not None
        else Path(metadata["medgs_repository"]).resolve()
    )
    runtime = load_medgs_runtime(repository)
    model_dir = canonical_run_dir / "model"
    dataset_dir = canonical_run_dir / "dataset"
    checkpoint_path = model_dir / f"chkpnt{config.iterations}.pth"
    cfg_path = model_dir / "cfg_args"
    frame_manifest = pd.read_csv(canonical_run_dir / "frame_manifest.csv")
    if not checkpoint_path.is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"Incomplete canonical model: {canonical_run_dir}")

    dataset_args = eval(
        cfg_path.read_text(encoding="utf-8"),
        {"Namespace": Namespace, "__builtins__": {}},
    )
    dataset_args.source_path = str(dataset_dir)
    dataset_args.model_path = str(model_dir)
    dataset_args.gs_type = "gs"
    dataset_args.camera = config.camera
    dataset_args.poly_degree = config.poly_degree
    dataset_args.data_device = device

    frame_count = len(list((dataset_dir / "original").glob("*.png")))
    if frame_count != len(frame_manifest):
        raise ValueError("Canonical image count does not match frame_manifest.csv")

    model_state, loaded_iteration = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if int(loaded_iteration) != config.iterations:
        raise ValueError(
            f"Checkpoint iteration {loaded_iteration} does not match config {config.iterations}"
        )

    gaussians = runtime.gaussian_model_factory["gs"](
        dataset_args.sh_degree,
        dataset_args.poly_degree,
        frame_count,
        use_dff=False,
    )
    scene = runtime.scene_class(
        dataset_args,
        gaussians,
        load_iteration=int(loaded_iteration),
        shuffle=False,
    )
    gaussians.restore(model_state, training_args=None, load_optimizer=False)
    cameras = list(scene.getTestCameras())
    if len(cameras) != frame_count:
        raise ValueError("Canonical camera count does not match frame count")

    parameter_names = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "m",
        "sigma",
        "_w1",
        "time_func",
    )
    for name in parameter_names:
        value = getattr(gaussians, name, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)
            value.grad = None
    gaussians.optimizer = None

    xyz = gaussians.get_xyz.detach().clone()
    m_logits = gaussians.m.detach().clone()
    m = gaussians.get_m.detach().clone()
    pipeline = Namespace(
        convert_SHs_python=getattr(dataset_args, "convert_SHs_python", False),
        compute_cov3D_python=getattr(dataset_args, "compute_cov3D_python", False),
        debug=False,
        antialiasing=getattr(dataset_args, "antialiasing", False),
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0]
        if dataset_args.white_background
        else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    return CanonicalAssets(
        run_dir=canonical_run_dir,
        config=config,
        gaussians=gaussians,
        scene=scene,
        cameras=cameras,
        xyz=xyz,
        xz=xyz[:, [0, -1]],
        m_logits=m_logits,
        m=m,
        pipeline=pipeline,
        background=background,
        dataset_args=dataset_args,
        loaded_iteration=int(loaded_iteration),
        runtime=runtime,
    )


def get_camera_for_slice(model: CanonicalAssets, slice_index: int) -> Any:
    """Return the canonical camera corresponding to one slice index."""

    return model.cameras[int(slice_index)]


def render_canonical_slice(
    model: CanonicalAssets,
    slice_index: int,
) -> torch.Tensor:
    """Render one slice using the frozen canonical Gaussian model."""

    result = model.runtime.render(
        get_camera_for_slice(model, slice_index),
        model.gaussians,
        model.pipeline,
        model.background,
    )
    return result["render"]
