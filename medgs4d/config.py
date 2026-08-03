from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeVar
import json


SplitMode = Literal["full", "phase-holdout"]
Representation = Literal["raw", "denoised"]


@dataclass(frozen=True)
class SplitConfig:
    """Define how phase-slice samples are assigned to train and validation."""

    mode: SplitMode = "full"
    validation_phases: tuple[float, ...] = ()


@dataclass(frozen=True)
class DeformationConfig:
    """Configure the phase-conditioned deformation MLP."""

    spatial_frequencies: int = 4
    phase_frequencies: int = 2
    hidden_dim: int = 256
    hidden_layers: int = 4
    chunk_size: int = 131_072


@dataclass(frozen=True)
class TrainingConfig:
    """Configure MedGS4D optimization, losses, logging, and checkpoints."""

    iterations: int = 7_000
    learning_rate: float = 5e-4
    checkpoint_every: int = 250
    log_every: int = 25
    validate_every: int = 250
    validation_samples: int = 20
    seed: int = 42
    max_gradient_norm: float = 10.0
    smoothness_gaussians: int = 65_536
    l1_weight: float = 2.0
    ssim_weight: float = 0.25
    magnitude_weight: float = 1e-4
    smoothness_weight: float = 1e-3
    phase_jitter_initial_std: float = 0.0


@dataclass(frozen=True)
class CanonicalConfig:
    """Configure one static canonical MedGS training run."""

    study_name: str
    run_name: str
    canonical_phase: float
    representation: Representation = "raw"
    iterations: int = 30_000
    poly_degree: int = 2
    batch_size: int = 3
    camera: str = "mirror"
    seed: int = 42


@dataclass(frozen=True)
class MedGS4DConfig:
    """Describe one complete and reproducible MedGS4D training run."""

    study_name: str
    run_name: str
    data_dir: str
    canonical_model_dir: str
    medgs_repository: str
    canonical_phase: float
    split: SplitConfig
    deformation: DeformationConfig
    training: TrainingConfig
    target_representation: Representation = "raw"


T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in names})


def canonical_config_from_dict(data: dict[str, Any]) -> CanonicalConfig:
    """Create a canonical configuration from a JSON-compatible dictionary."""

    return _dataclass_from_dict(CanonicalConfig, data)


def medgs4d_config_from_dict(data: dict[str, Any]) -> MedGS4DConfig:
    """Create a nested MedGS4D configuration from saved JSON data."""

    payload = dict(data)
    split_payload = dict(payload["split"])
    split_payload["validation_phases"] = tuple(
        float(value) for value in split_payload.get("validation_phases", ())
    )
    payload["split"] = _dataclass_from_dict(SplitConfig, split_payload)
    payload["deformation"] = _dataclass_from_dict(
        DeformationConfig, payload["deformation"]
    )
    payload["training"] = _dataclass_from_dict(
        TrainingConfig, payload["training"]
    )
    return _dataclass_from_dict(MedGS4DConfig, payload)


def validate_canonical_config(config: CanonicalConfig) -> None:
    """Validate canonical training parameters and fail on inconsistent values."""

    if config.representation not in {"raw", "denoised"}:
        raise ValueError(f"Unknown representation: {config.representation}")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.poly_degree < 0:
        raise ValueError("poly_degree must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")


def validate_medgs4d_config(config: MedGS4DConfig) -> None:
    """Validate model, split, and training parameters before creating a run."""

    if config.split.mode not in {"full", "phase-holdout"}:
        raise ValueError(f"Unknown split mode: {config.split.mode}")
    if config.split.mode == "full" and config.split.validation_phases:
        raise ValueError("full split cannot define validation phases")
    if config.split.mode == "phase-holdout" and not config.split.validation_phases:
        raise ValueError("phase-holdout requires validation phases")
    if any(
        abs(float(phase) - float(config.canonical_phase)) < 1e-6
        for phase in config.split.validation_phases
    ):
        raise ValueError("canonical phase cannot be a validation phase")
    if config.deformation.spatial_frequencies < 0:
        raise ValueError("spatial_frequencies must be non-negative")
    if config.deformation.phase_frequencies <= 0:
        raise ValueError("phase_frequencies must be positive")
    if config.deformation.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.deformation.hidden_layers <= 0:
        raise ValueError("hidden_layers must be positive")
    if config.deformation.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    training = config.training
    if training.iterations <= 0:
        raise ValueError("iterations must be positive")
    if training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if training.checkpoint_every <= 0 or training.log_every <= 0:
        raise ValueError("checkpoint_every and log_every must be positive")
    if training.validate_every < 0 or training.validation_samples < 0:
        raise ValueError("validation cadence and sample count cannot be negative")
    if training.smoothness_gaussians <= 0:
        raise ValueError("smoothness_gaussians must be positive")
    if training.phase_jitter_initial_std < 0:
        raise ValueError("phase jitter standard deviation cannot be negative")


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a configuration dataclass into JSON-compatible data."""

    return asdict(config)


def save_config(config: Any, path: Path) -> None:
    """Serialize a configuration dataclass to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_canonical_config(path: Path) -> CanonicalConfig:
    """Load a canonical model configuration from JSON."""

    return canonical_config_from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_medgs4d_config(path: Path) -> MedGS4DConfig:
    """Load a MedGS4D run configuration from JSON."""

    return medgs4d_config_from_dict(json.loads(path.read_text(encoding="utf-8")))
