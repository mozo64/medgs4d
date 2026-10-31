"""Validate requirements for execution, without pinning a particular machine."""
from pathlib import Path
import os
import torch


def validate_environment(repo_root, gpu_index, metrics_module):
    repo = Path(repo_root).expanduser().resolve()
    if not (repo / "benchmark/worker.py").is_file():
        raise FileNotFoundError(f"REPO_ROOT does not contain benchmark/worker.py: {repo}")
    # Kernel imports and subprocess workers must use the same implementation.
    actual = Path(metrics_module.__file__).resolve()
    expected = repo / "benchmark/metrics.py"
    if actual != expected:
        raise RuntimeError(
            f"Kernel imported {actual}, but REPO_ROOT points to {repo}. "
            "Install that checkout into this kernel's environment with "
            "python -m pip install --no-deps -e . and restart the kernel."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MedGS renderer; select a CUDA-capable kernel.")
    try:
        index = int(gpu_index)
    except (ValueError, TypeError) as exc:
        raise ValueError("GPU_INDEX must be an integer device index") from exc
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(f"GPU_INDEX={index}, but this kernel sees {count} CUDA devices")
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility is None:
        return str(index)
    devices = [item.strip() for item in visibility.split(",")]
    if index >= len(devices) or not devices[index]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the initialized kernel; restart it")
    return devices[index]
