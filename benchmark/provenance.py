"""Record and verify the exact source files used by an experiment."""
from pathlib import Path
import hashlib
import json
import subprocess

SOURCE_DIRS = ("arguments", "gaussian_renderer", "models", "scene", "utils", "training", "benchmark", "submodules")
SUFFIXES = {".py", ".cu", ".cpp", ".h", ".hpp", ".cuh", ".inl", ".toml", ".cmake"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(repo):
    files = {repo / name for name in ("train.py", "render.py", "evaluate_temporal.py", "requirements.txt", "pyproject.toml", ".gitmodules")}
    for name in SOURCE_DIRS:
        for path in (repo / name).rglob("*"):
            parts = path.relative_to(repo).parts
            if any(p in {".git", "__pycache__", "build", ".pytest_cache"} or p.endswith(".egg-info") for p in parts):
                continue
            if path.is_file() and (path.suffix in SUFFIXES or path.name == "CMakeLists.txt"):
                files.add(path)
    return sorted(p for p in files if p.is_file())


def record_sources(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    submodules = git("submodule", "status", "--recursive")
    if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
        raise RuntimeError("Initialize submodules at their pinned revisions before running")
    record = {
        "commit": git("rev-parse", "HEAD"),
        "submodules": submodules,
        "files": {str(p.relative_to(repo)): digest(p) for p in source_files(repo)},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise RuntimeError("Code changed since this experiment started; choose a new experiment ID")
    else:
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def verify_sources(repo, manifest):
    repo = Path(repo).resolve()
    record = json.loads(Path(manifest).read_text())
    current = {str(p.relative_to(repo)): digest(p) for p in source_files(repo)}
    if record["files"] != current:
        raise RuntimeError("Experiment sources changed; restart with a new experiment ID")
    return record
