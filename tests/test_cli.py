from pathlib import Path
import os
import subprocess
import sys

import pytest


SCRIPTS = [
    "data.py",
    "train_canonical.py",
    "evaluate_canonical.py",
    "train_medgs4d.py",
    "evaluate_medgs4d.py",
    "visualize_medgs4d.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
