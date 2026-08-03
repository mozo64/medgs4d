#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the MedGS4D evaluation command-line interface."""

    parser = argparse.ArgumentParser(description="Evaluate one MedGS4D run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["all", "train", "validation"], default="all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a selected run on train, validation, or all samples."""

    args = build_parser().parse_args(argv)
    from medgs4d.evaluation import evaluate_run

    output = evaluate_run(
        args.run_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        force=args.force,
        device=args.device,
    )
    print(f"Evaluation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
