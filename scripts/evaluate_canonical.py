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
    """Build the canonical-model evaluation command-line interface."""

    parser = argparse.ArgumentParser(description="Evaluate a canonical MedGS model.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medgs-repo", type=Path, required=True)
    parser.add_argument("--target-representation", choices=["raw", "denoised"], default="raw")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one canonical model and save reconstruction metrics."""

    args = build_parser().parse_args(argv)
    from medgs4d.evaluation import save_canonical_evaluation

    output = save_canonical_evaluation(
        args.run_dir,
        args.medgs_repo,
        target_representation=args.target_representation,
        device=args.device,
    )
    print(f"Evaluation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
