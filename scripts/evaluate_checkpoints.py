#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fully evaluate and compare saved MedGS4D checkpoints."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        help="Evaluate only the selected saved iterations.",
    )
    selection.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="Evaluate explicit deformation checkpoint paths.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import pandas as pd
    from medgs4d.evaluation import evaluate_checkpoints

    comparison_path = evaluate_checkpoints(
        args.run_dir,
        checkpoints=args.checkpoints,
        iterations=args.iterations,
        split=args.split,
        force=args.force,
        device=args.device,
    )
    comparison = pd.read_csv(comparison_path)
    print(comparison.to_string(index=False))
    print(f"Checkpoint comparison CSV: {comparison_path}")
    print(
        "Checkpoint comparison PDF: "
        f"{comparison_path.with_suffix('.pdf')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
