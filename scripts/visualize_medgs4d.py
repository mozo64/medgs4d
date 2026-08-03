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
    """Build the static MedGS4D visualization command-line interface."""

    parser = argparse.ArgumentParser(description="Save static visualizations for one run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--phase", type=float)
    parser.add_argument("--all-phases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Save a selected slice comparison or full breathing-cycle figure."""

    args = build_parser().parse_args(argv)
    if args.all_phases == (args.phase is not None):
        raise SystemExit("Choose exactly one of --phase or --all-phases")
    from medgs4d.results import load_run
    from medgs4d.visualization import save_breathing_cycle_grid, save_slice_comparison

    run = load_run(run_dir=args.run_dir)
    if args.all_phases:
        save_breathing_cycle_grid(
            run, args.output, slice_index=args.slice, device=args.device
        )
    else:
        save_slice_comparison(
            run,
            args.output,
            phase=float(args.phase),
            slice_index=args.slice,
            device=args.device,
        )
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
