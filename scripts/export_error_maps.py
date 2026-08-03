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
    """Build the static error-map export command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Export ground-truth, baseline, dynamic reconstruction, "
            "and absolute error-map grids for a completed MedGS4D run."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed or checkpointed MedGS4D run directory.",
    )
    parser.add_argument(
        "--phases",
        type=float,
        nargs="+",
        help="Respiratory phases to export. Defaults to all prepared phases.",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="+",
        help="Slice indices to export. Defaults to the middle slice.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Exact deformation checkpoint. Defaults to the run's latest checkpoint.",
    )
    parser.add_argument(
        "--error-max",
        type=float,
        help=(
            "Fixed upper limit for both absolute-error maps. "
            "By default, each figure uses the joint 99th percentile."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for rendering. Default: cuda.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load one run and export the requested static error-map figures."""

    args = build_parser().parse_args(argv)

    from medgs4d.results import load_run
    from medgs4d.visualization import save_run_error_maps

    run = load_run(run_dir=args.run_dir)
    output_dir = save_run_error_maps(
        run,
        phases=args.phases,
        slice_indices=args.slices,
        checkpoint=args.checkpoint,
        device=args.device,
        error_max=args.error_max,
    )

    files = sorted(output_dir.glob("phase_*_slice_*.png"))
    print(f"Error-map directory: {output_dir}")
    print(f"PNG files currently present: {len(files)}")
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
