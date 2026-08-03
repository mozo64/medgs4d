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
        description="Measure MedGS4D deformation magnitude and temporal continuity."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--phases",
        type=float,
        nargs="*",
        help="Defaults to every phase in the prepared study.",
    )
    parser.add_argument(
        "--sample-gaussians",
        type=int,
        default=0,
        help="0 evaluates all Gaussians.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from medgs4d.diagnostics import diagnose_run_deformation

    output_dir = diagnose_run_deformation(
        args.run_dir,
        checkpoint=args.checkpoint,
        phases=args.phases,
        sample_gaussians=args.sample_gaussians,
        seed=args.seed,
        device=args.device,
    )
    print(f"Deformation diagnostics: {output_dir}")
    print(f"Per-phase CSV: {output_dir / 'per_phase.csv'}")
    print(f"Temporal CSV: {output_dir / 'temporal_steps.csv'}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Diagnostics PDF: {output_dir / 'diagnostics.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
