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
    """Build the canonical MedGS training command-line interface."""

    parser = argparse.ArgumentParser(description="Train a static canonical MedGS model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--medgs-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--canonical-phase", type=float, required=True)
    parser.add_argument("--representation", choices=["raw", "denoised"], default="raw")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--camera", default="mirror")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Train one named canonical model without overwriting existing results."""

    args = build_parser().parse_args(argv)
    from medgs4d.canonical import train_canonical_model
    from medgs4d.config import CanonicalConfig, validate_canonical_config
    from medgs4d.data import load_study_manifest

    study = load_study_manifest(args.data_dir)
    config = CanonicalConfig(
        study_name=study.study_name,
        run_name=args.run_name,
        canonical_phase=args.canonical_phase,
        representation=args.representation,
        iterations=args.iterations,
        poly_degree=args.poly_degree,
        batch_size=args.batch_size,
        camera=args.camera,
        seed=args.seed,
    )
    validate_canonical_config(config)
    run_dir = train_canonical_model(
        study,
        args.medgs_repo,
        args.output_root,
        config,
        force=args.force,
    )
    print(f"Canonical run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
