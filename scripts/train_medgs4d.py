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
    """Build the MedGS4D training command-line interface."""

    parser = argparse.ArgumentParser(description="Train a phase-conditioned MedGS4D model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--canonical-model", type=Path, required=True)
    parser.add_argument("--medgs-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--canonical-phase", type=float, required=True)
    parser.add_argument("--target-representation", choices=["raw", "denoised"], default="raw")
    parser.add_argument("--split-mode", choices=["full", "phase-holdout"], default="full")
    parser.add_argument("--validation-phases", type=float, nargs="*", default=[])
    parser.add_argument("--iterations", type=int, default=7_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--spatial-frequencies", type=int, default=4)
    parser.add_argument("--phase-frequencies", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=131_072)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-jitter-std", type=float, default=0.0)
    parser.add_argument("--skip-final-evaluation", action="store_true")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def config_from_args(args: argparse.Namespace, study_name: str):
    """Create the complete saved training configuration from CLI arguments."""

    from medgs4d.config import (
        DeformationConfig,
        MedGS4DConfig,
        SplitConfig,
        TrainingConfig,
    )

    return MedGS4DConfig(
        study_name=study_name,
        run_name=args.run_name,
        data_dir=str(args.data_dir.resolve()),
        canonical_model_dir=str(args.canonical_model.resolve()),
        medgs_repository=str(args.medgs_repo.resolve()),
        canonical_phase=args.canonical_phase,
        split=SplitConfig(
            mode=args.split_mode,
            validation_phases=tuple(args.validation_phases),
        ),
        deformation=DeformationConfig(
            spatial_frequencies=args.spatial_frequencies,
            phase_frequencies=args.phase_frequencies,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            chunk_size=args.chunk_size,
        ),
        training=TrainingConfig(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            validate_every=args.validate_every,
            validation_samples=args.validation_samples,
            seed=args.seed,
            phase_jitter_initial_std=args.phase_jitter_std,
        ),
        target_representation=args.target_representation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Start, resume, or force-restart one named MedGS4D run."""

    args = build_parser().parse_args(argv)
    from medgs4d.canonical import load_frozen_canonical
    from medgs4d.config import (
        load_medgs4d_config,
        validate_medgs4d_config,
    )
    from medgs4d.data import load_study_manifest
    from medgs4d.runs import (
        assert_resume_compatible,
        build_run_paths,
        prepare_output_directory,
    )
    from medgs4d.training import train_medgs4d

    study = load_study_manifest(args.data_dir)
    config = config_from_args(args, study.study_name)
    validate_medgs4d_config(config)
    run_paths = build_run_paths(args.output_root, study.study_name, args.run_name)
    prepare_output_directory(run_paths.root, force=args.force, resume=args.resume)
    if args.resume:
        saved = load_medgs4d_config(run_paths.config)
        assert_resume_compatible(saved, config)
    canonical = load_frozen_canonical(
        args.canonical_model, args.medgs_repo, device=args.device
    )
    if abs(canonical.config.canonical_phase - args.canonical_phase) > 1e-6:
        raise ValueError("Dynamic and canonical model phases do not match")
    output = train_medgs4d(
        study,
        canonical,
        run_paths,
        config,
        resume=args.resume,
        final_evaluation=not args.skip_final_evaluation,
    )
    print(f"MedGS4D run: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
