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
    """Build the data-management command-line interface."""

    parser = argparse.ArgumentParser(description="Inspect and prepare 4D-CT studies.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_patients = commands.add_parser("list-patients", help="List available patients.")
    source = list_patients.add_mutually_exclusive_group(required=True)
    source.add_argument("--archives-dir", type=Path)
    source.add_argument("--dicom-dir", type=Path)

    extract = commands.add_parser("extract", help="Extract one patient's series archives.")
    extract.add_argument("--archives-dir", type=Path, required=True)
    extract.add_argument("--dicom-dir", type=Path, required=True)
    extract.add_argument("--patient-id", required=True)
    extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--dry-run", action="store_true")
    extract.add_argument("--log", type=Path)

    studies = commands.add_parser("list-studies", help="List studies for one patient.")
    studies.add_argument("--dicom-dir", type=Path, required=True)
    studies.add_argument("--patient-id", required=True)

    series = commands.add_parser(
        "list-series",
        help="List complete DICOM series metadata without CT phase selection.",
    )
    series.add_argument("--dicom-dir", type=Path, required=True)
    series.add_argument("--patient-id", required=True)
    series.add_argument("--study-uid")
    series.add_argument("--modality")

    rtstructs = commands.add_parser(
        "list-rtstructs",
        help="List RTSTRUCT objects, ROI names, and referenced CT series.",
    )
    rtstructs.add_argument("--dicom-dir", type=Path, required=True)
    rtstructs.add_argument("--patient-id", required=True)
    rtstructs.add_argument("--referenced-study-uid")

    inspect = commands.add_parser("inspect", help="Inspect respiratory series in one study.")
    inspect.add_argument("--dicom-dir", type=Path, required=True)
    inspect.add_argument("--patient-id", required=True)
    inspect.add_argument("--study-uid", required=True)

    prepare = commands.add_parser("prepare", help="Prepare reusable 4D-CT volumes.")
    prepare.add_argument("--dicom-dir", type=Path, required=True)
    prepare.add_argument("--prepared-root", type=Path, required=True)
    prepare.add_argument("--patient-id", required=True)
    prepare.add_argument("--study-uid", required=True)
    prepare.add_argument("--study-name", required=True)
    prepare.add_argument("--hu-window", type=float, nargs=2, default=(-1000.0, 400.0))
    prepare.add_argument(
        "--denoise-sigma", type=float, nargs=3, default=(0.20, 0.40, 0.40)
    )
    prepare.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the selected data-management subcommand."""

    args = build_parser().parse_args(argv)
    from medgs4d.data import (
        extract_patient_archives,
        inspect_study,
        list_archive_patients,
        list_extracted_patients,
        list_patient_series,
        list_patient_studies,
        list_rtstructs,
        prepare_study,
    )

    if args.command == "list-patients":
        frame = (
            list_archive_patients(args.archives_dir)
            if args.archives_dir is not None
            else list_extracted_patients(args.dicom_dir)
        )
        print(frame.to_string(index=False))
    elif args.command == "extract":
        frame = extract_patient_archives(
            args.archives_dir,
            args.dicom_dir,
            args.patient_id,
            workers=args.workers,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(frame["Status"].value_counts().to_string())
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.log, index=False)
            print(f"Log: {args.log}")
    elif args.command == "list-studies":
        print(list_patient_studies(args.dicom_dir, args.patient_id).to_string(index=False))
    elif args.command == "list-series":
        frame = list_patient_series(
            args.dicom_dir,
            args.patient_id,
            study_instance_uid=args.study_uid,
            modality=args.modality,
        )
        print(frame.to_string(index=False))
    elif args.command == "list-rtstructs":
        frame = list_rtstructs(
            args.dicom_dir,
            args.patient_id,
            referenced_study_uid=args.referenced_study_uid,
        )
        print(frame.to_string(index=False))
    elif args.command == "inspect":
        print(
            inspect_study(args.dicom_dir, args.patient_id, args.study_uid).to_string(
                index=False
            )
        )
    elif args.command == "prepare":
        manifest = prepare_study(
            args.dicom_dir,
            args.prepared_root,
            patient_id=args.patient_id,
            study_instance_uid=args.study_uid,
            study_name=args.study_name,
            hu_window=tuple(args.hu_window),
            denoise_sigma=tuple(args.denoise_sigma),
            force=args.force,
        )
        print(f"Prepared study: {manifest.root}")
        print(f"Phases: {manifest.phases}")
        print(f"Volume shape: {manifest.volume_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
