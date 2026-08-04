#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory RTSTRUCT objects and build validated reference meshes."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser(
        "inventory", help="List RTSTRUCT objects, phases, and ROI names."
    )
    inventory.add_argument("--dicom-dir", type=Path, required=True)
    inventory.add_argument("--patient-id", required=True)
    inventory.add_argument("--study-uid", required=True)
    inventory.add_argument("--csv", type=Path)

    inspect_rois = commands.add_parser(
        "inspect-rois",
        help="Describe contour geometry for every ROI in one phase.",
    )
    inspect_rois.add_argument("--dicom-dir", type=Path, required=True)
    inspect_rois.add_argument("--patient-id", required=True)
    inspect_rois.add_argument("--study-uid", required=True)
    inspect_rois.add_argument("--phase", type=float, default=0.0)
    inspect_rois.add_argument("--rtstruct-file", type=Path)
    inspect_rois.add_argument("--csv", type=Path)

    build = commands.add_parser(
        "build", help="Build and validate one ROI mesh for one phase."
    )
    _add_common_build_arguments(build)
    build.add_argument("--phase", type=float, default=0.0)
    build.add_argument("--roi", required=True)
    build.add_argument("--rtstruct-file", type=Path)

    build_series = commands.add_parser(
        "build-series",
        help="Build masks, meshes, and compact CT volumes for multiple phases.",
    )
    _add_common_build_arguments(build_series)
    build_series.add_argument(
        "--phases",
        type=float,
        nargs="+",
        default=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90),
    )
    build_series.add_argument(
        "--roi-template",
        default="Tumor_c{phase:02d}",
        help="Python format string used to derive the ROI name for each phase.",
    )
    return parser


def _add_common_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--study-uid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hu-window", type=float, nargs=2, default=(-1000.0, 400.0))
    parser.add_argument("--roundtrip-dice-min", type=float, default=0.95)
    parser.add_argument(
        "--ct-vis-stride",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(2, 4, 4),
        help="Downsampling stride for the saved notebook visualization CT volume.",
    )
    parser.add_argument("--force", action="store_true")


def run_inventory(args: argparse.Namespace) -> int:
    from medgs4d.rtstruct import inventory_rtstruct_objects

    frame = inventory_rtstruct_objects(
        args.dicom_dir,
        args.patient_id,
        args.study_uid,
    )
    print(frame.to_string(index=False))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"Inventory CSV: {args.csv}")
    return 0


def run_inspect_rois(args: argparse.Namespace) -> int:
    from medgs4d.rtstruct import inspect_phase_rois

    frame = inspect_phase_rois(
        args.dicom_dir,
        args.patient_id,
        args.study_uid,
        args.phase,
        rtstruct_file=args.rtstruct_file,
    )
    print(frame.to_string(index=False))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"ROI inspection CSV: {args.csv}")
    return 0


def run_build(args: argparse.Namespace) -> int:
    from medgs4d.mesh_series import build_rtstruct_mesh_case

    result = build_rtstruct_mesh_case(
        dicom_dir=args.dicom_dir,
        patient_id=args.patient_id,
        study_instance_uid=args.study_uid,
        phase=args.phase,
        roi_name=args.roi,
        output_dir=args.output_dir,
        rtstruct_file=args.rtstruct_file,
        hu_window=args.hu_window,
        roundtrip_dice_min=args.roundtrip_dice_min,
        ct_vis_stride_zyx=args.ct_vis_stride,
        force=args.force,
    )
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    print(f"Case directory: {result['output_dir']}")
    print(f"Preview: {Path(result['output_dir']) / 'validation_overview.png'}")
    print(f"Mesh: {Path(result['output_dir']) / 'mesh_raw.ply'}")
    print(f"Visualization CT: {Path(result['output_dir']) / 'ct_vis_hu.npy'}")
    return 0


def run_build_series(args: argparse.Namespace) -> int:
    from medgs4d.mesh_series import build_rtstruct_mesh_series

    summary = build_rtstruct_mesh_series(
        dicom_dir=args.dicom_dir,
        patient_id=args.patient_id,
        study_instance_uid=args.study_uid,
        phases=args.phases,
        roi_template=args.roi_template,
        output_dir=args.output_dir,
        hu_window=args.hu_window,
        roundtrip_dice_min=args.roundtrip_dice_min,
        ct_vis_stride_zyx=args.ct_vis_stride,
        force=args.force,
    )
    print(summary.to_string(index=False))
    print(f"Series directory: {args.output_dir}")
    print(f"Series manifest: {args.output_dir / 'series_manifest.json'}")
    print(f"Series summary: {args.output_dir / 'series_summary.csv'}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "inventory":
        return run_inventory(args)
    if args.command == "inspect-rois":
        return run_inspect_rois(args)
    if args.command == "build":
        return run_build(args)
    return run_build_series(args)


if __name__ == "__main__":
    raise SystemExit(main())
