#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory RTSTRUCT objects and validate one ROI as a 3D mesh."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser(
        "inventory", help="List RTSTRUCT objects, phases, and ROI names."
    )
    inventory.add_argument("--dicom-dir", type=Path, required=True)
    inventory.add_argument("--patient-id", required=True)
    inventory.add_argument("--study-uid", required=True)
    inventory.add_argument("--csv", type=Path)

    build = commands.add_parser(
        "build", help="Rasterize one ROI, build one mesh, and validate round-trip geometry."
    )
    build.add_argument("--dicom-dir", type=Path, required=True)
    build.add_argument("--patient-id", required=True)
    build.add_argument("--study-uid", required=True)
    build.add_argument("--phase", type=float, default=0.0)
    build.add_argument("--roi", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--rtstruct-file", type=Path)
    build.add_argument("--hu-window", type=float, nargs=2, default=(-1000.0, 400.0))
    build.add_argument("--roundtrip-dice-min", type=float, default=0.95)
    build.add_argument("--force", action="store_true")
    return parser


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


def run_build(args: argparse.Namespace) -> int:
    from medgs4d.mesh_validation import save_validation_overview
    from medgs4d.meshes import (
        mask_to_mesh,
        mesh_to_mask_roundtrip,
        mesh_validation_report,
        save_mesh_npz,
        save_report,
        write_ply,
    )
    from medgs4d.rtstruct import (
        contours_to_json,
        find_rtstruct_and_ct_series,
        load_ct_geometry,
        load_ct_volume,
        rasterize_roi,
        read_roi_contours,
        save_geometry,
    )

    output_dir = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\nUse --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rtstruct_path, ct_series = find_rtstruct_and_ct_series(
        args.dicom_dir,
        args.patient_id,
        args.study_uid,
        args.phase,
        roi_name=args.roi,
        rtstruct_file=args.rtstruct_file,
    )
    geometry = load_ct_geometry(
        Path(ct_series["SeriesPath"]),
        str(ct_series["SeriesInstanceUID"]),
    )
    _, contours = read_roi_contours(rtstruct_path, args.roi)
    rasterized = rasterize_roi(contours, geometry)
    mesh = mask_to_mesh(rasterized.mask, geometry)
    roundtrip_mask = mesh_to_mask_roundtrip(
        mesh.vertices_zyx,
        mesh.faces,
        geometry.shape_zyx,
    )
    report = mesh_validation_report(
        rasterized.mask,
        roundtrip_mask,
        mesh,
        geometry,
    )
    report.update(
        {
            "patient_id": args.patient_id,
            "study_instance_uid": args.study_uid,
            "phase_percent": args.phase,
            "roi_name": args.roi,
            "rtstruct_file": str(rtstruct_path),
            "ct_series_instance_uid": str(ct_series["SeriesInstanceUID"]),
            "ct_series_path": str(ct_series["SeriesPath"]),
            "contour_count": len(contours),
            "roundtrip_dice_minimum": args.roundtrip_dice_min,
        }
    )

    np.save(output_dir / "mask.npy", rasterized.mask)
    np.save(output_dir / "roundtrip_mask.npy", roundtrip_mask)
    save_geometry(output_dir / "geometry.json", geometry)
    rasterized.contour_table.to_csv(output_dir / "contour_report.csv", index=False)
    (output_dir / "contours.json").write_text(
        json.dumps(contours_to_json(contours, geometry), indent=2),
        encoding="utf-8",
    )
    save_mesh_npz(output_dir / "mesh_raw.npz", mesh)
    write_ply(output_dir / "mesh_raw.ply", mesh.vertices_xyz, mesh.faces)
    save_report(output_dir / "validation_report.json", report)

    ct_volume = load_ct_volume(geometry)
    save_validation_overview(
        output_dir / "validation_overview.png",
        ct_volume,
        rasterized.mask,
        roundtrip_mask,
        contours,
        geometry,
        tuple(args.hu_window),
    )

    manifest = {
        "patient_id": args.patient_id,
        "study_instance_uid": args.study_uid,
        "phase_percent": args.phase,
        "roi_name": args.roi,
        "rtstruct_file": str(rtstruct_path),
        "ct_series_instance_uid": str(ct_series["SeriesInstanceUID"]),
        "geometry_file": "geometry.json",
        "mask_file": "mask.npy",
        "roundtrip_mask_file": "roundtrip_mask.npy",
        "contours_file": "contours.json",
        "contour_report_file": "contour_report.csv",
        "mesh_npz_file": "mesh_raw.npz",
        "mesh_ply_file": "mesh_raw.ply",
        "validation_report_file": "validation_report.json",
        "validation_overview_file": "validation_overview.png",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not report["finite_vertices"]:
        raise AssertionError("Mesh contains non-finite vertices")
    if not report["mesh_watertight"]:
        raise AssertionError("Generated mesh is not watertight")
    if report["roundtrip_dice"] < args.roundtrip_dice_min:
        raise AssertionError(
            f"Round-trip Dice {report['roundtrip_dice']:.6f} is below "
            f"{args.roundtrip_dice_min:.6f}"
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Case directory: {output_dir}")
    print(f"Preview: {output_dir / 'validation_overview.png'}")
    print(f"Mesh: {output_dir / 'mesh_raw.ply'}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "inventory":
        return run_inventory(args)
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
