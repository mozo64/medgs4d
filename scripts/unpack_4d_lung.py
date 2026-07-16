#!/usr/bin/env python3
#SBATCH --job-name=unpack-4d-lung
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=../../logs/%x-%j.out
#SBATCH --error=../../logs/%x-%j.err

"""
Resumably unpack TCIA 4D-Lung series archives.

Expected input layout:
    <project-root>/data/raw/series_zips/<PatientID>/<SeriesInstanceUID>.zip

Output layout:
    <project-root>/data/raw/dicom_by_series/<PatientID>/<SeriesInstanceUID>/

Submit directly with Slurm from the medgs4d repository root:

    sbatch scripts/unpack_4d_lung.py

The script verifies already extracted series by comparing archive member sizes,
so existing data can be adopted without being unpacked again. Interrupted runs
can be safely resubmitted.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MARKER_NAME = ".unpack_complete.json"


def default_project_root():
    explicit = os.environ.get("MEDGS4D_ROOT")
    if explicit:
        return Path(explicit)

    groups_storage = os.environ.get("PLG_GROUPS_STORAGE")
    user = os.environ.get("USER")
    if groups_storage and user:
        return Path(groups_storage) / "plggtriplane" / user / "medgs4d"

    raise RuntimeError(
        "Cannot determine the project root. Set MEDGS4D_ROOT or pass "
        "--project-root explicitly."
    )


def parse_args():
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    parser = argparse.ArgumentParser(
        description="Unpack TCIA 4D-Lung series ZIP archives by patient and series."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="MedGS4D project root containing data/, logs/, repo/, and envs/.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, slurm_cpus),
        help="Parallel extraction workers. Default: SLURM_CPUS_PER_TASK or 8.",
    )
    parser.add_argument(
        "--patient",
        action="append",
        default=[],
        help="Extract only this PatientID. May be supplied more than once.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N archives, useful for a smoke test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract archives even when the destination appears complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the planned work without extracting files.",
    )
    return parser.parse_args()


def archive_members(zf):
    members = []
    for info in zf.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe archive member path: {0}".format(info.filename))
        if not info.is_dir():
            members.append(info)

    if not members:
        raise ValueError("Archive contains no files")

    return members


def destination_path(root, member_name):
    relative = PurePosixPath(member_name)
    return root.joinpath(*relative.parts)


def marker_matches(marker_path, archive_path):
    try:
        data = json.loads(marker_path.read_text())
        stat = archive_path.stat()
        return (
            data.get("archive_size") == stat.st_size
            and data.get("archive_mtime_ns") == stat.st_mtime_ns
        )
    except (OSError, ValueError, TypeError):
        return False


def existing_tree_matches(destination, members):
    if not destination.is_dir():
        return False

    for info in members:
        path = destination_path(destination, info.filename)
        try:
            if not path.is_file() or path.stat().st_size != info.file_size:
                return False
        except OSError:
            return False

    return True


def apply_permissions(root):
    for current_root, directories, files in os.walk(str(root)):
        current_path = Path(current_root)
        current_path.chmod(0o2750)

        for name in directories:
            (current_path / name).chmod(0o2750)

        for name in files:
            (current_path / name).chmod(0o640)


def write_marker(destination, archive_path, members):
    stat = archive_path.stat()
    payload = {
        "archive": str(archive_path),
        "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
        "member_count": len(members),
        "uncompressed_bytes": sum(info.file_size for info in members),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }

    marker = destination / MARKER_NAME
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    marker.chmod(0o640)


def extract_one(task):
    archive_text, source_root_text, destination_root_text, force, dry_run = task

    archive = Path(archive_text)
    source_root = Path(source_root_text)
    destination_root = Path(destination_root_text)

    relative = archive.relative_to(source_root)
    patient_id = relative.parent.name
    series_uid = archive.stem
    destination = destination_root / patient_id / series_uid
    marker = destination / MARKER_NAME
    started = time.time()

    result = {
        "patient_id": patient_id,
        "series_uid": series_uid,
        "archive": str(archive),
        "destination": str(destination),
        "status": "",
        "files": 0,
        "uncompressed_bytes": 0,
        "seconds": 0.0,
        "message": "",
    }

    try:
        if dry_run:
            result["status"] = "dry-run"
            return result

        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o2750)
        destination.parent.chmod(0o2750)

        with zipfile.ZipFile(str(archive), "r") as zf:
            members = archive_members(zf)
            result["files"] = len(members)
            result["uncompressed_bytes"] = sum(info.file_size for info in members)

            if not force:
                if marker.is_file() and marker_matches(marker, archive):
                    result["status"] = "skipped"
                    result["message"] = "completion marker matches archive"
                    return result

                if existing_tree_matches(destination, members):
                    apply_permissions(destination)
                    write_marker(destination, archive, members)
                    result["status"] = "skipped"
                    result["message"] = "existing files verified and marker created"
                    return result

            temporary = destination.parent / (".{0}.extracting".format(series_uid))
            if temporary.exists():
                shutil.rmtree(str(temporary))

            temporary.mkdir(parents=True, mode=0o2750)
            temporary.chmod(0o2750)

            for info in members:
                target = destination_path(temporary, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o2750)
                target.parent.chmod(0o2750)

                with zf.open(info, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

            if not existing_tree_matches(temporary, members):
                raise RuntimeError("extracted file verification failed")

            apply_permissions(temporary)

            if destination.exists():
                shutil.rmtree(str(destination))

            os.replace(str(temporary), str(destination))
            write_marker(destination, archive, members)

            result["status"] = "extracted"
            result["message"] = "archive extracted and verified"

    except Exception as exc:
        result["status"] = "failed"
        result["message"] = "{0}: {1}".format(type(exc).__name__, exc)

    finally:
        result["seconds"] = round(time.time() - started, 3)

    return result


def discover_archives(source_root, patients):
    archives = sorted(source_root.glob("*/*.zip"))

    if patients:
        selected = set(patients)
        archives = [path for path in archives if path.parent.name in selected]

        found = set(path.parent.name for path in archives)
        missing = selected - found
        if missing:
            raise RuntimeError(
                "No archives found for PatientID(s): {0}".format(
                    ", ".join(sorted(missing))
                )
            )

    return archives


def main():
    os.umask(0o027)
    args = parse_args()

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else default_project_root().resolve()
    )

    source_root = project_root / "data" / "raw" / "series_zips"
    destination_root = project_root / "data" / "raw" / "dicom_by_series"
    logs_root = project_root / "logs"

    if not source_root.is_dir():
        raise RuntimeError("Archive directory not found: {0}".format(source_root))

    destination_root.mkdir(parents=True, exist_ok=True, mode=0o2750)
    destination_root.chmod(0o2750)
    logs_root.mkdir(parents=True, exist_ok=True, mode=0o2750)
    logs_root.chmod(0o2750)

    archives = discover_archives(source_root, args.patient)
    if args.limit is not None:
        archives = archives[: max(0, args.limit)]
    if not archives:
        raise RuntimeError("No ZIP archives selected")

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    log_path = logs_root / "unpack_4d_lung_{0}.csv".format(job_id)

    fieldnames = [
        "patient_id",
        "series_uid",
        "archive",
        "destination",
        "status",
        "files",
        "uncompressed_bytes",
        "seconds",
        "message",
    ]

    print("4D-Lung extraction")
    print("  Project root:      {0}".format(project_root))
    print("  Archive root:      {0}".format(source_root))
    print("  Destination root:  {0}".format(destination_root))
    print("  Selected archives: {0}".format(len(archives)))
    print("  Workers:           {0}".format(args.workers))
    print("  Force:             {0}".format(args.force))
    print("  Dry run:           {0}".format(args.dry_run))
    print("  CSV log:           {0}".format(log_path))
    print("", flush=True)

    tasks = [
        (
            str(archive),
            str(source_root),
            str(destination_root),
            args.force,
            args.dry_run,
        )
        for archive in archives
    ]

    counts = {"extracted": 0, "skipped": 0, "failed": 0, "dry-run": 0}
    total_uncompressed = 0
    started = time.time()

    with log_path.open("w", newline="") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
        writer.writeheader()
        log_handle.flush()

        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {
                executor.submit(extract_one, task): task[0]
                for task in tasks
            }

            completed = 0
            for future in as_completed(future_map):
                completed += 1
                archive = future_map[future]

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "patient_id": "",
                        "series_uid": "",
                        "archive": archive,
                        "destination": "",
                        "status": "failed",
                        "files": 0,
                        "uncompressed_bytes": 0,
                        "seconds": 0.0,
                        "message": "WorkerFailure: {0}".format(exc),
                    }

                counts[result["status"]] = counts.get(result["status"], 0) + 1
                total_uncompressed += int(result["uncompressed_bytes"])
                writer.writerow(result)
                log_handle.flush()

                if (
                    result["status"] == "failed"
                    or completed == len(tasks)
                    or completed % 25 == 0
                ):
                    elapsed = time.time() - started
                    print(
                        "[{0}/{1}] extracted={2} skipped={3} failed={4} "
                        "elapsed={5:.1f} min".format(
                            completed,
                            len(tasks),
                            counts.get("extracted", 0),
                            counts.get("skipped", 0),
                            counts.get("failed", 0),
                            elapsed / 60.0,
                        ),
                        flush=True,
                    )

                    if result["status"] == "failed":
                        print(
                            "  FAILED: {0} -- {1}".format(
                                result["archive"], result["message"]
                            ),
                            flush=True,
                        )

    elapsed = time.time() - started

    print("")
    print("Extraction summary")
    print("  Extracted:          {0}".format(counts.get("extracted", 0)))
    print("  Skipped:            {0}".format(counts.get("skipped", 0)))
    print("  Failed:             {0}".format(counts.get("failed", 0)))
    print("  Dry-run:            {0}".format(counts.get("dry-run", 0)))
    print("  Selected archives:  {0}".format(len(archives)))
    print("  Reported raw size:  {0:.2f} GiB".format(total_uncompressed / 1024.0**3))
    print("  Elapsed:            {0:.1f} minutes".format(elapsed / 60.0))
    print("  CSV log:            {0}".format(log_path))

    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
