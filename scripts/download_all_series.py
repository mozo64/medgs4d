from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import csv
import json
import time
import traceback

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
ROOT = Path("/opt/jupyterhub/fast/mtm_medgs_stack/data/tcia_4d_lung")
META = ROOT / "metadata" / "series.json"
OUT = ROOT / "raw" / "series_zips"
LOG = ROOT / "metadata" / "download_log.csv"

WORKERS = 4
RETRIES = 4
SLEEP_BASE = 10

OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

series = json.loads(META.read_text(encoding="utf-8"))

def safe_patient(pid: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(pid or "UNKNOWN"))

def series_target(s):
    pid = safe_patient(s.get("PatientID", "UNKNOWN"))
    uid = s["SeriesInstanceUID"]
    return OUT / pid / f"{uid}.zip"

def already_done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024

def download_one(s):
    uid = s["SeriesInstanceUID"]
    pid = s.get("PatientID", "")
    modality = s.get("Modality", "")
    image_count = s.get("ImageCount", "")
    desc = s.get("SeriesDescription", "")

    target = series_target(s)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".zip.part")

    if already_done(target):
        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "status": "skipped_existing",
            "PatientID": pid,
            "Modality": modality,
            "ImageCount": image_count,
            "SeriesDescription": desc,
            "SeriesInstanceUID": uid,
            "bytes": target.stat().st_size,
            "path": str(target),
            "error": "",
        }

    url = f"{BASE}/getImage?{urlencode({'SeriesInstanceUID': uid})}"

    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            if part.exists():
                part.unlink()

            req = Request(url, headers={"User-Agent": "MedGS-TCIA-download/1.0"})
            with urlopen(req, timeout=600) as r, part.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            size = part.stat().st_size
            if size <= 1024:
                raise RuntimeError(f"Downloaded file too small: {size} bytes")

            part.rename(target)

            return {
                "time": datetime.now().isoformat(timespec="seconds"),
                "status": "downloaded",
                "PatientID": pid,
                "Modality": modality,
                "ImageCount": image_count,
                "SeriesDescription": desc,
                "SeriesInstanceUID": uid,
                "bytes": size,
                "path": str(target),
                "error": "",
            }

        except Exception as e:
            last_err = f"attempt {attempt}/{RETRIES}: {repr(e)}"
            if attempt < RETRIES:
                time.sleep(SLEEP_BASE * attempt)

    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "status": "failed",
        "PatientID": pid,
        "Modality": modality,
        "ImageCount": image_count,
        "SeriesDescription": desc,
        "SeriesInstanceUID": uid,
        "bytes": 0,
        "path": str(target),
        "error": last_err + "\\n" + traceback.format_exc(limit=1),
    }

fields = [
    "time",
    "status",
    "PatientID",
    "Modality",
    "ImageCount",
    "SeriesDescription",
    "SeriesInstanceUID",
    "bytes",
    "path",
    "error",
]

write_header = not LOG.exists()

print(f"Series in manifest: {len(series)}")
print(f"Output directory: {OUT}")
print(f"Log: {LOG}")
print(f"Workers: {WORKERS}")

completed = 0
failed = 0
downloaded = 0
skipped = 0
total_bytes = 0

with LOG.open("a", newline="", encoding="utf-8") as log_f:
    writer = csv.DictWriter(log_f, fieldnames=fields)
    if write_header:
        writer.writeheader()
        log_f.flush()

    futures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s in series:
            futures.append(ex.submit(download_one, s))

        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="Downloading series")

        for fut in iterator:
            row = fut.result()
            writer.writerow(row)
            log_f.flush()

            completed += 1
            total_bytes += int(row.get("bytes") or 0)

            if row["status"] == "downloaded":
                downloaded += 1
            elif row["status"] == "skipped_existing":
                skipped += 1
            elif row["status"] == "failed":
                failed += 1

            if completed % 25 == 0 or row["status"] == "failed":
                gb = total_bytes / (1024 ** 3)
                print(
                    f"[{completed}/{len(series)}] "
                    f"downloaded={downloaded} skipped={skipped} failed={failed} "
                    f"seen_size={gb:.2f} GB"
                )

print("Done.")
print(f"downloaded={downloaded}, skipped={skipped}, failed={failed}")
print(f"total seen size={total_bytes / (1024 ** 3):.2f} GB")
print(f"log={LOG}")
