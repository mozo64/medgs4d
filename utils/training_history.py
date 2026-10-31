"""Optional CSV logging for canonical image training."""
import csv
from pathlib import Path


class CSVTrainingHistory:
    def __init__(self, path, log_every=100):
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every

    def __call__(self, row):
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
