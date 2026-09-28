"""Audit the original recordings before constructing supervised examples.

CSV rows are timestamped and correspond one-to-one to thermal frames for four
subjects. Anestis has extra CSV rows, so its frame timestamps are unknown.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

SUBJECTS = ("Joao", "anestis", "Claudio", "Manuel", "Jaime")
TIME_FORMAT = "%Y%m%d-%H:%M:%S.%f"


def read_respiration(path: Path) -> tuple[list[datetime], list[int]]:
    times: list[datetime] = []
    markers: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if len(row) != 2:
                raise ValueError(f"Malformed CSV row in {path}: {row}")
            times.append(datetime.strptime(row[0], TIME_FORMAT))
            markers.append(int(row[1]))
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"Missing or non-increasing timestamps: {path}")
    return times, markers


def breath_events(times: list[datetime], markers: list[int]) -> list[datetime]:
    """Marker 1 is a breath event; Joao's marker 2 is its other phase."""
    return [t for t, marker in zip(times, markers) if marker == 1]


def recording_rate(times: list[datetime], markers: list[int]) -> float:
    events = breath_events(times, markers)
    duration = (times[-1] - times[0]).total_seconds()
    return 60.0 * len(events) / duration


def audit(root: Path) -> list[dict]:
    records = []
    for subject in SUBJECTS:
        frame_paths = sorted((root / subject).glob("frame_*.png"))
        csv_paths = list(root.glob(f"*{subject}_respiration.csv"))
        if len(csv_paths) != 1:
            raise FileNotFoundError(f"Expected one respiration CSV for {subject}")
        times, markers = read_respiration(csv_paths[0])
        intervals = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        marker_values = sorted(set(markers))
        records.append({
            "subject": subject,
            "frames": len(frame_paths),
            "csv_rows": len(times),
            "frame_csv_aligned_by_row": len(frame_paths) == len(times),
            "csv_duration_s": round((times[-1] - times[0]).total_seconds(), 3),
            "median_csv_interval_s": round(statistics.median(intervals), 4),
            "marker_values": marker_values,
            "breath_events_marker_1": len(breath_events(times, markers)),
            "reference_bpm_full_csv": round(recording_rate(times, markers), 2),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frame and reference-label alignment")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit(args.root)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
