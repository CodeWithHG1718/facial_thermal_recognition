"""Shared timestamp-preserving windows and explicit evaluation exclusions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_HZ = 10.0
EXCLUDED_SUBJECTS = {"anestis": "unresolved frame/reference alignment"}


@dataclass
class Window:
    subject: str
    start_s: float
    end_s: float
    condition: str
    x: np.ndarray | None
    reference_bpm: float | None
    missing_fraction: float | None
    signal_rejection: str | None
    reference_rejection: str | None

    @property
    def eligible(self):
        return self.signal_rejection is None and self.reference_rejection is None

    def metadata(self):
        return {key: value for key, value in vars(self).items() if key != "x"} | {
            "eligible": self.eligible,
        }


def condition_for(start: float, end: float, boundary: float = 60.0) -> str:
    if end <= boundary:
        return "still"
    if start >= boundary:
        return "moving"
    return "transition"


def reference_rate(event_t, start, end):
    events = event_t[(event_t >= start) & (event_t < end)]
    if len(events) < 4:
        return None, "fewer_than_four_events"
    intervals = np.diff(events)
    if np.any((intervals < 0.8) | (intervals > 10.0)):
        return None, "implausible_event_interval"
    bpm = 60.0 / float(np.median(intervals))
    if not 6.0 <= bpm <= 45.0:
        return None, "reference_out_of_range"
    # Preserve the precision used by the existing trainers.
    return float(np.float32(bpm)), None


def resample_signal(t, trace, valid, target_t):
    good_t = t[valid]
    if len(good_t) < 2 or target_t[0] < good_t[0] or target_t[-1] > good_t[-1]:
        return None, None, "insufficient_valid_support"
    channels = np.stack([np.interp(target_t, good_t, trace[valid, i]) for i in range(9)])
    nearest = np.searchsorted(good_t, target_t).clip(1, len(good_t) - 1)
    gap = good_t[nearest] - good_t[nearest - 1]
    missing = (gap > 0.25).astype(np.float32)
    missing_fraction = float(missing.mean())
    if missing_fraction > 0.2:
        return None, missing_fraction, "missing_fraction_over_20_percent"
    channels -= np.median(channels, axis=1, keepdims=True)
    channels /= np.maximum(np.std(channels, axis=1, keepdims=True), 1.0)
    x = np.concatenate([channels, missing[None, :]], axis=0).astype(np.float32)
    return x, missing_fraction, None


def load_window_records(folder: Path, window_s=20.0, step_s=5.0, still_until_s=60.0):
    if not all(np.isfinite(v) and v > 0 for v in (window_s, step_s, still_until_s)):
        raise ValueError("Window, stride, and condition boundary must be positive and finite")
    grid = np.arange(0.0, window_s, 1.0 / SAMPLE_HZ, dtype=np.float64)
    records = {}
    for path in sorted(folder.glob("*.npz")):
        if path.stem.lower() in EXCLUDED_SUBJECTS:
            continue
        with np.load(path, allow_pickle=False) as data:
            t, trace, valid, marker = (data[k] for k in ("seconds", "trace", "valid", "marker"))
        if (t.ndim != 1 or len(t) < 2 or trace.shape != (len(t), 9)
                or valid.shape != t.shape or marker.shape != t.shape
                or valid.dtype != np.bool_ or not np.isfinite(t).all()
                or not np.isfinite(trace).all() or np.any(np.diff(t) <= 0)
                or not np.isin(marker, (0, 1, 2)).all()):
            raise ValueError(f"Invalid timestamped trace record: {path}")
        event_t = t[marker == 1]
        windows = []
        # Include a window ending exactly at the final timestamp.
        for start in np.arange(t[0], t[-1] - window_s + 1e-9, step_s):
            end = start + window_s
            bpm, reference_rejection = reference_rate(event_t, start, end)
            x, missing_fraction, signal_rejection = resample_signal(t, trace, valid, start + grid)
            windows.append(Window(
                path.stem, float(start), float(end),
                condition_for(start - t[0], end - t[0], still_until_s),
                x, bpm, missing_fraction, signal_rejection, reference_rejection,
            ))
        records[path.stem] = windows
    return records


def load_windows(folder: Path, window_s=20.0, step_s=5.0):
    """Compatibility view used by the CNN, GRU and spectral MLP trainers."""
    return {
        subject: [(w.x, np.float32(w.reference_bpm)) for w in windows if w.eligible]
        for subject, windows in load_window_records(folder, window_s, step_s).items()
    }
