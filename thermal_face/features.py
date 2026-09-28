"""Extract timestamped nose-region traces from original thermal frames.

The five boxes in config/nose_boxes.json are provisional hand-inspected boxes
for this dataset, not an automatic face detector. No derived frame directories
are required. Frozen runs are flagged and excluded from interpolation anchors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import SUBJECTS, read_respiration


def extract(root: Path, out: Path) -> None:
    import numpy as np
    from PIL import Image

    boxes = json.loads((root / "config" / "nose_boxes.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        frames = sorted((root / subject).glob("frame_*.png"))
        csv_path = next(root.glob(f"*{subject}_respiration.csv"))
        times, markers = read_respiration(csv_path)
        if len(frames) != len(times):
            print(f"Skipping {subject}: {len(frames)} frames vs {len(times)} CSV rows; no verified alignment")
            continue
        x, y, w, h = boxes[subject]
        traces = np.zeros((len(frames), 9), dtype=np.float32)
        repeated = np.zeros(len(frames), dtype=np.bool_)
        prev = None
        for i, path in enumerate(frames):
            with Image.open(path) as image:
                frame = np.asarray(image).copy()
            if frame.ndim == 3 and frame.shape[2] == 3:
                if not (np.array_equal(frame[:, :, 0], frame[:, :, 1]) and
                        np.array_equal(frame[:, :, 0], frame[:, :, 2])):
                    raise ValueError(f"Expected grayscale channels in {path}")
                frame = frame[:, :, 0]
            if frame is None or frame.ndim != 2 or frame.shape[0] < y + h or frame.shape[1] < x + w:
                raise ValueError(f"Invalid frame or nose box: {path}")
            if prev is not None:
                repeated[i] = np.array_equal(frame, prev)
            prev = frame
            crop = frame[y:y+h, x:x+w].astype(np.float32)
            for row in range(3):
                for col in range(3):
                    part = crop[row*h//3:(row+1)*h//3, col*w//3:(col+1)*w//3]
                    traces[i, row*3+col] = float(part.mean())
        # Treat a run of five or more identical frames as missing thermal data.
        valid = np.ones(len(frames), dtype=np.bool_)
        start = None
        for i in range(1, len(frames) + 1):
            same = i < len(frames) and repeated[i]
            if same and start is None:
                start = i - 1
            elif not same and start is not None:
                if i - start >= 5:
                    valid[start:i] = False
                start = None
        seconds = np.array([(t - times[0]).total_seconds() for t in times], dtype=np.float64)
        np.savez_compressed(
            out / f"{subject}.npz", seconds=seconds, trace=traces,
            valid=valid, marker=np.asarray(markers, dtype=np.int8),
        )
        print(f"{subject}: {len(frames)} frames, {np.count_nonzero(~valid)} frozen-run frames")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("artifacts/traces"))
    args = parser.parse_args()
    extract(args.root, args.out)


if __name__ == "__main__":
    main()
