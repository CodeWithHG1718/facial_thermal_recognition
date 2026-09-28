"""
step2_step3.py
==============
Step 2 : Load already-extracted 8-bit PNG frames → detect & remove frozen frames
         → save valid frames to <subject>_clean/

Step 3 : Thermal-based perinasal ROI detection (warm-blob analysis)
         → crop lower face / nose+mouth zone from each clean frame
         → save to <subject>_roi/

NOTE: Since the original .bag files are no longer available, we work from
      the 8-bit PNGs extracted in Phase 0. The 8-bit normalization is retained;
      frozen frame removal is the primary quality-control step.

Usage:  python step2_step3.py
"""

import sys, time, traceback
import numpy as np
import cv2
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────
DATASET_DIR    = Path(r"c:\Users\arpit\OneDrive\Desktop\dataset")
SUBJECTS       = ["Joao", "anestis", "Claudio", "Manuel", "Jaime"]

FROZEN_MIN_RUN = 5    # min consecutive identical frames = frozen run
FROZEN_MARGIN  = 2    # extra frames to remove around each frozen run

# Perinasal ROI fractions (of warm-face bounding box height)
PERI_Y_TOP     = 0.35  # skip forehead + eyes (top 35%)
PERI_Y_BOT     = 0.95  # stop before chin
ROI_PAD        = 8     # pixel padding
# ───────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════
# FROZEN FRAME DETECTION
# ══════════════════════════════════════════════════════════════════════════

def find_frozen_runs(frames: list, min_run: int = 5) -> list:
    """Return list of (start, end) inclusive index ranges for frozen runs."""
    runs, run_start = [], None
    for i in range(1, len(frames)):
        if np.array_equal(frames[i], frames[i-1]):
            if run_start is None:
                run_start = i - 1
        else:
            if run_start is not None:
                if (i-1) - run_start + 1 >= min_run:
                    runs.append((run_start, i-1))
                run_start = None
    if run_start is not None and (len(frames)-1) - run_start + 1 >= min_run:
        runs.append((run_start, len(frames)-1))
    return runs


def valid_mask(total, frozen_runs, margin=2):
    mask = np.ones(total, dtype=bool)
    for s, e in frozen_runs:
        mask[max(0,s-margin): min(total-1,e+margin)+1] = False
    return mask


# ══════════════════════════════════════════════════════════════════════════
# THERMAL FACE ROI DETECTION  (warm-blob method — no model needed)
# ══════════════════════════════════════════════════════════════════════════

def detect_perinasal_roi(img8: np.ndarray) -> tuple:
    """
    Find face bounding box in an 8-bit grayscale thermal image,
    then return the perinasal sub-region (x, y, w, h).

    Algorithm:
      1. Threshold top-25% brightest pixels → warm (face/body) region.
      2. Morphological close → fill gaps.
      3. Largest contour = face/torso.
      4. Lower 35–95% of that bounding box = nose + mouth zone.
    """
    H, W = img8.shape

    thresh_val = int(np.percentile(img8, 75))
    _, binary  = cv2.threshold(img8, thresh_val, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: image-centre crop
        return (W//4, H//4, W//2, H//2)

    largest = max(contours, key=cv2.contourArea)
    fx, fy, fw, fh = cv2.boundingRect(largest)

    roi_y1 = max(0, int(fy + fh * PERI_Y_TOP) - ROI_PAD)
    roi_y2 = min(H, int(fy + fh * PERI_Y_BOT) + ROI_PAD)
    roi_x1 = max(0, fx - ROI_PAD)
    roi_x2 = min(W, fx + fw + ROI_PAD)

    rw = roi_x2 - roi_x1
    rh = roi_y2 - roi_y1

    if rw <= 0 or rh <= 0:
        return (W//4, H//4, W//2, H//2)

    return (roi_x1, roi_y1, rw, rh)


def stable_roi_from_samples(sample_imgs: list) -> tuple:
    """Median ROI across up to 20 sample frames → stable crop box."""
    rois = [detect_perinasal_roi(img) for img in sample_imgs]
    arr  = np.array(rois)
    return tuple(int(np.median(arr[:, i])) for i in range(4))


def save_preview(img8: np.ndarray, roi: tuple, path: Path):
    """Save a colour-mapped thermal image with ROI rectangle overlaid."""
    coloured = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    rx, ry, rw, rh = roi
    cv2.rectangle(coloured, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
    cv2.putText(coloured, "Perinasal ROI", (rx, max(8, ry-4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(path), coloured)


# ══════════════════════════════════════════════════════════════════════════
# PER-SUBJECT PROCESSOR
# ══════════════════════════════════════════════════════════════════════════

def process_subject(subj: str):
    src_dir  = DATASET_DIR / subj
    clean_dir = DATASET_DIR / f"{subj}_clean"
    roi_dir   = DATASET_DIR / f"{subj}_roi"
    clean_dir.mkdir(exist_ok=True)
    roi_dir.mkdir(exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  SUBJECT  : {subj}")
    print(f"  Source   : {src_dir.name}/")
    print(f"  Clean out: {clean_dir.name}/")
    print(f"  ROI out  : {roi_dir.name}/")
    print(f"{'='*62}")

    t_total = time.time()

    # ── Load all frames from disk ────────────────────────────────────────
    print("\n[Step 2] Loading frames from disk...")
    t0 = time.time()

    frame_paths = sorted(src_dir.glob("frame_*.png"))
    if not frame_paths:
        print(f"  ERROR: No PNG frames found in {src_dir}")
        return

    print(f"  Found {len(frame_paths)} PNG frames")
    frames = []
    for i, p in enumerate(frame_paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        frames.append(img)
        if (i+1) % 500 == 0:
            print(f"  Loaded {i+1}/{len(frame_paths)}  ({time.time()-t0:.1f}s)")

    print(f"  Loaded all {len(frames)} frames in {time.time()-t0:.1f}s")
    print(f"  Frame shape: {frames[0].shape}  dtype: {frames[0].dtype}")

    # ── Frozen frame detection ───────────────────────────────────────────
    print("\n[Step 2] Detecting frozen frame runs...")
    frozen_runs = find_frozen_runs(frames, FROZEN_MIN_RUN)
    mask        = valid_mask(len(frames), frozen_runs, FROZEN_MARGIN)
    n_valid     = int(mask.sum())
    n_frozen    = len(frames) - n_valid

    print(f"  Total frames  : {len(frames)}")
    print(f"  Frozen runs   : {len(frozen_runs)}")
    for s, e in frozen_runs:
        print(f"    frames {s+1:5d}–{e+1:5d}  ({e-s+1} frames = ~{(e-s+1)/25:.1f}s)")
    print(f"  Frozen removed: {n_frozen}  ({n_frozen/len(frames)*100:.1f}%)")
    print(f"  Valid frames  : {n_valid}")

    # ── Save clean (frozen-removed) frames ──────────────────────────────
    print("\n[Step 2] Saving clean frames (frozen removed)...")
    t1 = time.time()
    valid_frames = []      # list of (orig_idx, 8-bit img)
    new_frame_num = 0

    for orig_idx, (img, is_valid) in enumerate(zip(frames, mask)):
        if not is_valid:
            continue
        new_frame_num += 1
        out_path = clean_dir / f"frame_{new_frame_num:06d}.png"
        cv2.imwrite(str(out_path), img)
        valid_frames.append((orig_idx, img))
        if new_frame_num % 500 == 0:
            print(f"  Saved {new_frame_num:5d}/{n_valid}  ({time.time()-t1:.1f}s)")

    print(f"  Step 2 done : {new_frame_num} clean frames → {clean_dir.name}/  "
          f"({time.time()-t1:.1f}s)")

    # ── STEP 3A: Find stable perinasal ROI ───────────────────────────────
    print("\n[Step 3] Computing stable perinasal ROI...")
    step = max(1, new_frame_num // 20)
    samples = [img for (_, img) in valid_frames[::step]][:20]
    roi = stable_roi_from_samples(samples)
    rx, ry, rw, rh = roi
    print(f"  Stable ROI  : x={rx}  y={ry}  w={rw}  h={rh}  ({rw}x{rh} px)")

    # Save colour preview
    mid = samples[len(samples)//2]
    preview_path = DATASET_DIR / f"{subj}_roi_preview.png"
    save_preview(mid, roi, preview_path)
    print(f"  Preview     : {preview_path.name}")

    # ── STEP 3B: Crop and save ROI frames ────────────────────────────────
    print(f"\n[Step 3] Cropping {new_frame_num} frames to perinasal ROI...")
    t2 = time.time()

    for fn_idx, (orig_idx, img) in enumerate(valid_frames):
        crop = img[ry:ry+rh, rx:rx+rw]
        out_path = roi_dir / f"frame_{fn_idx+1:06d}.png"
        cv2.imwrite(str(out_path), crop)
        if (fn_idx+1) % 500 == 0:
            print(f"  Cropped {fn_idx+1:5d}/{new_frame_num}  ({time.time()-t2:.1f}s)")

    print(f"  Step 3 done : {new_frame_num} ROI frames → {roi_dir.name}/  "
          f"({time.time()-t2:.1f}s)")
    print(f"  ROI size    : {rw} x {rh} pixels")
    print(f"\n  Subject '{subj}' complete in {time.time()-t_total:.1f}s total")

    # Print summary stats
    sample_roi = valid_frames[len(valid_frames)//2][1][ry:ry+rh, rx:rx+rw]
    print(f"  ROI pixel stats: mean={sample_roi.mean():.1f}  "
          f"std={sample_roi.std():.1f}  "
          f"range=[{sample_roi.min()},{sample_roi.max()}]")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Pipeline: Step 2 (Frozen Frame Removal) + Step 3 (Perinasal ROI Crop)")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Subjects: {SUBJECTS}")

    available = [s for s in SUBJECTS if (DATASET_DIR / s).exists()]
    missing   = [s for s in SUBJECTS if s not in available]
    if missing:
        print(f"\nWARNING: Source folders not found for: {missing}")
    if not available:
        print("ERROR: No subject folders found. Run Phase 0 (bag extraction) first.")
        sys.exit(1)

    print(f"\nProcessing {len(available)} subject(s): {available}")
    overall = time.time()

    for subj in available:
        try:
            process_subject(subj)
        except Exception:
            print(f"\nERROR processing {subj}:")
            traceback.print_exc()

    total_mins = (time.time() - overall) / 60
    print(f"\n{'='*62}")
    print(f"  ALL DONE in {total_mins:.1f} minutes")
    print(f"\n  Final frame counts:")
    for subj in available:
        src   = DATASET_DIR / subj
        clean = DATASET_DIR / f"{subj}_clean"
        roi   = DATASET_DIR / f"{subj}_roi"
        n_src   = len(list(src.glob("*.png")))   if src.exists()   else 0
        n_clean = len(list(clean.glob("*.png"))) if clean.exists() else 0
        n_roi   = len(list(roi.glob("*.png")))   if roi.exists()   else 0
        removed = n_src - n_clean
        print(f"    {subj:10s}  "
              f"original={n_src:5d}  "
              f"clean={n_clean:5d} (-{removed})  "
              f"roi={n_roi:5d}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
