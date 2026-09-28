"""
step3_roi.py
============
Step 3: Perinasal ROI detection and cropping.
Reads from <subject>_clean/ folders (output of Step 2).
Saves cropped perinasal regions to <subject>_roi/.

ROI detection method: Thermal warm-blob analysis
- Threshold top 25% brightest pixels (warm face/body region)
- Morphological closing to fill gaps
- Largest contour = face bounding box
- Take lower 35-95% of box = perinasal zone (nose + mouth)

Usage: python step3_roi.py
"""

import sys, time, traceback
import numpy as np
import cv2
from pathlib import Path

# -- Configuration -----------------------------------------------------------
DATASET_DIR    = Path(r"c:\Users\arpit\OneDrive\Desktop\dataset")
SUBJECTS       = ["Joao", "anestis", "Claudio", "Manuel", "Jaime"]

PERI_Y_TOP     = 0.35   # skip forehead/eyes (top 35% of face box)
PERI_Y_BOT     = 0.95   # stop before chin
ROI_PAD        = 8      # pixel padding around detected region
# ----------------------------------------------------------------------------


def detect_perinasal_roi(img8):
    """
    Detect perinasal (nose+mouth) ROI in 8-bit thermal grayscale image.
    Returns (x, y, w, h).
    """
    H, W = img8.shape
    thresh_val = int(np.percentile(img8, 75))
    _, binary  = cv2.threshold(img8, thresh_val, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return (W // 4, H // 4, W // 2, H // 2)

    largest = max(contours, key=cv2.contourArea)
    fx, fy, fw, fh = cv2.boundingRect(largest)

    roi_y1 = max(0, int(fy + fh * PERI_Y_TOP) - ROI_PAD)
    roi_y2 = min(H, int(fy + fh * PERI_Y_BOT) + ROI_PAD)
    roi_x1 = max(0, fx - ROI_PAD)
    roi_x2 = min(W, fx + fw + ROI_PAD)

    rw = roi_x2 - roi_x1
    rh = roi_y2 - roi_y1
    if rw <= 0 or rh <= 0:
        return (W // 4, H // 4, W // 2, H // 2)

    return (roi_x1, roi_y1, rw, rh)


def stable_roi(sample_imgs):
    """Compute median ROI across a list of sample images."""
    rois = [detect_perinasal_roi(img) for img in sample_imgs]
    arr  = np.array(rois)
    return tuple(int(np.median(arr[:, i])) for i in range(4))


def save_preview(img8, roi, path):
    """Save colourised thermal image with ROI rectangle drawn."""
    coloured = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    rx, ry, rw, rh = roi
    cv2.rectangle(coloured, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(coloured, "Perinasal ROI", (rx, max(8, ry - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(path), coloured)


def process_subject(subj):
    src_dir = DATASET_DIR / f"{subj}_clean"
    roi_dir = DATASET_DIR / f"{subj}_roi"
    roi_dir.mkdir(exist_ok=True)

    print("")
    print("=" * 62)
    print("  SUBJECT  :", subj)
    print("  Source   :", src_dir.name + "/")
    print("  ROI out  :", roi_dir.name + "/")
    print("=" * 62)

    if not src_dir.exists():
        print("  ERROR: Source folder not found:", src_dir)
        return

    t_total = time.time()

    # Load all clean frames
    print("\n[Step 3] Loading clean frames...")
    t0 = time.time()
    frame_paths = sorted(src_dir.glob("frame_*.png"))
    if not frame_paths:
        print("  ERROR: No PNG frames found in", src_dir)
        return

    print("  Found", len(frame_paths), "clean frames")
    frames = []
    for i, p in enumerate(frame_paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        frames.append(img)
        if (i + 1) % 500 == 0:
            print(f"  Loaded {i+1}/{len(frame_paths)}  ({time.time()-t0:.1f}s)")

    print(f"  Loaded {len(frames)} frames in {time.time()-t0:.1f}s")
    print(f"  Frame shape: {frames[0].shape}  dtype: {frames[0].dtype}")

    # Compute stable perinasal ROI from samples
    print("\n[Step 3] Computing stable perinasal ROI from samples...")
    step = max(1, len(frames) // 20)
    samples = frames[::step][:20]
    roi = stable_roi(samples)
    rx, ry, rw, rh = roi
    print(f"  Perinasal ROI: x={rx}  y={ry}  w={rw}  h={rh}  ({rw}x{rh} px)")

    # Save colour preview
    mid = samples[len(samples) // 2]
    preview_path = DATASET_DIR / f"{subj}_roi_preview.png"
    save_preview(mid, roi, preview_path)
    print(f"  Preview saved: {preview_path.name}")

    # Crop and save all ROI frames
    print(f"\n[Step 3] Cropping {len(frames)} frames to perinasal ROI...")
    t2 = time.time()

    for i, img in enumerate(frames):
        crop     = img[ry:ry+rh, rx:rx+rw]
        out_path = roi_dir / f"frame_{i+1:06d}.png"
        cv2.imwrite(str(out_path), crop)
        if (i + 1) % 500 == 0:
            print(f"  Cropped {i+1:5d}/{len(frames)}  ({time.time()-t2:.1f}s)")

    print(f"  Step 3 done: {len(frames)} ROI frames saved in {time.time()-t2:.1f}s")
    print(f"  ROI size   : {rw} x {rh} pixels")

    # Pixel stats on mid-sequence ROI
    sample_roi = frames[len(frames) // 2][ry:ry+rh, rx:rx+rw]
    print(f"  ROI stats  : mean={sample_roi.mean():.1f}  "
          f"std={sample_roi.std():.1f}  "
          f"range=[{sample_roi.min()},{sample_roi.max()}]")
    print(f"\n  Subject '{subj}' done in {time.time()-t_total:.1f}s")


def main():
    print("Step 3: Perinasal ROI Detection and Cropping")
    print("Dataset:", DATASET_DIR)
    print("")

    available = [s for s in SUBJECTS
                 if (DATASET_DIR / f"{s}_clean").exists()]
    missing   = [s for s in SUBJECTS if s not in available]

    if missing:
        print("WARNING: No _clean folder found for:", missing)
    if not available:
        print("ERROR: No _clean folders found. Run step2 first.")
        sys.exit(1)

    print(f"Processing {len(available)} subjects: {available}")

    overall = time.time()
    for subj in available:
        try:
            process_subject(subj)
        except Exception:
            print(f"\nERROR processing {subj}:")
            traceback.print_exc()

    total_mins = (time.time() - overall) / 60
    print("")
    print("=" * 62)
    print(f"  ALL DONE in {total_mins:.1f} minutes")
    print("")
    print("  Results per subject:")
    for subj in available:
        clean = DATASET_DIR / f"{subj}_clean"
        roi   = DATASET_DIR / f"{subj}_roi"
        n_c   = len(list(clean.glob("*.png"))) if clean.exists() else 0
        n_r   = len(list(roi.glob("*.png")))   if roi.exists()   else 0
        preview = DATASET_DIR / f"{subj}_roi_preview.png"
        print(f"    {subj:10s}  clean={n_c:5d}  roi={n_r:5d}  "
              f"preview={'OK' if preview.exists() else 'MISSING'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
