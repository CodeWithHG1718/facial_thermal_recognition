"""
step4_5_6.py
============
Step 4 : KLT optical flow stabilization (head motion compensation)
         Input : <subject>_roi/       Output: <subject>_stable/

Step 5 : BAF map - rolling per-pixel standard deviation (100-frame window)
         Input : <subject>_stable/    Output: <subject>_baf/

Step 6 : BAP map - normalize BAF by frame maximum -> probability [0,1]
         Input : <subject>_baf data   Output: <subject>_bap/
                                               <subject>_bap_mean.png (heatmap summary)

Usage:  python step4_5_6.py
"""

import sys, time, traceback
import numpy as np
import cv2
from pathlib import Path

# ---- Configuration ----------------------------------------------------------
DATASET_DIR    = Path(r"c:\Users\arpit\OneDrive\Desktop\dataset")
SUBJECTS       = ["Joao", "anestis", "Claudio", "Manuel", "Jaime"]

BAF_WINDOW     = 100      # frames in rolling std-dev window (= 4 seconds @ 25 fps)

# KLT parameters
KLT_MAX_CORNERS   = 300   # max Shi-Tomasi corners to detect
KLT_QUALITY       = 0.01  # quality level for corner detection (low = more corners)
KLT_MIN_DIST      = 5     # min pixel distance between corners
KLT_WIN_SIZE      = (15, 15)
KLT_MAX_LEVEL     = 2     # pyramid levels for Lucas-Kanade
KLT_REDETECT_FREQ = 100   # redetect features every N frames (prevents drift)
KLT_MIN_FEATURES  = 10    # min tracked features; redetect if below this
# -----------------------------------------------------------------------------


# =============================================================================
# STEP 4 - KLT STABILIZATION
# =============================================================================

def detect_features(frame: np.ndarray) -> np.ndarray:
    """Detect Shi-Tomasi corner features in a grayscale frame."""
    pts = cv2.goodFeaturesToTrack(
        frame,
        maxCorners=KLT_MAX_CORNERS,
        qualityLevel=KLT_QUALITY,
        minDistance=KLT_MIN_DIST,
        blockSize=7,
    )
    return pts if pts is not None else np.zeros((0, 1, 2), dtype=np.float32)


def track_features(prev: np.ndarray, curr: np.ndarray,
                   prev_pts: np.ndarray) -> tuple:
    """
    Track features from prev to curr using Lucas-Kanade optical flow.
    Returns (prev_good, curr_good) matched point arrays.
    """
    if prev_pts is None or len(prev_pts) == 0:
        return None, None

    lk_params = dict(
        winSize=KLT_WIN_SIZE,
        maxLevel=KLT_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, prev_pts, None, **lk_params
    )
    if curr_pts is None or status is None:
        return None, None

    status = status.ravel().astype(bool)
    if status.sum() < KLT_MIN_FEATURES:
        return None, None

    return prev_pts[status], curr_pts[status]


def estimate_transform(src_pts: np.ndarray,
                       dst_pts: np.ndarray) -> np.ndarray | None:
    """
    Estimate a partial affine transform (4-DOF: translation + rotation + scale)
    from matched point pairs. Returns 2x3 affine matrix or None on failure.
    """
    if src_pts is None or len(src_pts) < 4:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return M


def stabilize_subject(subj: str):
    """
    Step 4: KLT stabilization.
    Reads from <subj>_roi/, writes stabilized frames to <subj>_stable/.
    Uses the first frame as the fixed reference.
    Accumulates the affine transform to map every frame back to reference coords.
    """
    src_dir = DATASET_DIR / f"{subj}_roi"
    out_dir = DATASET_DIR / f"{subj}_stable"
    out_dir.mkdir(exist_ok=True)

    frame_paths = sorted(src_dir.glob("frame_*.png"))
    if not frame_paths:
        print(f"  ERROR: No frames in {src_dir}")
        return 0

    n = len(frame_paths)
    print(f"\n  [Step 4] Stabilizing {n} frames with KLT...")
    t0 = time.time()

    # Load reference (first) frame
    ref = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    H, W = ref.shape

    # Save reference frame unchanged
    cv2.imwrite(str(out_dir / "frame_000001.png"), ref)

    # Cumulative transform from reference to current frame
    # Start as identity
    cumulative_M = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)

    prev_frame = ref.copy()
    prev_pts   = detect_features(prev_frame)

    n_fallback = 0  # count frames where transform estimation failed

    for i, path in enumerate(frame_paths[1:], start=1):
        curr_frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        # Optionally redetect features to prevent long-term drift
        if i % KLT_REDETECT_FREQ == 0 or len(prev_pts) < KLT_MIN_FEATURES:
            prev_pts = detect_features(prev_frame)

        # Track features from prev to curr
        prev_good, curr_good = track_features(prev_frame, curr_frame, prev_pts)

        # Estimate incremental transform (prev -> curr)
        M_inc = estimate_transform(prev_good, curr_good)

        if M_inc is not None:
            # Compose cumulative transform: ref -> curr = (ref -> prev) . (prev -> curr)
            # Expand to 3x3 for matrix multiplication
            M_inc_3x3 = np.vstack([M_inc, [0, 0, 1]])
            cum_3x3   = np.vstack([cumulative_M, [0, 0, 1]])
            new_3x3   = M_inc_3x3 @ cum_3x3
            cumulative_M = new_3x3[:2, :]

            # Update previous features to current positions for next iteration
            prev_pts = curr_good.reshape(-1, 1, 2)
        else:
            # Fallback: keep previous transform, redetect in next frame
            prev_pts = detect_features(curr_frame)
            n_fallback += 1

        # Warp current frame to reference coordinate system
        stabilized = cv2.warpAffine(
            curr_frame, cumulative_M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        out_path = out_dir / f"frame_{i+1:06d}.png"
        cv2.imwrite(str(out_path), stabilized)

        prev_frame = curr_frame  # use unwarped for next KLT step

        if (i + 1) % 500 == 0:
            print(f"    Stabilized {i+1:5d}/{n}  ({time.time()-t0:.1f}s)")

    print(f"    Step 4 done: {n} frames stabilized in {time.time()-t0:.1f}s")
    print(f"    KLT fallbacks (frames with failed tracking): {n_fallback}/{n}")
    return n


# =============================================================================
# STEP 5 + 6 - BAF MAP + BAP MAP
# =============================================================================

def compute_baf_bap(subj: str, n_frames: int):
    """
    Step 5: Compute BAF maps using memory-efficient running-sum rolling std dev.
    Step 6: Normalize BAF by per-frame max to get BAP probability map [0,1].

    Memory approach:
      - Load all frames as uint8 numpy array (N x H x W)  ~117-130 MB per subject
      - Maintain running sum and running sum-of-squares (float64, H x W each)
      - Slide window of BAF_WINDOW frames, update O(1) per step
      - Write BAF and BAP PNG immediately -> never store all maps in memory
    """
    src_dir = DATASET_DIR / f"{subj}_stable"
    baf_dir = DATASET_DIR / f"{subj}_baf"
    bap_dir = DATASET_DIR / f"{subj}_bap"
    baf_dir.mkdir(exist_ok=True)
    bap_dir.mkdir(exist_ok=True)

    frame_paths = sorted(src_dir.glob("frame_*.png"))
    if not frame_paths:
        print(f"  ERROR: No frames in {src_dir}")
        return

    n = len(frame_paths)
    if n < BAF_WINDOW:
        print(f"  ERROR: Only {n} frames, need at least {BAF_WINDOW} for BAF window.")
        return

    # --- Load all frames into uint8 array ---
    print(f"\n  [Step 5] Loading {n} stabilized frames...")
    t0 = time.time()

    sample = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    H, W = sample.shape
    frames = np.empty((n, H, W), dtype=np.uint8)
    frames[0] = sample

    for i, p in enumerate(frame_paths[1:], start=1):
        frames[i] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if (i + 1) % 500 == 0:
            print(f"    Loaded {i+1:5d}/{n}  ({time.time()-t0:.1f}s)")

    print(f"    Loaded {n} frames ({frames.nbytes/1e6:.1f} MB) in {time.time()-t0:.1f}s")

    # --- Initialize running sums for first window ---
    print(f"\n  [Step 5+6] Computing BAF + BAP maps (window={BAF_WINDOW} frames)...")
    t1 = time.time()

    window_f = frames[:BAF_WINDOW].astype(np.float64)
    rsum    = window_f.sum(axis=0)           # shape (H, W)
    rsum_sq = (window_f ** 2).sum(axis=0)    # shape (H, W)

    n_windows  = n - BAF_WINDOW + 1
    bap_accum  = np.zeros((H, W), dtype=np.float64)  # for time-averaged BAP
    max_baf_global = 0.0

    for i in range(n_windows):
        # Compute variance from running sums
        mean     = rsum / BAF_WINDOW
        variance = rsum_sq / BAF_WINDOW - mean ** 2
        np.maximum(variance, 0, out=variance)  # numerical floor at 0
        baf_map  = np.sqrt(variance).astype(np.float32)

        # BAF max for this frame
        baf_max = float(baf_map.max())
        if baf_max > max_baf_global:
            max_baf_global = baf_max

        # Save BAF as uint8 PNG (scale by global frame max for lossless comparison)
        if baf_max > 0:
            baf_u8 = (baf_map / baf_max * 255).astype(np.uint8)
        else:
            baf_u8 = np.zeros((H, W), dtype=np.uint8)

        cv2.imwrite(str(baf_dir / f"frame_{i+1:06d}.png"), baf_u8)

        # BAP = BAF / max(BAF) -> already = baf_u8 / 255 when reloaded
        # Save the same scaled image to bap_dir
        cv2.imwrite(str(bap_dir / f"frame_{i+1:06d}.png"), baf_u8)

        # Accumulate for time-averaged BAP
        if baf_max > 0:
            bap_accum += (baf_map / baf_max)

        # Update running sums: slide window by 1
        if i + BAF_WINDOW < n:
            old_frame = frames[i].astype(np.float64)
            new_frame = frames[i + BAF_WINDOW].astype(np.float64)
            rsum    += new_frame - old_frame
            rsum_sq += new_frame ** 2 - old_frame ** 2

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t1
            pct = (i + 1) / n_windows * 100
            print(f"    BAF/BAP map {i+1:5d}/{n_windows}  ({pct:.0f}%)  "
                  f"{elapsed:.1f}s  max_baf_so_far={max_baf_global:.2f}")

    elapsed = time.time() - t1
    print(f"    Step 5+6 done: {n_windows} BAF/BAP maps in {elapsed:.1f}s")
    print(f"    Max BAF value observed: {max_baf_global:.3f} (out of 255)")

    # --- Save time-averaged BAP heatmap ---
    bap_mean = (bap_accum / n_windows).astype(np.float32)

    # Normalise to [0,255] for saving
    bap_mean_u8 = (bap_mean / max(bap_mean.max(), 1e-6) * 255).astype(np.uint8)

    # Grayscale version
    cv2.imwrite(str(DATASET_DIR / f"{subj}_bap_mean.png"), bap_mean_u8)

    # Colour heatmap version (INFERNO colourmap)
    bap_colour = cv2.applyColorMap(bap_mean_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(DATASET_DIR / f"{subj}_bap_mean_colour.png"), bap_colour)

    print(f"    Saved: {subj}_bap_mean.png  +  {subj}_bap_mean_colour.png")
    print(f"    BAP mean range: [{bap_mean.min():.4f}, {bap_mean.max():.4f}]")

    # --- Print where BAP is highest (expected = perinasal zone) ---
    flat_idx = int(np.argmax(bap_mean))
    peak_y, peak_x = np.unravel_index(flat_idx, (H, W))
    print(f"    Peak BAP pixel: ({peak_x}, {peak_y}) in the {W}x{H} ROI")
    print(f"    (Expected to be in the nose/upper-lip area)")

    return n_windows, max_baf_global


# =============================================================================
# MAIN
# =============================================================================

def process_subject(subj: str):
    t_total = time.time()

    print("")
    print("=" * 64)
    print(f"  SUBJECT : {subj}")
    print(f"  ROI in  : {subj}_roi/")
    print(f"  Outputs : {subj}_stable/  {subj}_baf/  {subj}_bap/")
    print("=" * 64)

    # Check source
    src = DATASET_DIR / f"{subj}_roi"
    if not src.exists():
        print(f"  ERROR: {src} not found. Run step3_roi.py first.")
        return

    # Step 4
    n_stable = stabilize_subject(subj)

    # Steps 5 + 6
    result = compute_baf_bap(subj, n_stable)
    if result:
        n_maps, max_baf = result
        print(f"\n  SUMMARY for {subj}:")
        print(f"    Stabilized frames : {n_stable}")
        print(f"    BAF/BAP maps      : {n_maps}")
        print(f"    Max BAF value     : {max_baf:.3f}")

    print(f"\n  Subject '{subj}' complete in {(time.time()-t_total)/60:.1f} minutes")


def main():
    print("Pipeline: Step 4 (KLT Stabilization) + Step 5 (BAF) + Step 6 (BAP)")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Subjects: {SUBJECTS}")
    print(f"BAF window: {BAF_WINDOW} frames (~{BAF_WINDOW/25:.0f} seconds at 25 fps)")

    available = [s for s in SUBJECTS if (DATASET_DIR / f"{s}_roi").exists()]
    missing   = [s for s in SUBJECTS if s not in available]

    if missing:
        print(f"WARNING: No _roi folder for: {missing}")
    if not available:
        print("ERROR: No _roi folders found. Run step3_roi.py first.")
        sys.exit(1)

    print(f"\nProcessing {len(available)} subjects: {available}")

    overall = time.time()
    for subj in available:
        try:
            process_subject(subj)
        except Exception:
            print(f"\nERROR processing {subj}:")
            traceback.print_exc()

    total = (time.time() - overall) / 60
    print("")
    print("=" * 64)
    print(f"  ALL DONE in {total:.1f} minutes")
    print("")
    print("  Output folders per subject:")
    for subj in available:
        s = DATASET_DIR / f"{subj}_stable"
        b = DATASET_DIR / f"{subj}_baf"
        p = DATASET_DIR / f"{subj}_bap"
        ns = len(list(s.glob("*.png"))) if s.exists() else 0
        nb = len(list(b.glob("*.png"))) if b.exists() else 0
        np_ = len(list(p.glob("*.png"))) if p.exists() else 0
        colour_ok = (DATASET_DIR / f"{subj}_bap_mean_colour.png").exists()
        print(f"    {subj:10s}  stable={ns:5d}  baf={nb:5d}  bap={np_:5d}  "
              f"heatmap={'OK' if colour_ok else 'MISSING'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
