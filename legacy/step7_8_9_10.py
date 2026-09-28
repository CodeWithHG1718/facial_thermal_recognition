"""
step7_8_9_10.py
===============
Step 7  : MRF + ICM segmentation on time-averaged BAP map -> binary BAFR mask
Step 8  : Extract breathing waveform from BAFR-masked stable frames
Step 9  : Bandpass filter + peak detection -> breathing rate (BPM)
Step 10 : Evaluation -- SNR, regularity, cross-subject summary table + plots

Inputs  : <subject>_bap_mean.png   (from Step 6)
          <subject>_stable/        (from Step 4)
Outputs : <subject>_mask.png              BAFR binary mask
          <subject>_mask_overlay.png      Mask overlaid on thermal frame
          <subject>_waveform.png          Raw + filtered breathing waveform plot
          results_summary.png             Cross-subject evaluation table plot
          results_summary.txt             Plain-text evaluation results

Usage   : python step7_8_9_10.py
"""

import sys, time, traceback
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display window needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal, ndimage
from pathlib import Path

# ---- Configuration ----------------------------------------------------------
DATASET_DIR = Path(r"c:\Users\arpit\OneDrive\Desktop\dataset")
SUBJECTS    = ["Joao", "anestis", "Claudio", "Manuel", "Jaime"]

FPS             = 25.0    # frames per second
MRF_LAMBDA      = 0.6     # MRF smoothness weight
MRF_ITERS       = 30      # ICM iterations
BP_LOW_HZ       = 0.1     # bandpass low  (6 bpm)
BP_HIGH_HZ      = 0.5     # bandpass high (30 bpm)
BP_ORDER        = 4       # Butterworth filter order
PEAK_MIN_DIST   = int(FPS * 1.5)  # min 1.5 s between breathing peaks
PEAK_PROM_FRAC  = 0.4     # peak prominence >= 40% of waveform std
# -----------------------------------------------------------------------------


# =============================================================================
# STEP 7 -- MRF + ICM SEGMENTATION
# =============================================================================

def mrf_icm(bap_float, lambda_s=MRF_LAMBDA, n_iter=MRF_ITERS):
    """
    2-class MRF segmentation using vectorised ICM.
    Input  : bap_float  -- float32 array in [0,1], shape (H,W)
    Output : labels     -- uint8 binary mask  1=BAFR  0=background
    """
    H, W = bap_float.shape
    neighbor_k = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)
    n_neigh    = ndimage.convolve(np.ones((H,W),dtype=np.float32),
                                  neighbor_k, mode='nearest')

    # Initialise: above-median pixels -> class 1 (BAFR)
    labels = (bap_float > np.median(bap_float)).astype(np.float32)

    mu    = np.zeros(2, dtype=np.float64)
    sigma = np.zeros(2, dtype=np.float64)

    for itr in range(n_iter):
        # Update class Gaussian parameters
        for cls in range(2):
            m = labels == cls
            if m.sum() > 5:
                mu[cls]    = float(bap_float[m].mean())
                sigma[cls] = max(float(bap_float[m].std()), 1e-3)
            else:
                mu[cls]    = 0.25 + cls * 0.5
                sigma[cls] = 0.1

        # Data energy for each class (vectorised, shape H x W)
        def data_e(cls):
            return (0.5 * np.log(2*np.pi*sigma[cls]**2)
                    + 0.5 * (bap_float - mu[cls])**2 / sigma[cls]**2)

        e_d0 = data_e(0)
        e_d1 = data_e(1)

        # Smooth energy: count neighbours with different label
        sum1    = ndimage.convolve(labels, neighbor_k, mode='nearest')
        e_s0    = lambda_s * sum1               # label=0: neighbours with label=1
        e_s1    = lambda_s * (n_neigh - sum1)   # label=1: neighbours with label=0

        new_labels = ((e_d1 + e_s1) < (e_d0 + e_s0)).astype(np.float32)

        if np.array_equal(new_labels, labels):
            print(f"      ICM converged at iteration {itr+1}")
            break
        labels = new_labels

    return labels.astype(np.uint8)


def segment_subject(subj):
    """Step 7: Load BAP mean map, run MRF+ICM, save mask + overlay."""
    bap_path  = DATASET_DIR / f"{subj}_bap_mean.png"
    stab_dir  = DATASET_DIR / f"{subj}_stable"

    if not bap_path.exists():
        print(f"  ERROR: {bap_path.name} not found. Run step4_5_6.py first.")
        return None

    print(f"\n  [Step 7] MRF+ICM segmentation for {subj}...")
    t0 = time.time()

    # Load time-averaged BAP map
    bap_u8    = cv2.imread(str(bap_path), cv2.IMREAD_GRAYSCALE)
    bap_float = bap_u8.astype(np.float32) / 255.0
    H, W      = bap_float.shape

    # Run MRF
    mask = mrf_icm(bap_float)

    # Clean up mask: remove small isolated blobs
    mask_clean = mask.copy()
    n_lab, lab_img, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            mask_clean[lab_img == i] = 0

    # Save mask
    mask_path = DATASET_DIR / f"{subj}_mask.png"
    cv2.imwrite(str(mask_path), mask_clean * 255)

    # Save coloured overlay on a stable frame midpoint
    frames = sorted(stab_dir.glob("frame_*.png")) if stab_dir.exists() else []
    if frames:
        mid_frame = cv2.imread(str(frames[len(frames)//2]), cv2.IMREAD_GRAYSCALE)
        overlay   = cv2.applyColorMap(mid_frame, cv2.COLORMAP_BONE)
        # Highlight BAFR region in bright green
        overlay[mask_clean == 1, 0] = 0
        overlay[mask_clean == 1, 1] = 220
        overlay[mask_clean == 1, 2] = 50
        ov_path = DATASET_DIR / f"{subj}_mask_overlay.png"
        cv2.imwrite(str(ov_path), overlay)
        print(f"    Overlay saved: {ov_path.name}")

    bafr_px = int(mask_clean.sum())
    total_px = H * W
    bafr_pct = bafr_px / total_px * 100
    print(f"    BAFR pixels: {bafr_px}/{total_px} ({bafr_pct:.1f}% of ROI)")
    print(f"    Step 7 done in {time.time()-t0:.1f}s")

    return mask_clean


# =============================================================================
# STEP 8 -- WAVEFORM EXTRACTION
# =============================================================================

def extract_waveform(subj, mask):
    """Step 8: Spatial mean of stable frames within BAFR mask -> waveform."""
    stab_dir = DATASET_DIR / f"{subj}_stable"
    frames   = sorted(stab_dir.glob("frame_*.png"))
    if not frames:
        print(f"  ERROR: No stable frames for {subj}")
        return None, None

    n = len(frames)
    print(f"\n  [Step 8] Extracting waveform from {n} frames (mask px = {mask.sum()})...")
    t0 = time.time()

    waveform = np.zeros(n, dtype=np.float64)
    mask_bool = mask.astype(bool)

    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        waveform[i] = float(img[mask_bool].mean())
        if (i+1) % 500 == 0:
            print(f"    Frame {i+1}/{n}  ({time.time()-t0:.1f}s)")

    print(f"    Waveform extracted in {time.time()-t0:.1f}s")
    print(f"    Waveform range: [{waveform.min():.2f}, {waveform.max():.2f}]  "
          f"mean={waveform.mean():.2f}  std={waveform.std():.2f}")

    # Time axis in seconds
    t_axis = np.arange(n) / FPS
    return waveform, t_axis


# =============================================================================
# STEP 9 -- BREATHING RATE ESTIMATION
# =============================================================================

def estimate_breathing_rate(subj, waveform, t_axis):
    """
    Step 9: Bandpass filter waveform, detect peaks, compute BPM.
    Returns dict with results.
    """
    if waveform is None or len(waveform) < 200:
        return None

    print(f"\n  [Step 9] Estimating breathing rate for {subj}...")

    # Detrend (remove linear drift)
    detrended = signal.detrend(waveform, type='linear')

    # Butterworth bandpass filter: 0.1-0.5 Hz (6-30 bpm)
    nyq = FPS / 2.0
    b, a = signal.butter(BP_ORDER, [BP_LOW_HZ/nyq, BP_HIGH_HZ/nyq], btype='band')
    filtered = signal.filtfilt(b, a, detrended)

    # Peak detection
    prominence = PEAK_PROM_FRAC * filtered.std()
    peaks, props = signal.find_peaks(
        filtered,
        distance=PEAK_MIN_DIST,
        prominence=prominence,
    )

    duration_s  = len(waveform) / FPS
    bpm         = len(peaks) / duration_s * 60.0

    # Inter-peak intervals
    if len(peaks) >= 2:
        ipi_s      = np.diff(peaks) / FPS          # seconds
        ipi_mean   = float(ipi_s.mean())
        ipi_std    = float(ipi_s.std())
        regularity  = max(0.0, 1.0 - ipi_std / max(ipi_mean, 0.01))
    else:
        ipi_mean = ipi_std = 0.0
        regularity = 0.0

    # Signal-to-noise ratio: power in breathing band vs total power
    freqs, psd = signal.welch(filtered, fs=FPS, nperseg=min(256, len(filtered)//4))
    band_mask  = (freqs >= BP_LOW_HZ) & (freqs <= BP_HIGH_HZ)
    snr = float(psd[band_mask].sum() / max(psd.sum(), 1e-9))

    # Dominant frequency from PSD peak
    if band_mask.any():
        dom_freq = float(freqs[band_mask][np.argmax(psd[band_mask])])
        dom_bpm  = dom_freq * 60.0
    else:
        dom_freq = 0.0
        dom_bpm  = 0.0

    print(f"    Duration    : {duration_s:.1f}s")
    print(f"    Peaks found : {len(peaks)}")
    print(f"    BPM (peaks) : {bpm:.1f}")
    print(f"    BPM (FFT)   : {dom_bpm:.1f}")
    print(f"    Regularity  : {regularity:.3f}  (1=perfect, 0=irregular)")
    print(f"    Band SNR    : {snr:.3f}")

    return {
        "subj":        subj,
        "n_frames":    len(waveform),
        "duration_s":  duration_s,
        "n_peaks":     len(peaks),
        "bpm_peaks":   bpm,
        "bpm_fft":     dom_bpm,
        "dom_freq_hz": dom_freq,
        "ipi_mean_s":  ipi_mean,
        "ipi_std_s":   ipi_std,
        "regularity":  regularity,
        "band_snr":    snr,
        "waveform":    waveform,
        "filtered":    filtered,
        "t_axis":      t_axis,
        "peaks":       peaks,
        "detrended":   detrended,
    }


# =============================================================================
# STEP 10 -- EVALUATION + VISUALISATION
# =============================================================================

def plot_waveform(res, mask, subj):
    """Save a waveform plot (raw detrended + filtered + peaks)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             facecolor="#0F172A")
    fig.suptitle(f"Breathing Waveform Analysis -- {subj}",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    t   = res["t_axis"]
    raw = res["detrended"]
    flt = res["filtered"]
    pk  = res["peaks"]

    ax_colors = ["#1E293B", "#1E293B", "#1E293B"]

    # Panel 1: Raw detrended waveform
    ax = axes[0]
    ax.set_facecolor(ax_colors[0])
    ax.plot(t, raw, color="#60A5FA", linewidth=0.6, label="Raw (detrended)")
    ax.set_ylabel("Mean Intensity", color="white", fontsize=9)
    ax.set_title("Raw Waveform (detrended, DC removed)", color="#93C5FD",
                 fontsize=9, pad=4)
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#334155")
    ax.grid(True, color="#1E3A5F", linewidth=0.4)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#0F172A",
              edgecolor="#334155")

    # Panel 2: Filtered waveform with peaks
    ax = axes[1]
    ax.set_facecolor(ax_colors[1])
    ax.plot(t, flt, color="#34D399", linewidth=0.9, label="Bandpass filtered (0.1-0.5 Hz)")
    if len(pk) > 0:
        ax.scatter(t[pk], flt[pk], color="#FBBF24", s=40, zorder=5,
                   label=f"Peaks ({len(pk)})")
    ax.set_ylabel("Amplitude", color="white", fontsize=9)
    ax.set_title(f"Filtered Waveform + Peaks  |  BPM = {res['bpm_peaks']:.1f}  |  "
                 f"FFT BPM = {res['bpm_fft']:.1f}",
                 color="#93C5FD", fontsize=9, pad=4)
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#334155")
    ax.grid(True, color="#1E3A5F", linewidth=0.4)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#0F172A",
              edgecolor="#334155")

    # Panel 3: Power spectral density
    ax = axes[2]
    ax.set_facecolor(ax_colors[2])
    freqs, psd = signal.welch(flt, fs=FPS,
                               nperseg=min(256, len(flt)//4))
    bpm_axis = freqs * 60
    ax.fill_between(bpm_axis, psd, color="#7C3AED", alpha=0.4)
    ax.plot(bpm_axis, psd, color="#A78BFA", linewidth=1.0)
    ax.axvspan(BP_LOW_HZ*60, BP_HIGH_HZ*60, alpha=0.1,
               color="#22D3EE", label="Breathing band (6-30 bpm)")
    if res["bpm_fft"] > 0:
        ax.axvline(res["bpm_fft"], color="#FBBF24", linewidth=1.2,
                   linestyle="--", label=f"Peak: {res['bpm_fft']:.1f} bpm")
    ax.set_xlabel("Frequency (bpm)", color="white", fontsize=9)
    ax.set_ylabel("PSD", color="white", fontsize=9)
    ax.set_title("Power Spectral Density", color="#93C5FD", fontsize=9, pad=4)
    ax.set_xlim(0, 60)
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#334155")
    ax.grid(True, color="#1E3A5F", linewidth=0.4)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#0F172A",
              edgecolor="#334155")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = DATASET_DIR / f"{subj}_waveform.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight",
                facecolor="#0F172A")
    plt.close()
    print(f"    Waveform plot saved: {out.name}")


def plot_summary(all_results):
    """Plot cross-subject evaluation table as an image."""
    subjects   = [r["subj"]         for r in all_results]
    bpm_peaks  = [r["bpm_peaks"]    for r in all_results]
    bpm_fft    = [r["bpm_fft"]      for r in all_results]
    regularity = [r["regularity"]   for r in all_results]
    snr        = [r["band_snr"]     for r in all_results]
    n_peaks    = [r["n_peaks"]      for r in all_results]
    duration   = [r["duration_s"]   for r in all_results]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5),
                             facecolor="#0F172A")
    fig.suptitle("Cross-Subject Evaluation -- Steps 7-10",
                 color="white", fontsize=13, fontweight="bold")

    bar_kwargs = dict(color="#2563EB", edgecolor="#60A5FA",
                      linewidth=0.6, alpha=0.85)
    tick_color = "gray"

    def style_ax(ax, title, ylabel):
        ax.set_facecolor("#1E293B")
        ax.set_title(title, color="#93C5FD", fontsize=9, pad=6)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.tick_params(colors=tick_color, labelsize=8)
        ax.spines[:].set_color("#334155")
        ax.grid(True, axis="y", color="#0F172A", linewidth=0.5)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=20, ha="right",
                           fontsize=8, color="white")

    # BPM comparison
    x = np.arange(len(subjects))
    ax = axes[0]
    ax.bar(x - 0.2, bpm_peaks, width=0.35, label="Peak-based",
           color="#2563EB", edgecolor="#60A5FA", linewidth=0.6)
    ax.bar(x + 0.2, bpm_fft,   width=0.35, label="FFT-based",
           color="#059669", edgecolor="#34D399", linewidth=0.6)
    ax.set_facecolor("#1E293B")
    ax.set_title("Breathing Rate (BPM)", color="#93C5FD", fontsize=9, pad=6)
    ax.set_ylabel("Breaths / min", color="white", fontsize=8)
    ax.tick_params(colors=tick_color, labelsize=8)
    ax.spines[:].set_color("#334155")
    ax.grid(True, axis="y", color="#0F172A", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=20, ha="right",
                       fontsize=8, color="white")
    ax.legend(fontsize=7.5, labelcolor="white", facecolor="#0F172A",
              edgecolor="#334155")
    ax.set_ylim(0, max(max(bpm_peaks), max(bpm_fft)) * 1.3)
    for i, (a, b) in enumerate(zip(bpm_peaks, bpm_fft)):
        ax.text(i-0.2, a+0.3, f"{a:.0f}", ha="center", va="bottom",
                color="white", fontsize=7)
        ax.text(i+0.2, b+0.3, f"{b:.0f}", ha="center", va="bottom",
                color="white", fontsize=7)

    # Regularity
    ax = axes[1]
    colors_reg = ["#22C55E" if r >= 0.7 else "#F59E0B" if r >= 0.4
                  else "#EF4444" for r in regularity]
    bars = ax.bar(range(len(subjects)), regularity,
                  color=colors_reg, edgecolor="#475569", linewidth=0.6)
    style_ax(ax, "Waveform Regularity", "Score (0-1)")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.7, color="#22C55E", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.4, color="#F59E0B", linewidth=0.8, linestyle="--", alpha=0.6)
    for i, v in enumerate(regularity):
        ax.text(i, v+0.02, f"{v:.2f}", ha="center", va="bottom",
                color="white", fontsize=7.5)

    # Band SNR
    ax = axes[2]
    ax.bar(range(len(subjects)), [s*100 for s in snr],
           **bar_kwargs)
    style_ax(ax, "Breathing Band SNR", "Signal Power (%)")
    for i, v in enumerate(snr):
        ax.text(i, v*100+0.3, f"{v*100:.1f}%", ha="center", va="bottom",
                color="white", fontsize=7.5)

    # Number of detected cycles
    ax = axes[3]
    ax.bar(range(len(subjects)), n_peaks, color="#7C3AED",
           edgecolor="#A78BFA", linewidth=0.6)
    style_ax(ax, "Detected Breathing Cycles", "Count")
    for i, (np_, dur) in enumerate(zip(n_peaks, duration)):
        ax.text(i, np_+0.3, f"{np_}", ha="center", va="bottom",
                color="white", fontsize=7.5)

    plt.tight_layout()
    out = DATASET_DIR / "results_summary.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight",
                facecolor="#0F172A")
    plt.close()
    print(f"\n  Summary chart saved: results_summary.png")


def save_text_results(all_results):
    """Save plain-text evaluation table."""
    lines = [
        "=" * 78,
        "  PIPELINE EVALUATION RESULTS -- Steps 7 to 10",
        "  Thermal Camera Breathing Rate Estimation",
        "=" * 78,
        "",
        f"  {'Subject':<12} {'BPM(peak)':>10} {'BPM(FFT)':>10} {'Cycles':>8} "
        f"{'Regularity':>12} {'SNR':>8} {'Duration':>10}",
        "  " + "-"*74,
    ]
    for r in all_results:
        lines.append(
            f"  {r['subj']:<12} {r['bpm_peaks']:>10.1f} {r['bpm_fft']:>10.1f} "
            f"{r['n_peaks']:>8d} {r['regularity']:>12.3f} "
            f"{r['band_snr']*100:>7.1f}% {r['duration_s']:>9.1f}s"
        )

    bpms = [r["bpm_peaks"] for r in all_results]
    lines += [
        "  " + "-"*74,
        f"  {'MEAN':<12} {np.mean(bpms):>10.1f}",
        f"  {'STD':<12} {np.std(bpms):>10.1f}",
        "",
        "  NOTE: No ground-truth physiological reference available.",
        "        BPM estimates are based on peak detection and FFT analysis.",
        "        Regularity score: 1.0 = perfectly regular breathing cycle.",
        "        Band SNR: fraction of waveform power in 0.1-0.5 Hz band.",
        "=" * 78,
    ]

    out = DATASET_DIR / "results_summary.txt"
    with open(str(out), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  Text results saved: results_summary.txt")


# =============================================================================
# MAIN
# =============================================================================

def process_subject(subj):
    t0 = time.time()
    print("")
    print("=" * 66)
    print(f"  SUBJECT : {subj}")
    print("=" * 66)

    # Step 7
    mask = segment_subject(subj)
    if mask is None:
        return None

    # Step 8
    waveform, t_axis = extract_waveform(subj, mask)
    if waveform is None:
        return None

    # Step 9
    res = estimate_breathing_rate(subj, waveform, t_axis)
    if res is None:
        return None

    # Step 10 -- per-subject plot
    plot_waveform(res, mask, subj)

    total = time.time() - t0
    print(f"\n  Subject '{subj}' complete in {total/60:.1f} minutes")
    return res


def main():
    print("Pipeline Steps 7-10: MRF Segmentation -> Waveform -> BPM -> Evaluation")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Subjects: {SUBJECTS}")
    print(f"FPS={FPS}  BAP_window={int(FPS*4)}  BPM_range={BP_LOW_HZ*60:.0f}-{BP_HIGH_HZ*60:.0f}")

    available = [s for s in SUBJECTS if (DATASET_DIR/f"{s}_bap_mean.png").exists()]
    if not available:
        print("ERROR: No BAP mean maps found. Run step4_5_6.py first.")
        sys.exit(1)

    print(f"\nProcessing {len(available)} subjects: {available}")

    overall      = time.time()
    all_results  = []

    for subj in available:
        try:
            res = process_subject(subj)
            if res:
                all_results.append(res)
        except Exception:
            print(f"\nERROR processing {subj}:")
            traceback.print_exc()

    # Step 10 -- cross-subject summary
    if all_results:
        print("\n" + "="*66)
        print("  STEP 10 -- Cross-Subject Evaluation")
        print("="*66)
        plot_summary(all_results)
        save_text_results(all_results)

    total_min = (time.time() - overall) / 60
    print(f"\n  Pipeline complete in {total_min:.1f} minutes")
    print(f"\n  Output files in {DATASET_DIR}:")
    for subj in available:
        for suffix in ["_mask.png", "_mask_overlay.png", "_waveform.png"]:
            fp = DATASET_DIR / f"{subj}{suffix}"
            print(f"    {'OK' if fp.exists() else 'MISSING':6s}  {fp.name}")
    for fname in ["results_summary.png", "results_summary.txt"]:
        fp = DATASET_DIR / fname
        print(f"    {'OK' if fp.exists() else 'MISSING':6s}  {fname}")


if __name__ == "__main__":
    main()
