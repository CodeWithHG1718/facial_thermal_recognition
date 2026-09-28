"""
gt_comparison.py
================
Compares our thermal-camera breathing rate estimates against
ground-truth chest-belt measurements from the BIOPAC CSV files.

CSV format:
  Heartbeat  : timestamp, instantaneous_HR_in_BPM
  Respiration: timestamp, breath_marker  (0 = no event, nonzero = breath peak)

Usage: python gt_comparison.py
"""

import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from scipy import signal as scipy_signal

# ---- Config -----------------------------------------------------------------
DATASET_DIR = Path(r"c:\Users\arpit\OneDrive\Desktop\dataset")
FPS         = 25.0

SUBJECTS = ["Joao", "anestis", "Claudio", "Manuel", "Jaime"]

# Our pipeline results (Step 9)
OUR_RESULTS = {
    "Joao":    {"bpm_peaks": 10.7, "bpm_fft": 11.7, "regularity": 0.731, "snr": 0.608, "n_peaks": 20, "duration_s": 112.5},
    "anestis": {"bpm_peaks":  5.7, "bpm_fft": 11.7, "regularity": 0.717, "snr": 0.601, "n_peaks": 10, "duration_s": 104.6},
    "Claudio": {"bpm_peaks":  7.9, "bpm_fft": 11.7, "regularity": 0.673, "snr": 0.701, "n_peaks": 14, "duration_s": 106.3},
    "Manuel":  {"bpm_peaks":  6.7, "bpm_fft": 11.7, "regularity": 0.204, "snr": 0.601, "n_peaks": 11, "duration_s":  98.8},
    "Jaime":   {"bpm_peaks":  8.7, "bpm_fft": 11.7, "regularity": 0.560, "snr": 0.638, "n_peaks": 14, "duration_s":  96.5},
}
# -----------------------------------------------------------------------------


def parse_timestamp(ts_str):
    """Parse '20161122-14:09:37.471' -> datetime."""
    return datetime.strptime(ts_str, "%Y%m%d-%H:%M:%S.%f")


def load_heartbeat(path):
    """
    Load heartbeat CSV. Each row: timestamp, BPM.
    Returns (timestamps_sec, bpm_values) relative to first timestamp.
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rows.append(row)
    if not rows:
        return np.array([]), np.array([])

    t0  = parse_timestamp(rows[0][0])
    ts  = np.array([(parse_timestamp(r[0]) - t0).total_seconds() for r in rows])
    bpm = np.array([float(r[1]) for r in rows])
    return ts, bpm


def load_respiration(path):
    """
    Load respiration CSV. Each row: timestamp, marker.
    Returns (timestamps_sec, marker_values) relative to first timestamp.
    The nonzero markers denote breath events (peak/onset from chest belt).
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rows.append(row)
    if not rows:
        return np.array([]), np.array([])

    t0     = parse_timestamp(rows[0][0])
    ts     = np.array([(parse_timestamp(r[0]) - t0).total_seconds() for r in rows])
    marker = np.array([float(r[1]) for r in rows])
    return ts, marker


def gt_breathing_rate(ts, marker, subject):
    """
    Derive GT breathing rate from chest-belt breath markers.

    Strategy:
      - Find all frames where marker > 0  -> these are breath events
      - Compute inter-event intervals
      - If intervals cluster into two groups (inhale + exhale), use every-other event
      - Otherwise treat each nonzero as one breath cycle marker
      - Report BPM from median interval and from total count / total duration
    """
    event_ts = ts[marker > 0]
    n_events = len(event_ts)
    duration = ts[-1] - ts[0]

    if n_events == 0:
        return {"bpm_count": 0, "bpm_interval": 0, "n_events": 0,
                "mean_interval_s": 0, "std_interval_s": 0}

    # Inter-event intervals
    intervals = np.diff(event_ts)

    # Check if there are two types of intervals (paired inhale+exhale events)
    # by looking at whether intervals have a bimodal distribution
    short_thr = 1.0   # intervals < 1s likely = inhale-exhale pair of same breath
    short_intervals = intervals[intervals < short_thr]
    long_intervals  = intervals[intervals >= short_thr]

    # Decision: if >30% of intervals are very short, events come in pairs
    if len(short_intervals) > 0.3 * len(intervals) and len(long_intervals) > 0:
        # Use only long intervals (between distinct breaths)
        breath_intervals = long_intervals
        n_breaths        = len(long_intervals) + 1
    else:
        breath_intervals = intervals
        n_breaths        = n_events

    # BPM from total count
    bpm_count    = n_breaths / duration * 60.0

    # BPM from median interval
    if len(breath_intervals) > 0:
        med_interval     = float(np.median(breath_intervals))
        std_interval     = float(np.std(breath_intervals))
        bpm_interval     = 60.0 / med_interval if med_interval > 0 else 0
    else:
        med_interval = std_interval = bpm_interval = 0.0

    print(f"    GT Resp  : {n_events} events in {duration:.1f}s")
    print(f"    Short intervals (<1s): {len(short_intervals)}  |  Long: {len(long_intervals)}")
    print(f"    Breath count : {n_breaths}  -> BPM(count)={bpm_count:.1f}")
    print(f"    Med interval : {med_interval:.2f}s -> BPM(interval)={bpm_interval:.1f}")

    return {
        "bpm_count":      bpm_count,
        "bpm_interval":   bpm_interval,
        "n_events":       n_events,
        "n_breaths":      n_breaths,
        "duration_s":     duration,
        "mean_interval_s": med_interval,
        "std_interval_s":  std_interval,
        "event_ts":       event_ts,
    }


def compare_subject(subj):
    """Load GT CSVs for subject and compare against our estimates."""
    hb_path  = next(DATASET_DIR.glob(f"*{subj}_heartbeat.csv"),  None)
    rs_path  = next(DATASET_DIR.glob(f"*{subj}_respiration.csv"), None)

    if not hb_path or not rs_path:
        print(f"  WARNING: CSV files not found for {subj}")
        return None

    print(f"\n  === {subj} ===")

    # Heartbeat
    hb_ts, hb_bpm = load_heartbeat(hb_path)
    gt_hr_mean = float(hb_bpm.mean()) if len(hb_bpm) > 0 else 0
    gt_hr_std  = float(hb_bpm.std())  if len(hb_bpm) > 0 else 0
    print(f"    GT Heart Rate : {gt_hr_mean:.1f} +/- {gt_hr_std:.1f} BPM")

    # Respiration
    rs_ts, rs_marker = load_respiration(rs_path)
    gt_resp = gt_breathing_rate(rs_ts, rs_marker, subj)

    # Our estimates
    our = OUR_RESULTS[subj]

    # Compute errors against both GT BPM estimates
    gt_bpm = gt_resp["bpm_count"]    # primary GT (count-based)
    gt_bpm_alt = gt_resp["bpm_interval"]  # secondary GT (interval-based)

    err_peaks = our["bpm_peaks"] - gt_bpm
    err_fft   = our["bpm_fft"]   - gt_bpm
    abs_err_peaks = abs(err_peaks)
    abs_err_fft   = abs(err_fft)
    pct_err_peaks = abs_err_peaks / max(gt_bpm, 1e-9) * 100
    pct_err_fft   = abs_err_fft   / max(gt_bpm, 1e-9) * 100

    print(f"\n    Our BPM (peaks) : {our['bpm_peaks']:.1f}")
    print(f"    Our BPM (FFT)   : {our['bpm_fft']:.1f}")
    print(f"    GT BPM (count)  : {gt_bpm:.1f}")
    print(f"    GT BPM (interval): {gt_bpm_alt:.1f}")
    print(f"\n    Error (peaks vs GT) : {err_peaks:+.1f} BPM  ({pct_err_peaks:.1f}%)")
    print(f"    Error (FFT vs GT)   : {err_fft:+.1f} BPM  ({pct_err_fft:.1f}%)")

    return {
        "subj":          subj,
        "gt_hr":         gt_hr_mean,
        "gt_hr_std":     gt_hr_std,
        "gt_bpm":        gt_bpm,
        "gt_bpm_alt":    gt_bpm_alt,
        "gt_n_breaths":  gt_resp.get("n_breaths", 0),
        "gt_duration":   gt_resp.get("duration_s", 0),
        "our_bpm_peaks": our["bpm_peaks"],
        "our_bpm_fft":   our["bpm_fft"],
        "our_n_peaks":   our["n_peaks"],
        "our_duration":  our["duration_s"],
        "regularity":    our["regularity"],
        "snr":           our["snr"],
        "err_peaks":     err_peaks,
        "err_fft":       err_fft,
        "abs_err_peaks": abs_err_peaks,
        "abs_err_fft":   abs_err_fft,
        "pct_err_peaks": pct_err_peaks,
        "pct_err_fft":   pct_err_fft,
        "hb_ts":         hb_ts,
        "hb_bpm":        hb_bpm,
        "rs_ts":         rs_ts,
        "rs_marker":     rs_marker,
        "event_ts":      gt_resp.get("event_ts", np.array([])),
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_comparison(all_res):
    """Main comparison figure: 4-panel comparison + per-subject waveform row."""
    n = len(all_res)
    subjects      = [r["subj"]          for r in all_res]
    gt_bpm        = [r["gt_bpm"]        for r in all_res]
    our_bpm_peaks = [r["our_bpm_peaks"] for r in all_res]
    our_bpm_fft   = [r["our_bpm_fft"]  for r in all_res]
    gt_hr         = [r["gt_hr"]         for r in all_res]
    abs_err_pk    = [r["abs_err_peaks"] for r in all_res]
    abs_err_fft   = [r["abs_err_fft"]  for r in all_res]
    pct_err_pk    = [r["pct_err_peaks"] for r in all_res]

    BG  = "#0F172A"
    BG2 = "#1E293B"
    C_BLUE   = "#3B82F6"
    C_GREEN  = "#10B981"
    C_AMBER  = "#F59E0B"
    C_RED    = "#EF4444"
    C_PURPLE = "#8B5CF6"
    C_CYAN   = "#06B6D4"
    C_BORDER = "#334155"

    def style(ax, title, xl="", yl=""):
        ax.set_facecolor(BG2)
        ax.set_title(title, color="#93C5FD", fontsize=9, pad=5)
        ax.set_xlabel(xl, color="gray", fontsize=8)
        ax.set_ylabel(yl, color="gray", fontsize=8)
        ax.tick_params(colors="gray", labelsize=7.5)
        ax.spines[:].set_color(C_BORDER)
        ax.grid(True, axis="y", color=C_BORDER, linewidth=0.4, alpha=0.6)

    # ── Figure 1: Overview comparison (4 panels) ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig.suptitle("Ground Truth vs Our Model -- Breathing Rate Comparison",
                 color="white", fontsize=13, fontweight="bold")

    x = np.arange(n)
    w = 0.25

    # Panel A: BPM comparison (grouped bar)
    ax = axes[0, 0]
    ax.bar(x - w, gt_bpm,        width=w, label="GT (chest belt)", color=C_GREEN,  alpha=0.88, edgecolor=BG2)
    ax.bar(x,     our_bpm_peaks, width=w, label="Ours (peak det.)", color=C_BLUE,  alpha=0.88, edgecolor=BG2)
    ax.bar(x + w, our_bpm_fft,   width=w, label="Ours (FFT)",       color=C_PURPLE, alpha=0.88, edgecolor=BG2)
    for i in range(n):
        ax.text(i-w, gt_bpm[i]+0.2,        f"{gt_bpm[i]:.1f}",        ha="center", color="white", fontsize=6.5)
        ax.text(i,   our_bpm_peaks[i]+0.2, f"{our_bpm_peaks[i]:.1f}", ha="center", color="white", fontsize=6.5)
        ax.text(i+w, our_bpm_fft[i]+0.2,  f"{our_bpm_fft[i]:.1f}",  ha="center", color="white", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(subjects, color="white", fontsize=8)
    ax.legend(fontsize=7.5, labelcolor="white", facecolor=BG, edgecolor=C_BORDER, loc="upper right")
    style(ax, "A. Breathing Rate: GT vs Our Estimates", yl="BPM")
    ax.set_ylim(0, max(max(gt_bpm), max(our_bpm_fft)) * 1.35)

    # Panel B: Absolute error
    ax = axes[0, 1]
    colors_e = [C_GREEN if e < 2 else C_AMBER if e < 5 else C_RED for e in abs_err_pk]
    bars = ax.bar(x - 0.2, abs_err_pk,  width=0.35, color=colors_e, edgecolor=BG2,
                  label="Peak det. |error|")
    ax.bar(x + 0.2, abs_err_fft, width=0.35, color=C_PURPLE, edgecolor=BG2,
           alpha=0.8, label="FFT |error|")
    ax.axhline(2, color=C_GREEN, linestyle="--", linewidth=0.8, alpha=0.7, label="2 BPM threshold")
    ax.axhline(5, color=C_RED,   linestyle="--", linewidth=0.8, alpha=0.7, label="5 BPM threshold")
    for i, (ep, ef) in enumerate(zip(abs_err_pk, abs_err_fft)):
        ax.text(i-0.2, ep+0.1, f"{ep:.1f}", ha="center", color="white", fontsize=6.5)
        ax.text(i+0.2, ef+0.1, f"{ef:.1f}", ha="center", color="white", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(subjects, color="white", fontsize=8)
    ax.legend(fontsize=7, labelcolor="white", facecolor=BG, edgecolor=C_BORDER)
    style(ax, "B. Absolute BPM Error vs Ground Truth", yl="|Error| (BPM)")

    # Panel C: Ground truth heart rate (reference, we didn't estimate this)
    ax = axes[1, 0]
    ax.bar(x, gt_hr, color=C_RED, edgecolor=BG2, alpha=0.85, label="GT HR (chest belt)")
    for i, v in enumerate(gt_hr):
        ax.text(i, v+0.5, f"{v:.0f}", ha="center", color="white", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(subjects, color="white", fontsize=8)
    ax.set_ylim(0, max(gt_hr)*1.3)
    style(ax, "C. Ground Truth Heart Rate (Reference Only)", yl="BPM (HR)")
    ax.legend(fontsize=7.5, labelcolor="white", facecolor=BG, edgecolor=C_BORDER)

    # Panel D: Scatter GT vs our_bpm_peaks
    ax = axes[1, 1]
    ax.set_facecolor(BG2)
    all_vals = gt_bpm + our_bpm_peaks + our_bpm_fft
    lims = [min(all_vals)-1, max(all_vals)+1]
    ax.plot(lims, lims, color="gray", linewidth=1, linestyle="--", alpha=0.5, label="Perfect agree")
    cmap = plt.cm.plasma
    for i, r in enumerate(all_res):
        c = cmap(i/n)
        ax.scatter(r["gt_bpm"], r["our_bpm_peaks"], s=90, color=c,
                   edgecolors="white", linewidths=0.5, zorder=5)
        ax.scatter(r["gt_bpm"], r["our_bpm_fft"],   s=90, color=c,
                   marker="^", edgecolors="white", linewidths=0.5, zorder=5)
        ax.annotate(r["subj"], (r["gt_bpm"], r["our_bpm_peaks"]),
                    textcoords="offset points", xytext=(5, 3),
                    color="white", fontsize=6.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    # Pearson r
    if n >= 3:
        r_pk  = float(np.corrcoef(gt_bpm, our_bpm_peaks)[0,1])
        r_fft = float(np.corrcoef(gt_bpm, our_bpm_fft)[0,1])
        ax.set_title(
            f"D. Scatter: GT vs Ours  (r_peak={r_pk:.2f}, r_fft={r_fft:.2f})",
            color="#93C5FD", fontsize=8.5, pad=5)
    else:
        style(ax, "D. Scatter: GT vs Ours")
    ax.set_xlabel("GT Breathing Rate (BPM)", color="gray", fontsize=8)
    ax.set_ylabel("Our Estimate (BPM)",      color="gray", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7.5)
    ax.spines[:].set_color(C_BORDER)
    ax.grid(True, color=C_BORDER, linewidth=0.4, alpha=0.6)
    from matplotlib.lines import Line2D
    legend_e = [
        Line2D([0],[0], marker="o", color="w", markersize=7, label="Peak det.", markerfacecolor="gray"),
        Line2D([0],[0], marker="^", color="w", markersize=7, label="FFT",       markerfacecolor="gray"),
    ]
    ax.legend(handles=legend_e, fontsize=7, labelcolor="white",
              facecolor=BG, edgecolor=C_BORDER)

    plt.tight_layout()
    out1 = DATASET_DIR / "gt_comparison_overview.png"
    plt.savefig(str(out1), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: gt_comparison_overview.png")

    # ── Figure 2: Per-subject respiration signal + our event markers ──────────
    fig2, axs = plt.subplots(n, 1, figsize=(14, 3.2*n), facecolor=BG,
                             sharex=False)
    if n == 1:
        axs = [axs]
    fig2.suptitle("Per-Subject: Chest-Belt Breath Events vs Our Detected Cycles",
                  color="white", fontsize=12, fontweight="bold")

    for idx, r in enumerate(all_res):
        ax = axs[idx]
        ax.set_facecolor(BG2)

        # Plot respiration marker positions as vertical lines
        ev = r["event_ts"]
        dur = r["rs_ts"][-1] if len(r["rs_ts"]) > 0 else r["our_duration"]
        for t_ev in ev:
            ax.axvline(t_ev, color=C_GREEN, linewidth=0.7, alpha=0.7)

        # Mark our duration range
        ax.axvspan(0, r["our_duration"], color=C_BLUE, alpha=0.05)

        # Add BPM labels
        ax.text(0.01, 0.85,
                f"GT: {r['gt_bpm']:.1f} BPM  ({r['gt_n_breaths']} breaths)",
                transform=ax.transAxes, color=C_GREEN, fontsize=8,
                fontweight="bold")
        ax.text(0.01, 0.68,
                f"Our (peaks): {r['our_bpm_peaks']:.1f} BPM  |  "
                f"Our (FFT): {r['our_bpm_fft']:.1f} BPM",
                transform=ax.transAxes, color=C_BLUE, fontsize=8)
        ax.text(0.01, 0.51,
                f"Abs error: {r['abs_err_peaks']:.1f} BPM ({r['pct_err_peaks']:.0f}%)  |  "
                f"Regularity: {r['regularity']:.3f}  SNR: {r['snr']*100:.0f}%",
                transform=ax.transAxes, color=C_AMBER, fontsize=7.5)

        ax.set_xlim(0, dur)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(r["subj"], color="white", fontsize=9, rotation=0,
                      labelpad=55, va="center")
        ax.tick_params(axis="x", colors="gray", labelsize=7.5)
        ax.spines[:].set_color(C_BORDER)
        if idx == n-1:
            ax.set_xlabel("Time (seconds)", color="gray", fontsize=8)
        ax.set_title(
            f"{r['subj']}  -- Green lines = chest-belt breath events  "
            f"| Blue region = thermal video duration",
            color="#93C5FD", fontsize=8, pad=3)

    plt.tight_layout()
    out2 = DATASET_DIR / "gt_comparison_timeline.png"
    plt.savefig(str(out2), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: gt_comparison_timeline.png")


def print_and_save_report(all_res):
    """Print final evaluation table and save to text file."""
    n      = len(all_res)
    gt_bpm = [r["gt_bpm"]        for r in all_res]
    pk_bpm = [r["our_bpm_peaks"] for r in all_res]
    ff_bpm = [r["our_bpm_fft"]   for r in all_res]
    mae_pk = float(np.mean([r["abs_err_peaks"] for r in all_res]))
    mae_ff = float(np.mean([r["abs_err_fft"]   for r in all_res]))
    mpe_pk = float(np.mean([r["pct_err_peaks"] for r in all_res]))

    if n >= 3:
        r_pk  = float(np.corrcoef(gt_bpm, pk_bpm)[0,1])
        r_fft = float(np.corrcoef(gt_bpm, ff_bpm)[0,1])
    else:
        r_pk = r_fft = float("nan")

    lines = [
        "=" * 84,
        "  GROUND TRUTH COMPARISON REPORT",
        "  Thermal Camera Breathing Estimation vs BIOPAC Chest Belt",
        "=" * 84,
        "",
        f"  {'Subject':<10} {'GT BPM':>8} {'GT HR':>7} {'Our(peak)':>10} "
        f"{'Our(FFT)':>9} {'Err(pk)':>9} {'Err%':>7} {'Regularity':>11} {'SNR':>6}",
        "  " + "-"*80,
    ]
    for r in all_res:
        lines.append(
            f"  {r['subj']:<10} {r['gt_bpm']:>8.1f} {r['gt_hr']:>7.0f} "
            f"{r['our_bpm_peaks']:>10.1f} {r['our_bpm_fft']:>9.1f} "
            f"{r['err_peaks']:>+9.1f} {r['pct_err_peaks']:>6.0f}% "
            f"{r['regularity']:>11.3f} {r['snr']*100:>5.0f}%"
        )
    lines += [
        "  " + "-"*80,
        f"  {'MAE (peak)':>20} : {mae_pk:.2f} BPM",
        f"  {'MAE (FFT)':>20} : {mae_ff:.2f} BPM",
        f"  {'Mean % err (peak)':>20} : {mpe_pk:.1f}%",
        f"  {'Pearson r (peak)':>20} : {r_pk:.3f}",
        f"  {'Pearson r (FFT)':>20} : {r_fft:.3f}",
        "",
        "  INTERPRETATION:",
        "  - GT BPM: chest-belt breath event count / total duration * 60",
        "  - Our (peak): bandpass-filtered waveform peak detection",
        "  - Our (FFT) : dominant frequency in 0.1-0.5 Hz band",
        "  - No ground-truth physiological reference for heart rate estimation",
        "    (heart rate not estimated by our thermal pipeline)",
        "=" * 84,
    ]

    report = "\n".join(lines)
    print(report)
    out = DATASET_DIR / "gt_comparison_report.txt"
    with open(str(out), "w") as f:
        f.write(report)
    print(f"\n  Report saved: gt_comparison_report.txt")
    return mae_pk, mae_ff, r_pk, r_fft


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Ground Truth Comparison: Thermal Pipeline vs BIOPAC Chest Belt")
    print(f"Dataset: {DATASET_DIR}")

    all_res = []
    for subj in SUBJECTS:
        res = compare_subject(subj)
        if res:
            all_res.append(res)

    if not all_res:
        print("ERROR: No results computed.")
        sys.exit(1)

    print("\n" + "="*66)
    print("  GENERATING COMPARISON PLOTS...")
    plot_comparison(all_res)

    print("\n" + "="*66)
    print("  FINAL EVALUATION REPORT")
    print("="*66)
    mae_pk, mae_ff, r_pk, r_fft = print_and_save_report(all_res)

    print(f"\n  Output files:")
    for fname in ["gt_comparison_overview.png",
                  "gt_comparison_timeline.png",
                  "gt_comparison_report.txt"]:
        fp = DATASET_DIR / fname
        print(f"    {'OK' if fp.exists() else 'MISSING':6s}  {fname}")


if __name__ == "__main__":
    main()
