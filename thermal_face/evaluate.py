"""Compare respiration methods on common windows with explicit coverage."""

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .baselines import spectral_rate
from .train_rate import execution_environment, fit_predict
from .windows import EXCLUDED_SUBJECTS, SAMPLE_HZ, load_window_records

METHODS = ("train_median", "spectral", "cnn", "cnn_gru")
CONDITIONS = ("still", "moving", "transition")


def error_metrics(rows, method):
    errors = {}
    for row in rows:
        prediction = row["predictions_bpm"][method]
        if row["eligible"] and prediction is not None:
            errors.setdefault(row["subject"], []).append(abs(prediction - row["reference_bpm"]))
    per_subject = {p: float(np.mean(values)) for p, values in errors.items()}
    return {
        "evaluated_windows": sum(map(len, errors.values())),
        "evaluated_people": len(errors),
        "macro_mae_bpm": float(np.mean(list(per_subject.values()))) if errors else None,
        "pooled_window_mae_bpm": float(np.mean([e for values in errors.values() for e in values])) if errors else None,
        "per_subject_mae_bpm": per_subject,
    }


def summarize(rows):
    eligible = sum(row["eligible"] for row in rows)
    common = [row for row in rows if row["eligible"] and all(
        row["predictions_bpm"][method] is not None for method in METHODS
    )]
    methods = {}
    for method in METHODS:
        metrics = error_metrics(rows, method)
        predicted = metrics["evaluated_windows"]
        methods[method] = metrics | {
            "coverage_of_candidates": predicted / len(rows) if rows else None,
            "coverage_of_eligible": predicted / eligible if eligible else None,
            "abstention_reasons_on_eligible": dict(Counter(
                row["abstention_reasons"][method] for row in rows
                if row["eligible"] and row["predictions_bpm"][method] is None
            )),
        }
    return {
        "candidate_windows": len(rows),
        "signal_valid_windows": sum(row["signal_rejection"] is None for row in rows),
        "reference_valid_windows": sum(row["reference_rejection"] is None for row in rows),
        "eligible_windows": eligible,
        "eligibility_fraction": eligible / len(rows) if rows else None,
        "signal_rejections": dict(Counter(row["signal_rejection"] for row in rows if row["signal_rejection"])),
        "reference_rejections": dict(Counter(row["reference_rejection"] for row in rows if row["reference_rejection"])),
        "methods": methods,
        "common_comparison": {
            "windows": len(common),
            "methods": {method: error_metrics(common, method) for method in METHODS},
        },
    }


def run(traces: Path, out: Path, epochs=40, seed=7, still_until_s=60.0, device="auto", verbose=False):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not traces.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {traces}")
    environment = execution_environment(device)
    if verbose:
        print(f"Device: {environment['device']} ({environment['device_name']})", flush=True)
    records = load_window_records(traces, still_until_s=still_until_s)
    eligible = {subject: [w for w in windows if w.eligible] for subject, windows in records.items()}
    people = [subject for subject, windows in eligible.items() if windows]
    if len(people) < 3:
        raise RuntimeError("At least three nonempty eligible subjects are required")
    rows = []
    lookup = {}
    for subject, windows in records.items():
        for window in windows:
            row = window.metadata() | {
                "predictions_bpm": dict.fromkeys(METHODS),
                "abstention_reasons": dict.fromkeys(METHODS, None if window.eligible else "window_ineligible"),
            }
            rows.append(row)
            lookup[(subject, window.start_s)] = row
    folds = []
    for fold_index, held_out in enumerate(people, 1):
        training_people = [person for person in people if person != held_out]
        train = [(w.x, np.float32(w.reference_bpm)) for person in training_people for w in eligible[person]]
        test = eligible[held_out]
        test_x = np.stack([w.x for w in test])
        median = float(np.median(np.asarray([y for _, y in train], dtype=np.float32)))
        if verbose:
            print(f"Fold {fold_index}/{len(people)} | held out: {held_out} | "
                  f"train: {len(train)} windows | test: {len(test)} windows", flush=True)
        parameter_counts = {}
        for architecture in ("cnn", "cnn-gru"):
            predictions, parameter_counts[architecture] = fit_predict(train, test_x, epochs, seed, architecture, device, verbose=verbose)
            if np.shape(predictions) != (len(test),):
                raise ValueError(f"Unexpected predictions from {architecture}")
            for window, prediction in zip(test, predictions):
                row = lookup[(held_out, window.start_s)]
                method = architecture.replace("-", "_")
                if np.isfinite(prediction):
                    row["predictions_bpm"][method] = float(prediction)
                else:
                    row["abstention_reasons"][method] = "nonfinite_prediction"
        for window in test:
            row = lookup[(held_out, window.start_s)]
            row["predictions_bpm"]["train_median"] = median
            prediction, reason = spectral_rate(window.x)
            row["predictions_bpm"]["spectral"] = prediction
            row["abstention_reasons"]["spectral"] = reason
        folds.append({
            "held_out": held_out, "train_subjects": training_people,
            "train_windows": len(train), "test_windows": len(test),
            "train_median_bpm": median, "trainable_parameters": parameter_counts,
        })
    report = {
        "status": "exploratory_one_recording_per_person",
        "protocol": {
            "split": "leave-one-person-out", "window_s": 20.0, "step_s": 5.0,
            "sample_hz": SAMPLE_HZ, "epochs": epochs, "seed": seed,
            "still_until_s": still_until_s,
            "condition_labels": "assumed from elapsed recording time; boundary-crossing windows are transition",
            "reference": "60 / median marker-1 inter-event interval within each window; at least four events",
            "alignment": "assumed frame/CSV row correspondence; not independently verified",
            "coverage": "predictions / all complete candidate windows; reference screening is included, not deployment availability",
            "primary_comparison": "common_comparison uses only windows with predictions from every method",
            "spectral": "timestamp-resampled 10 Hz traces; linear detrend; Hann taper; mean normalized patch power; 6-45 BPM; 8x zero padding",
        },
        "environment": environment,
        "trace_sha256": {person: hashlib.sha256((traces / f"{person}.npz").read_bytes()).hexdigest() for person in records},
        "independent_test_people": len(people),
        "excluded_subjects": EXCLUDED_SUBJECTS,
        "subjects_without_eligible_windows": [p for p in records if not eligible[p]],
        "folds": folds,
        "overall": summarize(rows),
        "by_condition": {condition: summarize([row for row in rows if row["condition"] == condition]) for condition in CONDITIONS},
        "by_subject": {person: summarize([row for row in rows if row["subject"] == person]) for person in records},
        "windows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("artifacts/traces"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto selects CUDA when available; cuda requires a working GPU")
    parser.add_argument("--still-until-s", type=float, default=60.0,
                        help="Assumed still/moving boundary in elapsed seconds")
    args = parser.parse_args()
    print("Starting evaluation; initializing device and loading traces...", flush=True)
    try:
        report = run(args.traces, args.out, args.epochs, args.seed, args.still_until_s, args.device, verbose=True)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted. Rerun the command to start again.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    summary = report["overall"]
    print(f"Report: {args.out}")
    print(f"Eligible windows: {summary['eligible_windows']}/{summary['candidate_windows']}; "
          f"held-out people: {report['independent_test_people']}")
    for method, metrics in summary["common_comparison"]["methods"].items():
        mae = metrics["macro_mae_bpm"]
        rendered = f"{mae:.3f}" if mae is not None else "unavailable"
        print(f"{method}: common-window macro MAE {rendered} breaths/min")


if __name__ == "__main__":
    main()
