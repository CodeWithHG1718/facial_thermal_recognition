"""Runnable small neural baseline on spectral features from nasal traces.

The four leave-one-person-out folds are exploratory because there is only one
session per person. The input extractor and all standardization are fit on
training people only, and each fold includes a train-median comparator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .train_rate import load_windows


def spectral_features(trace: np.ndarray) -> np.ndarray:
    # 20 seconds at 10 Hz gives 3-breath/min raw FFT spacing.
    centered = trace[:9] - trace[:9].mean(axis=1, keepdims=True)
    taper = np.hanning(centered.shape[1])
    spectrum = np.abs(np.fft.rfft(centered * taper[None, :], axis=1))
    frequencies = np.fft.rfftfreq(centered.shape[1], d=0.1)
    band = (frequencies >= 0.1) & (frequencies <= 0.75)
    selected = spectrum[:, band]
    selected /= np.maximum(selected.sum(axis=1, keepdims=True), 1e-6)
    return np.concatenate([selected.ravel(), [float(trace[9].mean())]]).astype(np.float32)


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """Fit a regularized one-hidden-layer MLP using NumPy only."""
    x_mean = train_x.mean(axis=0)
    x_std = np.maximum(train_x.std(axis=0), 0.05)
    x = (train_x - x_mean) / x_std
    xt = (test_x - x_mean) / x_std
    y_mean = float(train_y.mean())
    y_std = max(float(train_y.std()), 1.0)
    y = ((train_y - y_mean) / y_std)[:, None]
    rng = np.random.default_rng(7)
    w1 = rng.normal(0, 0.06, size=(x.shape[1], 16))
    b1 = np.zeros((1, 16))
    w2 = rng.normal(0, 0.06, size=(16, 1))
    b2 = np.zeros((1, 1))
    params = [w1, b1, w2, b2]
    moments = [np.zeros_like(p) for p in params]
    velocities = [np.zeros_like(p) for p in params]
    for epoch in range(1, 801):
        hidden = np.tanh(x @ w1 + b1)
        prediction = hidden @ w2 + b2
        d_output = 2.0 * (prediction - y) / len(y)
        gradients = [
            x.T @ ((d_output @ w2.T) * (1.0 - hidden**2)) + 0.02 * w1,
            np.sum((d_output @ w2.T) * (1.0 - hidden**2), axis=0, keepdims=True),
            hidden.T @ d_output + 0.02 * w2,
            np.sum(d_output, axis=0, keepdims=True),
        ]
        for i, (param, grad) in enumerate(zip(params, gradients)):
            moments[i] = 0.9 * moments[i] + 0.1 * grad
            velocities[i] = 0.999 * velocities[i] + 0.001 * grad**2
            update = (moments[i] / (1.0 - 0.9**epoch)) / (
                np.sqrt(velocities[i] / (1.0 - 0.999**epoch)) + 1e-8
            )
            param -= 0.002 * update
    return ((np.tanh(xt @ w1 + b1) @ w2 + b2).ravel() * y_std + y_mean)


def run(traces: Path, out: Path) -> dict:
    examples = load_windows(traces)
    people = [person for person, samples in examples.items() if samples]
    if len(people) < 3:
        raise RuntimeError("Need at least three aligned subjects with valid windows")
    folds = []
    for held_out in people:
        train = [sample for person in people if person != held_out for sample in examples[person]]
        test = examples[held_out]
        train_x = np.stack([spectral_features(x) for x, _ in train])
        train_y = np.array([y for _, y in train])
        test_x = np.stack([spectral_features(x) for x, _ in test])
        test_y = np.array([y for _, y in test])
        predicted = fit_predict(train_x, train_y, test_x)
        baseline = float(np.median(train_y))
        folds.append({
            "held_out": held_out,
            "test_windows": len(test),
            "mlp_mae_bpm": round(float(np.mean(np.abs(predicted - test_y))), 3),
            "train_median_mae_bpm": round(float(np.mean(np.abs(baseline - test_y))), 3),
            "reference_mean_bpm": round(float(test_y.mean()), 3),
            "prediction_mean_bpm": round(float(predicted.mean()), 3),
        })
    report = {
        "status": "exploratory_four_people_one_session_each",
        "protocol": "leave-one-person-out; 20-second overlapping windows; no anestis due to alignment mismatch",
        "folds": folds,
        "macro_mlp_mae_bpm": round(float(np.mean([f["mlp_mae_bpm"] for f in folds])), 3),
        "macro_train_median_mae_bpm": round(float(np.mean([f["train_median_mae_bpm"] for f in folds])), 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=Path("artifacts/traces"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/rate_mlp_pilot.json"))
    args = parser.parse_args()
    run(args.traces, args.out)


if __name__ == "__main__":
    main()
