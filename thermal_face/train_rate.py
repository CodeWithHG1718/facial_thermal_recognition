"""CNN and CNN + GRU pilots with leave-one-person-out evaluation.

This is a research baseline, not a validated clinical or identity model. Train
only after auditing the labels and extracting traces. Each held-out person's
windows are entirely absent from that fold's training set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from time import perf_counter
from pathlib import Path

from .windows import load_windows


def resolve_device(device="auto"):
    """An explicit CUDA request must never silently fall back to CPU."""
    import torch

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cpu":
        return torch.device("cpu")
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable. Install CUDA-enabled PyTorch and check the NVIDIA driver.")
    return torch.device("cuda", torch.cuda.current_device()) if available else torch.device("cpu")


def execution_environment(device="auto"):
    import platform
    import numpy as np
    import torch

    selected = resolve_device(device)
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "requested_device": device,
        "device": str(selected),
        "device_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else "CPU",
        "cuda_runtime": torch.version.cuda,
    }


def fit_predict(train, test_x, epochs=40, seed=7, architecture="cnn-gru", device="auto", verbose=False):
    """Fit one fold using training labels only; predict unlabeled test inputs."""
    import numpy as np
    import torch
    from torch import nn
    from .models import CNNGRURateNet, CNNRateNet

    if epochs < 1 or architecture not in ("cnn", "cnn-gru"):
        raise ValueError("Positive epochs and a supported architecture are required")
    selected_device = resolve_device(device)
    started = perf_counter()
    if verbose:
        print(f"  {architecture}: initializing model and tensors on {selected_device}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = (CNNGRURateNet if architecture == "cnn-gru" else CNNRateNet)().to(selected_device)
    train_x = torch.from_numpy(np.stack([x for x, _ in train])).to(selected_device)
    train_y = torch.from_numpy(np.array([y for _, y in train], dtype=np.float32)).to(selected_device)
    test_x = torch.from_numpy(np.asarray(test_x, dtype=np.float32)).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        show_progress = verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs)
        if show_progress:
            epoch_loss = torch.zeros((), device=selected_device)
        model.train()
        # Keep CPU shuffle generation consistent across CPU and CUDA runs.
        order = torch.randperm(len(train_x)).to(selected_device)
        for ids in order.split(16):
            prediction = model(train_x[ids])
            loss = nn.functional.smooth_l1_loss(prediction, train_y[ids], beta=3.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if show_progress:
                epoch_loss += loss.detach() * len(ids)
        if show_progress:
            mean_loss = epoch_loss.item() / len(train_x)
            print(f"  {architecture}: epoch {epoch}/{epochs} | train loss {mean_loss:.4f} | "
                  f"{perf_counter() - started:.1f}s elapsed", flush=True)
    model.eval()
    with torch.no_grad():
        predicted = model(test_x).cpu().numpy()
    return predicted, sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(folder: Path, out: Path, epochs: int = 40, seed: int = 7,
        architecture: str = "cnn-gru", device: str = "auto", verbose=False) -> dict:
    import numpy as np

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if architecture not in ("cnn", "cnn-gru"):
        raise ValueError(f"Unknown architecture: {architecture}")
    environment = execution_environment(device)
    if verbose:
        print(f"Device: {environment['device']} ({environment['device_name']})", flush=True)
    metric_name = architecture.replace("-", "_")
    examples = load_windows(folder)
    people = [person for person, samples in examples.items() if samples]
    if len(people) < 3:
        raise RuntimeError("At least three aligned, nonempty subjects are needed")

    fold_results = []
    for fold_index, held_out in enumerate(people, 1):
        train = [sample for person in people if person != held_out for sample in examples[person]]
        test = examples[held_out]
        if verbose:
            print(f"Fold {fold_index}/{len(people)} | held out: {held_out} | "
                  f"train: {len(train)} windows | test: {len(test)} windows", flush=True)
        test_y = np.array([y for _, y in test])
        predicted, parameter_count = fit_predict(
            train, np.stack([x for x, _ in test]), epochs, seed, architecture, device, verbose=verbose,
        )
        baseline = float(np.median(np.array([y for _, y in train])))
        fold_results.append({
            "held_out": held_out,
            "train_subjects": [p for p in people if p != held_out],
            "test_windows": len(test),
            f"{metric_name}_mae_bpm": round(float(np.mean(np.abs(predicted - test_y))), 3),
            "train_median_baseline_mae_bpm": round(float(np.mean(np.abs(baseline - test_y))), 3),
            "mean_reference_bpm": round(float(np.mean(test_y)), 3),
            "mean_predicted_bpm": round(float(np.mean(predicted)), 3),
            "reference_bpm": test_y.tolist(),
            "predicted_bpm": predicted.tolist(),
        })
    report = {
        "status": "pilot_only_one_recording_per_person",
        "protocol": "leave-one-subject-out; 20s windows/5s stride; marker-1 median interval label",
        "architecture": architecture,
        "environment": environment,
        "epochs": epochs,
        "seed": seed,
        "trainable_parameters": parameter_count,
        "independent_test_people": len(people),
        "excluded_subjects": {"anestis": "unresolved frame/reference alignment"},
        "folds": fold_results,
        f"macro_{metric_name}_mae_bpm": round(float(np.mean([r[f"{metric_name}_mae_bpm"] for r in fold_results])), 3),
        "macro_baseline_mae_bpm": round(float(np.mean([r["train_median_baseline_mae_bpm"] for r in fold_results])), 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=Path("artifacts/traces"))
    parser.add_argument("--out", type=Path, help="Default: artifacts/rate_<architecture>_pilot.json")
    parser.add_argument("--architecture", choices=("cnn", "cnn-gru"), default="cnn-gru")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto selects CUDA when available; cuda requires a working GPU")
    args = parser.parse_args()
    model_name = args.architecture.replace("-", "_")
    out = args.out or Path(f"artifacts/rate_{model_name}_pilot.json")
    print("Starting training; initializing device and loading traces...", flush=True)
    try:
        run(args.traces, out, args.epochs, args.seed, args.architecture, args.device, verbose=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Rerun the command to start again.", file=sys.stderr, flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
