"""Known-frequency, timing, coverage and evaluation integration checks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from thermal_face.baselines import spectral_rate
from thermal_face.evaluate import METHODS, run, summarize
from thermal_face.windows import condition_for, load_window_records


def write_trace(path, bpm=18.0, duration=80.0, irregular=False, gap=None):
    t = np.arange(0.0, duration + 0.01, 0.1)
    if irregular:
        t[1:-1] += 0.015 * np.sin(np.arange(1, len(t) - 1))
    trace = np.stack([3 * np.sin(2 * np.pi * bpm / 60 * t + i / 10) for i in range(9)], axis=1)
    markers = np.zeros(len(t), dtype=np.int8)
    for event in np.arange(0.0, duration, 60 / bpm):
        markers[np.argmin(abs(t - event))] = 1
    valid = np.ones(len(t), dtype=np.bool_)
    if gap is not None:
        valid[(t >= gap[0]) & (t <= gap[1])] = False
    np.savez(path, seconds=t, trace=trace, valid=valid, marker=markers)


class EvaluationTests(unittest.TestCase):
    def test_spectral_rate_recovers_known_rates_after_irregular_resampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for bpm in (12.0, 18.0, 27.0, 36.0):
                write_trace(folder / "person.npz", bpm=bpm, duration=25, irregular=True)
                windows = load_window_records(folder)["person"]
                self.assertTrue(windows[0].eligible)
                predicted, reason = spectral_rate(windows[0].x)
                self.assertIsNone(reason)
                self.assertAlmostEqual(predicted, bpm, delta=0.75)

    def test_spectral_flat_signal_abstains_and_mask_is_not_a_signal(self):
        x = np.zeros((10, 200), dtype=np.float32)
        x[9] = np.sin(np.arange(200))
        self.assertEqual(spectral_rate(x), (None, "no_spectral_energy"))

    def test_condition_boundaries_and_exact_end_window(self):
        self.assertEqual(condition_for(40, 60), "still")
        self.assertEqual(condition_for(45, 65), "transition")
        self.assertEqual(condition_for(60, 80), "moving")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_trace(folder / "person.npz")
            windows = load_window_records(folder)["person"]
            self.assertEqual(len(windows), 13)
            self.assertEqual((windows[-1].start_s, windows[-1].end_s), (60.0, 80.0))
            self.assertEqual(windows[-1].condition, "moving")

    def test_frozen_gap_is_reported_without_compressing_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_trace(folder / "person.npz", gap=(10, 30))
            windows = load_window_records(folder)["person"]
            self.assertEqual(len(windows), 13)
            self.assertEqual(windows[0].signal_rejection, "missing_fraction_over_20_percent")
            self.assertGreater(windows[0].missing_fraction, 0.2)
            self.assertFalse(windows[0].eligible)
            self.assertEqual(windows[-1].start_s, 60)
            self.assertTrue(windows[-1].eligible)

    def test_invalid_timestamps_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_trace(folder / "person.npz")
            with np.load(folder / "person.npz") as data:
                record = {key: data[key] for key in data.files}
            record["seconds"][10] = record["seconds"][9]
            np.savez(folder / "person.npz", **record)
            with self.assertRaisesRegex(ValueError, "Invalid timestamped"):
                load_window_records(folder)

    def test_macro_error_and_common_coverage_do_not_hide_abstentions(self):
        def row(subject, prediction):
            return {
                "subject": subject, "eligible": True, "reference_bpm": 18.0,
                "signal_rejection": None, "reference_rejection": None,
                "predictions_bpm": dict.fromkeys(METHODS, prediction),
                "abstention_reasons": dict.fromkeys(METHODS),
            }
        rows = [row("a", 18), row("a", 18), row("b", 30)]
        report = summarize(rows)
        self.assertEqual(report["methods"]["cnn"]["macro_mae_bpm"], 6)
        self.assertEqual(report["methods"]["cnn"]["pooled_window_mae_bpm"], 4)
        rows[2]["predictions_bpm"]["spectral"] = None
        rows[2]["abstention_reasons"]["spectral"] = "no_spectral_energy"
        rejected = row("b", None)
        rejected.update(eligible=False, signal_rejection="insufficient_valid_support")
        rows.append(rejected)
        report = summarize(rows)
        self.assertEqual(report["candidate_windows"], 4)
        self.assertEqual(report["eligible_windows"], 3)
        self.assertEqual(report["methods"]["spectral"]["coverage_of_candidates"], 0.5)
        self.assertEqual(report["common_comparison"]["windows"], 2)
        for metrics in report["common_comparison"]["methods"].values():
            self.assertEqual(metrics["evaluated_windows"], 2)
            self.assertEqual(metrics["evaluated_people"], 1)
        self.assertIsNone(summarize([])["methods"]["cnn"]["macro_mae_bpm"])

    def test_full_evaluator_serializes_and_uses_training_median_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for person, bpm in (("a", 12.0), ("b", 18.0), ("c", 24.0)):
                write_trace(folder / f"{person}.npz", bpm=bpm, duration=25)
            (folder / "anestis.npz").write_bytes(b"unverified stale trace")
            out = folder / "report.json"
            report = run(folder, out, epochs=1, device="cpu")
            self.assertEqual(json.loads(out.read_text()), report)
            self.assertEqual(report["independent_test_people"], 3)
            self.assertNotIn("anestis", report["by_subject"])
            for fold in report["folds"]:
                self.assertNotIn(fold["held_out"], fold["train_subjects"])
                training_labels = [row["reference_bpm"] for row in report["windows"]
                                   if row["subject"] in fold["train_subjects"] and row["eligible"]]
                expected = float(np.median(np.array(training_labels, dtype=np.float32)))
                self.assertEqual(fold["train_median_bpm"], expected)
            self.assertEqual(report["overall"]["common_comparison"]["windows"], 6)
            self.assertIsNone(report["by_condition"]["moving"]["methods"]["cnn"]["macro_mae_bpm"])


if __name__ == "__main__":
    unittest.main()
