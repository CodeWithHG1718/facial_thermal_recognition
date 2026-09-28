"""Check gradient flow, subject separation, and pilot report integration."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from thermal_face.models import CNNGRURateNet, CNNRateNet
from thermal_face.train_rate import load_windows, run


class RateModelTests(unittest.TestCase):
    def test_models_accept_single_windows_and_train_all_blocks(self):
        torch.set_num_threads(2)
        for model_class in (CNNRateNet, CNNGRURateNet):
            for batch in (1, 3):
                with self.subTest(model=model_class.__name__, batch=batch):
                    torch.manual_seed(7)
                    model = model_class()
                    x = torch.randn(batch, 10, 200)
                    x[:, 9] = 0
                    x[:, 9, 50:70] = 1
                    predicted = model(x)
                    self.assertEqual(tuple(predicted.shape), (batch,))
                    torch.nn.functional.smooth_l1_loss(
                        predicted, torch.full((batch,), 20.0), beta=3.0,
                    ).backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    blocks = [model.net] if model_class is CNNRateNet else [
                        model.encoder, model.rnn, model.head,
                    ]
                    for block in blocks:
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in block.parameters()), 0)

    def test_stale_anestis_traces_are_excluded_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "anestis.npz").write_bytes(b"unverified stale artifact")
            self.assertEqual(load_windows(folder), {})

    def test_training_excludes_held_out_person_and_writes_both_reports(self):
        # The first channel identifies each person so actual forward calls
        # can verify separation independently of the reported subject names.
        examples = {
            name: [(np.full((10, 200), index, dtype=np.float32), np.float32(rate))]
            for index, (name, rate) in enumerate((("a", 12), ("b", 18), ("c", 24)), 1)
        }
        for architecture, base in (("cnn", CNNRateNet), ("cnn-gru", CNNGRURateNet)):
            seen = []

            class ObservedModel(base):
                def __init__(self):
                    super().__init__()
                    self.calls = {"train": set(), "test": set()}
                    seen.append(self.calls)

                def forward(self, x):
                    self.calls["train" if self.training else "test"].update(x[:, 0, 0].tolist())
                    return super().forward(x)

            with tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary) / "report.json"
                with patch("thermal_face.train_rate.load_windows", return_value=examples), \
                     patch(f"thermal_face.models.{base.__name__}", ObservedModel), \
                     contextlib.redirect_stdout(io.StringIO()):
                    report = run(Path(temporary), out, epochs=1, architecture=architecture, device="cpu")
                self.assertEqual(json.loads(out.read_text()), report)
                self.assertEqual(report["architecture"], architecture)
                self.assertEqual(report["independent_test_people"], 3)
                self.assertGreater(report["trainable_parameters"], 0)
                for index, (fold, calls) in enumerate(zip(report["folds"], seen), 1):
                    self.assertEqual(calls["test"], {float(index)})
                    self.assertEqual(calls["train"], {1.0, 2.0, 3.0} - {float(index)})
                    self.assertEqual(len(fold["predicted_bpm"]), 1)
                    self.assertTrue(np.isfinite(fold["predicted_bpm"]).all())
                    self.assertNotIn(fold["held_out"], fold["train_subjects"])


if __name__ == "__main__":
    unittest.main()
