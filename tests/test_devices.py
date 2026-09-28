"""Device selection and real CUDA execution, when CUDA is available."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from thermal_face.models import CNNGRURateNet, CNNRateNet
from thermal_face.train_rate import execution_environment, fit_predict, resolve_device


class DeviceTests(unittest.TestCase):
    def test_cpu_override_and_auto_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("cpu").type, "cpu")
            self.assertEqual(resolve_device("auto").type, "cpu")
            self.assertEqual(execution_environment("cpu")["device_name"], "CPU")

    def test_explicit_cuda_cannot_silently_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                resolve_device("cuda")
        with self.assertRaises(ValueError):
            resolve_device("invalid")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_both_models_train_and_predict_on_cuda(self):
        self.assertEqual(resolve_device("auto").type, "cuda")
        rng = np.random.default_rng(7)
        train = [(rng.normal(size=(10, 200)).astype(np.float32), np.float32(18)) for _ in range(3)]
        test_x = rng.normal(size=(2, 10, 200)).astype(np.float32)
        for architecture, base in (("cnn", CNNRateNet), ("cnn-gru", CNNGRURateNet)):
            observed = []

            class CheckedModel(base):
                def forward(self, x):
                    observed.append((self.training, x.device.type, next(self.parameters()).device.type))
                    return super().forward(x)

            with patch(f"thermal_face.models.{base.__name__}", CheckedModel):
                predictions, count = fit_predict(train, test_x, epochs=1, architecture=architecture, device="cuda")
            self.assertEqual(predictions.shape, (2,))
            self.assertTrue(np.isfinite(predictions).all())
            self.assertGreater(count, 0)
            self.assertEqual({training for training, _, _ in observed}, {True, False})
            self.assertTrue(all(x_device == model_device == "cuda" for _, x_device, model_device in observed))


if __name__ == "__main__":
    unittest.main()
