"""Spectral respiration baseline on traces resampled using actual timestamps."""

import numpy as np

from .windows import SAMPLE_HZ


def spectral_rate(x: np.ndarray) -> tuple[float | None, str | None]:
    """Average normalized nasal-patch power in the fixed 6--45 BPM band.

    Linear detrending and a Hann taper suppress drift and edge discontinuities.
    Eightfold zero padding samples the spectrum more densely; it does not
    improve the underlying resolution of a 20-second observation.
    """
    if x.ndim != 2 or x.shape[0] != 10 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("Expected finite 10-channel resampled traces")
    signal = x[:9].astype(np.float64)
    time = np.arange(signal.shape[1], dtype=np.float64) / SAMPLE_HZ
    time -= time.mean()
    signal -= signal.mean(axis=1, keepdims=True)
    signal -= ((signal @ time) / (time @ time))[:, None] * time
    active = np.std(signal, axis=1) > 1e-8
    if not active.any():
        return None, "no_spectral_energy"
    signal = signal[active] * np.hanning(signal.shape[1])
    n_fft = 8 * signal.shape[1]
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_HZ)
    band = (frequencies >= 0.1) & (frequencies <= 0.75)
    power = np.abs(np.fft.rfft(signal, n=n_fft, axis=1))[:, band] ** 2
    total = power.sum(axis=1)
    active = total > 1e-16
    if not active.any():
        return None, "no_spectral_energy"
    spectrum = (power[active] / total[active, None]).mean(axis=0)
    return float(60.0 * frequencies[band][np.argmax(spectrum)]), None
