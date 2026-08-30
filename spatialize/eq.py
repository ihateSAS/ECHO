from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter


def peaking_eq_coefficients(
    center_hz: float, gain_db: float, q: float, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < center_hz < sample_rate / 2.0:
        raise ValueError("center_hz must be between 0 and the Nyquist frequency")
    if q <= 0.0:
        raise ValueError("q must be positive")

    a_amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * center_hz / sample_rate
    alpha = math.sin(omega) / (2.0 * q)
    cos_omega = math.cos(omega)

    b0 = 1.0 + alpha * a_amplitude
    b1 = -2.0 * cos_omega
    b2 = 1.0 - alpha * a_amplitude
    a0 = 1.0 + alpha / a_amplitude
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha / a_amplitude

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_peaking_eq(
    stereo: np.ndarray, sample_rate: int, center_hz: float, gain_db: float, q: float = 1.0
) -> np.ndarray:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("stereo must be a (samples, 2) array")

    b, a = peaking_eq_coefficients(center_hz, gain_db, q, sample_rate)
    left = lfilter(b, a, stereo[:, 0])
    right = lfilter(b, a, stereo[:, 1])
    return np.column_stack((left, right))
