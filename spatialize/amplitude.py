from __future__ import annotations

import math

import numpy as np

SPEAKER_ANGLE_DEG = 30.0


def gains_for_azimuth(azimuth_deg: float) -> tuple[float, float]:
    if not math.isfinite(azimuth_deg):
        raise ValueError("azimuth_deg must be finite")
    if abs(azimuth_deg) > SPEAKER_ANGLE_DEG:
        raise ValueError("amplitude-panning azimuth must be within +/-30 degrees")

    difference = math.tan(math.radians(azimuth_deg)) / math.tan(
        math.radians(SPEAKER_ANGLE_DEG)
    )
    left = (1.0 + difference) / 2.0
    right = (1.0 - difference) / 2.0
    norm = math.hypot(left, right)
    return left / norm, right / norm


def render_amplitude_pan(mono: np.ndarray, azimuth_deg: float) -> np.ndarray:
    signal = np.asarray(mono)
    if signal.ndim != 1:
        raise ValueError("mono must be a one-dimensional array")
    if signal.size == 0:
        raise ValueError("mono must not be empty")

    left_gain, right_gain = gains_for_azimuth(azimuth_deg)
    return np.column_stack((signal * left_gain, signal * right_gain))
