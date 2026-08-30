from __future__ import annotations

import numpy as np


def delay_right_channel(stereo: np.ndarray, delay_samples: int) -> np.ndarray:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("stereo must be a (samples, 2) array")
    if delay_samples < 0:
        raise ValueError("delay_samples must be non-negative")
    if delay_samples == 0:
        return stereo.copy()

    left = np.concatenate((stereo[:, 0], np.zeros(delay_samples, dtype=stereo.dtype)))
    right = np.concatenate((np.zeros(delay_samples, dtype=stereo.dtype), stereo[:, 1]))
    return np.column_stack((left, right))


def samples_for_delay(delay_us: float, sample_rate: int) -> int:
    if delay_us < 0.0:
        raise ValueError("delay_us must be non-negative")
    return round(delay_us * sample_rate / 1_000_000.0)


def widen_with_delay(stereo: np.ndarray, sample_rate: int, delay_us: float) -> np.ndarray:
    delay_samples = samples_for_delay(delay_us, sample_rate)
    return delay_right_channel(stereo, delay_samples)
