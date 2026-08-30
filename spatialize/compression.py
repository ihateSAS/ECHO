from __future__ import annotations

import math

import numpy as np


def _one_pole_coefficient(time_constant_s: float, sample_rate: int) -> float:
    if time_constant_s <= 0.0:
        return 0.0
    return math.exp(-1.0 / (time_constant_s * sample_rate))


def gain_reduction_db(level_db: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    if ratio < 1.0:
        raise ValueError("ratio must be >= 1")
    excess = np.maximum(level_db - threshold_db, 0.0)
    return excess * (1.0 - 1.0 / ratio)


def stereo_linked_compressor(
    stereo: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    attack_s: float = 0.005,
    release_s: float = 0.10,
    makeup_gain_db: float = 0.0,
) -> np.ndarray:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("stereo must be a (samples, 2) array")
    if ratio < 1.0:
        raise ValueError("ratio must be >= 1")

    detector = np.maximum(np.abs(stereo[:, 0]), np.abs(stereo[:, 1]))
    floor = np.finfo(np.float64).tiny
    level_db = 20.0 * np.log10(np.maximum(detector, floor))
    target_reduction_db = gain_reduction_db(level_db, threshold_db, ratio)

    attack_coeff = _one_pole_coefficient(attack_s, sample_rate)
    release_coeff = _one_pole_coefficient(release_s, sample_rate)

    smoothed_reduction_db = np.empty_like(target_reduction_db)
    current = 0.0
    for i in range(target_reduction_db.shape[0]):
        target = target_reduction_db[i]

        coeff = attack_coeff if target > current else release_coeff
        current = coeff * current + (1.0 - coeff) * target
        smoothed_reduction_db[i] = current

    gain_db = makeup_gain_db - smoothed_reduction_db
    gain_linear = 10.0 ** (gain_db / 20.0)
    return stereo * gain_linear[:, np.newaxis]
