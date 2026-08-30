from __future__ import annotations

import numpy as np


def apply_width(stereo: np.ndarray, width: float) -> np.ndarray:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("stereo must be a (samples, 2) array")

    left, right = stereo[:, 0], stereo[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    widened_side = width * side
    return np.column_stack((mid + widened_side, mid - widened_side))
