from __future__ import annotations

import math

from scipy.optimize import brentq

from spatialize.binaural import woodworth_itd_seconds

MAX_ANGLE_RADIANS = math.pi / 2.0


DELAY_CLIP_MARGIN_US = 100.0


def invert_woodworth(delay_us: float) -> float:
    target_seconds = delay_us * 1e-6
    max_seconds = woodworth_itd_seconds(MAX_ANGLE_RADIANS)
    margin_seconds = DELAY_CLIP_MARGIN_US * 1e-6

    if abs(target_seconds) > max_seconds + margin_seconds:
        raise ValueError(
            f"delay {delay_us:g} us is outside the +-90 degree Woodworth range"
        )
    clipped_seconds = max(-max_seconds, min(max_seconds, target_seconds))

    def residual(angle_radians: float) -> float:
        return woodworth_itd_seconds(angle_radians) - clipped_seconds

    root_radians = brentq(residual, -MAX_ANGLE_RADIANS, MAX_ANGLE_RADIANS)
    return math.degrees(root_radians)
