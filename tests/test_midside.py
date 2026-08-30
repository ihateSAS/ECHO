from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatialize.midside import apply_width  # noqa: E402

SAMPLE_RATE = 48_000
THETA0_DEG = 30.0


def _gains_for_angle(angle_deg: float) -> tuple[float, float]:
    d = math.tan(math.radians(angle_deg)) / math.tan(math.radians(THETA0_DEG))
    gain_left, gain_right = (1.0 + d) / 2.0, (1.0 - d) / 2.0
    norm = math.hypot(gain_left, gain_right)
    return gain_left / norm, gain_right / norm


def _recover_signed_gains(stereo: np.ndarray, mono: np.ndarray) -> tuple[float, float]:
    denominator = np.dot(mono, mono)
    return float(np.dot(stereo[:, 0], mono) / denominator), float(
        np.dot(stereo[:, 1], mono) / denominator
    )


@pytest.mark.parametrize("angle_deg", [10.0, 25.0, -15.0])
@pytest.mark.parametrize("width", [0.0, 0.5, 1.0, 1.5])
def test_sum_and_difference_scale_exactly(angle_deg, width):
    gain_left, gain_right = _gains_for_angle(angle_deg)
    n = 4_800
    t = np.arange(n) / SAMPLE_RATE
    mono = np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.column_stack((mono * gain_left, mono * gain_right))

    out = apply_width(stereo, width)
    gain_left_out, gain_right_out = _recover_signed_gains(out, mono)

    assert gain_left_out + gain_right_out == pytest.approx(gain_left + gain_right, abs=1e-9)
    assert gain_left_out - gain_right_out == pytest.approx(
        width * (gain_left - gain_right), abs=1e-9
    )


def test_width_one_is_identity():
    n = 4_800
    stereo = np.random.default_rng(0).standard_normal((n, 2))
    out = apply_width(stereo, 1.0)
    assert np.allclose(out, stereo)


def test_width_zero_collapses_to_exact_center():
    gain_left, gain_right = _gains_for_angle(25.0)
    n = 4_800
    mono = np.sin(2.0 * np.pi * 440.0 * np.arange(n) / SAMPLE_RATE)
    stereo = np.column_stack((mono * gain_left, mono * gain_right))
    out = apply_width(stereo, 0.0)
    assert np.array_equal(out[:, 0], out[:, 1])


def test_negative_width_inverts_side():
    gain_left, gain_right = _gains_for_angle(20.0)
    n = 4_800
    mono = np.sin(2.0 * np.pi * 440.0 * np.arange(n) / SAMPLE_RATE)
    stereo = np.column_stack((mono * gain_left, mono * gain_right))
    out = apply_width(stereo, -1.0)
    gain_left_out, gain_right_out = _recover_signed_gains(out, mono)

    assert gain_left_out == pytest.approx(gain_right, abs=1e-9)
    assert gain_right_out == pytest.approx(gain_left, abs=1e-9)


def test_rejects_mono_input():
    mono = np.zeros((1000, 1))
    with pytest.raises(ValueError):
        apply_width(mono, 0.5)
