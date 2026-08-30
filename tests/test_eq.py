from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import freqz

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatialize.eq import apply_peaking_eq, peaking_eq_coefficients  # noqa: E402

SAMPLE_RATE = 48_000


@pytest.mark.parametrize(
    "center_hz,gain_db,q",
    [(1_000.0, 6.0, 1.0), (500.0, -8.0, 0.7), (4_000.0, 3.0, 2.0), (100.0, -12.0, 1.4)],
)
def test_gain_at_center_frequency_matches_target(center_hz, gain_db, q):
    b, a = peaking_eq_coefficients(center_hz, gain_db, q, SAMPLE_RATE)
    omega = 2.0 * np.pi * center_hz / SAMPLE_RATE
    _, response = freqz(b, a, worN=[omega])
    measured_gain_db = 20.0 * np.log10(np.abs(response[0]))
    assert measured_gain_db == pytest.approx(gain_db, abs=1e-6)


def test_zero_gain_is_near_identity_response():
    b, a = peaking_eq_coefficients(1_000.0, 0.0, 1.0, SAMPLE_RATE)
    frequencies = np.linspace(20.0, 20_000.0, 200)
    omegas = 2.0 * np.pi * frequencies / SAMPLE_RATE
    _, response = freqz(b, a, worN=omegas)
    assert np.allclose(np.abs(response), 1.0, atol=1e-9)


def test_rejects_frequency_above_nyquist():
    with pytest.raises(ValueError):
        peaking_eq_coefficients(30_000.0, 6.0, 1.0, SAMPLE_RATE)


def test_rejects_non_positive_q():
    with pytest.raises(ValueError):
        peaking_eq_coefficients(1_000.0, 6.0, 0.0, SAMPLE_RATE)


def test_applied_identically_preserves_ild():
    n = int(1.0 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    mono = np.sin(2 * np.pi * 1_000 * t)
    stereo = np.column_stack((mono, mono))
    out = apply_peaking_eq(stereo, SAMPLE_RATE, center_hz=1_000.0, gain_db=6.0, q=1.0)
    assert np.allclose(out[:, 0], out[:, 1])


def test_rejects_mono_input():
    mono = np.zeros((1000, 1))
    with pytest.raises(ValueError):
        apply_peaking_eq(mono, SAMPLE_RATE, center_hz=1_000.0, gain_db=6.0)
