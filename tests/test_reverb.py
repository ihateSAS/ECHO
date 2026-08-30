from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from estimate_angle import estimate_angle  # noqa: E402
from scipy.signal import coherence  # noqa: E402
from spatialize.reverb import (  # noqa: E402
    allpass_filter,
    comb_feedback_gain,
    feedback_comb,
    schroeder_reverb,
)

SAMPLE_RATE = 48_000
THETA0_DEG = 30.0
RT60_TOL_FRACTION = 0.15
AZIMUTH_TOL_DEG = 5.0


def _measure_rt60_seconds(x: np.ndarray, sample_rate: int, floor_db: float = -60.0) -> float:
    energy = x.astype(np.float64) ** 2
    reverse_cumulative = np.cumsum(energy[::-1])[::-1]
    normalized = reverse_cumulative / reverse_cumulative[0]
    decay_db = 10.0 * np.log10(np.maximum(normalized, 1e-300))

    below_floor = np.where(decay_db <= floor_db)[0]
    if below_floor.size == 0:
        raise AssertionError(f"decay curve never reached {floor_db} dB")
    index = int(below_floor[0])
    if index == 0:
        return 0.0

    t_before, t_after = (index - 1) / sample_rate, index / sample_rate
    db_before, db_after = decay_db[index - 1], decay_db[index]
    fraction = (floor_db - db_before) / (db_after - db_before)
    return t_before + fraction * (t_after - t_before)


def _panned_tone(angle_deg: float, duration_s: float, sample_rate: int) -> np.ndarray:
    d = math.tan(math.radians(angle_deg)) / math.tan(math.radians(THETA0_DEG))
    gain_left, gain_right = (1.0 + d) / 2.0, (1.0 - d) / 2.0
    norm = math.hypot(gain_left, gain_right)
    gain_left, gain_right = gain_left / norm, gain_right / norm

    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    mono = np.sin(2.0 * np.pi * 440.0 * t)
    return np.column_stack((mono * gain_left, mono * gain_right))


def _broadband_coherence(left: np.ndarray, right: np.ndarray, sample_rate: int) -> float:
    frequencies, gamma_sq = coherence(left, right, fs=sample_rate, nperseg=4096)
    in_band = (frequencies >= 80.0) & (frequencies <= 8_000.0)
    return float(np.median(gamma_sq[in_band]))


def test_dry_is_identity():
    stereo = _panned_tone(15.0, 0.2, SAMPLE_RATE)
    out = schroeder_reverb(stereo, SAMPLE_RATE, rt60_s=0.4, wet=0.0)
    assert np.array_equal(out[: stereo.shape[0]], stereo)
    assert np.all(out[stereo.shape[0] :] == 0.0)


def test_rejects_mono_input():
    mono = np.zeros((1000, 1))
    with pytest.raises(ValueError):
        schroeder_reverb(mono, SAMPLE_RATE, rt60_s=0.4, wet=0.5)


@pytest.mark.parametrize("wet", [-0.1, 1.1])
def test_rejects_wet_out_of_range(wet):
    stereo = _panned_tone(0.0, 0.1, SAMPLE_RATE)
    with pytest.raises(ValueError):
        schroeder_reverb(stereo, SAMPLE_RATE, rt60_s=0.4, wet=wet)


@pytest.mark.parametrize("diffusion", [-0.1, 1.1])
def test_rejects_diffusion_out_of_range(diffusion):
    stereo = _panned_tone(0.0, 0.1, SAMPLE_RATE)
    with pytest.raises(ValueError):
        schroeder_reverb(stereo, SAMPLE_RATE, rt60_s=0.4, wet=0.5, diffusion=diffusion)


def test_deterministic_given_seed():
    stereo = _panned_tone(10.0, 0.3, SAMPLE_RATE)
    first = schroeder_reverb(stereo, SAMPLE_RATE, rt60_s=0.3, wet=0.8, diffusion=0.6, seed=7)
    second = schroeder_reverb(stereo, SAMPLE_RATE, rt60_s=0.3, wet=0.8, diffusion=0.6, seed=7)
    assert np.array_equal(first, second)


def test_different_seeds_differ():
    stereo = _panned_tone(10.0, 0.3, SAMPLE_RATE)
    first = schroeder_reverb(stereo, SAMPLE_RATE, rt60_s=0.3, wet=0.8, diffusion=0.6, seed=1)
    second = schroeder_reverb(stereo, SAMPLE_RATE, rt60_s=0.3, wet=0.8, diffusion=0.6, seed=2)
    assert not np.array_equal(first, second)


def _reference_comb_feedback_gain(delay_samples: int, sample_rate: int, rt60_s: float) -> float:
    round_trips_per_rt60 = rt60_s * sample_rate / delay_samples
    return 10.0 ** (math.log10(1e-3) / round_trips_per_rt60)


@pytest.mark.parametrize("delay_samples,rt60_s", [(1200, 0.3), (1800, 0.6), (900, 1.0)])
def test_comb_feedback_gain_matches_independent_derivation(delay_samples, rt60_s):
    expected = _reference_comb_feedback_gain(delay_samples, SAMPLE_RATE, rt60_s)
    assert comb_feedback_gain(delay_samples, SAMPLE_RATE, rt60_s) == pytest.approx(
        expected, rel=1e-9
    )


def test_single_comb_reaches_planted_rt60():
    rt60_target = 0.5
    delay_samples = 1200
    gain = comb_feedback_gain(delay_samples, SAMPLE_RATE, rt60_target)

    impulse = np.zeros(int(2.0 * SAMPLE_RATE))
    impulse[0] = 1.0
    response = feedback_comb(impulse, delay_samples, gain)

    measured = _measure_rt60_seconds(response, SAMPLE_RATE)
    assert measured == pytest.approx(rt60_target, rel=0.05)


def test_allpass_does_not_change_energy_decay():
    rt60_target = 0.5
    gain = comb_feedback_gain(1200, SAMPLE_RATE, rt60_target)
    impulse = np.zeros(int(2.0 * SAMPLE_RATE))
    impulse[0] = 1.0
    comb_response = feedback_comb(impulse, 1200, gain)

    before = _measure_rt60_seconds(comb_response, SAMPLE_RATE)
    after = _measure_rt60_seconds(allpass_filter(comb_response, 556, 0.5), SAMPLE_RATE)
    assert after == pytest.approx(before, rel=0.10)


@pytest.mark.parametrize("rt60_target", [0.3, 0.5, 0.9])
def test_full_network_rt60_within_tolerance(rt60_target):
    n = int(0.02 * SAMPLE_RATE)
    impulse = np.zeros((n, 2))
    impulse[0, 0] = 1.0
    impulse[0, 1] = 1.0

    rendered = schroeder_reverb(
        impulse, SAMPLE_RATE, rt60_s=rt60_target, wet=1.0, damping=0.25, diffusion=0.0
    )
    measured_left = _measure_rt60_seconds(rendered[:, 0], SAMPLE_RATE)
    measured_right = _measure_rt60_seconds(rendered[:, 1], SAMPLE_RATE)

    assert measured_left == pytest.approx(rt60_target, rel=RT60_TOL_FRACTION)
    assert measured_right == pytest.approx(rt60_target, rel=RT60_TOL_FRACTION)


def test_dry_panned_tone_is_fully_coherent():
    stereo = _panned_tone(20.0, 1.0, SAMPLE_RATE)
    gamma_sq = _broadband_coherence(stereo[:, 0], stereo[:, 1], SAMPLE_RATE)
    assert gamma_sq > 0.99


@pytest.mark.parametrize("wet", [0.3, 0.6, 1.0])
def test_coherence_drops_with_reverb(wet):
    stereo = _panned_tone(20.0, 1.0, SAMPLE_RATE)
    wet_audio = schroeder_reverb(
        stereo, SAMPLE_RATE, rt60_s=0.4, wet=wet, diffusion=0.6, seed=0
    )
    gamma_sq = _broadband_coherence(wet_audio[:, 0], wet_audio[:, 1], SAMPLE_RATE)
    assert gamma_sq < 0.3


def test_coherence_insensitive_to_diffusion_zero():
    stereo = _panned_tone(20.0, 1.0, SAMPLE_RATE)
    wet_audio = schroeder_reverb(
        stereo, SAMPLE_RATE, rt60_s=0.4, wet=1.0, diffusion=0.0, seed=0
    )
    gamma_sq = _broadband_coherence(wet_audio[:, 0], wet_audio[:, 1], SAMPLE_RATE)
    assert gamma_sq > 0.9


def test_reverb_pulls_estimated_angle_toward_center(tmp_path):
    import soundfile as sf

    angle_planted = 25.0
    stereo = _panned_tone(angle_planted, 1.0, SAMPLE_RATE)

    measured_angles = {}
    for wet in (0.0, 0.3, 0.7, 1.0):
        wet_audio = schroeder_reverb(
            stereo, SAMPLE_RATE, rt60_s=0.4, wet=wet, diffusion=0.6, seed=0
        )
        path = tmp_path / f"wet_{wet}.wav"
        sf.write(path, wet_audio, SAMPLE_RATE)
        angle, _, _ = estimate_angle(path)
        measured_angles[wet] = abs(angle)

    assert measured_angles[0.0] == pytest.approx(angle_planted, abs=AZIMUTH_TOL_DEG)
    for wet in (0.3, 0.7, 1.0):
        assert measured_angles[wet] < AZIMUTH_TOL_DEG
        assert measured_angles[wet] < measured_angles[0.0] - 10.0
