from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from create_wav_for_angle import read_mono  # noqa: E402
from cues.binaural import invert_woodworth  # noqa: E402
from estimate_binaural import estimate_binaural_angle  # noqa: E402
from estimate_regime import DELAY_THRESHOLD_US, estimate_regime  # noqa: E402
from spatialize.binaural import (  # noqa: E402
    apply_fractional_delay,
    apply_head_shadow,
    binaural_render,
    head_shadow_gain_db,
    woodworth_itd_seconds,
    woodworth_itd_us,
)

SAMPLE_RATE = 48_000
CELLO_PATH = Path(__file__).parent / "AuSep_2_vc_01_Jupiter.wav"
ANGLE_RECOVERY_TOL_DEGREES = 5.0
DELAY_RECOVERY_TOL_US = 30.0
TEST_ANGLES_DEGREES = (-60.0, -30.0, -10.0, 10.0, 30.0, 60.0, 90.0)


def _load_mono_excerpt(seconds: float = 3.0) -> tuple[np.ndarray, int]:
    mono, sample_rate, _ = read_mono(CELLO_PATH)
    return mono[: int(seconds * sample_rate)], sample_rate


def _write(tmp_path: Path, name: str, stereo: np.ndarray, sample_rate: int) -> Path:
    path = tmp_path / name
    sf.write(path, stereo, sample_rate)
    return path


def test_woodworth_matches_ninety_degree_worked_example():
    assert woodworth_itd_us(90.0) == pytest.approx(655.8, abs=1.0)


def test_woodworth_matches_claude_md_worked_example():
    assert invert_woodworth(380.7) == pytest.approx(45.0, abs=0.1)


def test_woodworth_is_odd_symmetric():
    for angle_deg in (5.0, 10.0, 30.0, 60.0, 89.0):
        angle_rad = math.radians(angle_deg)
        assert woodworth_itd_seconds(-angle_rad) == pytest.approx(
            -woodworth_itd_seconds(angle_rad), abs=1e-12
        )


def test_woodworth_zero_angle_is_zero_delay():
    assert woodworth_itd_seconds(0.0) == 0.0


@pytest.mark.parametrize("delay_us", [50.0, 200.0, 380.7, 500.0, 655.8])
def test_invert_woodworth_round_trips_forward_formula(delay_us):
    angle_deg = invert_woodworth(delay_us)
    assert woodworth_itd_us(angle_deg) == pytest.approx(delay_us, abs=1e-3)


def test_invert_woodworth_rejects_delay_outside_range():
    with pytest.raises(ValueError):
        invert_woodworth(1_000.0)


def test_invert_woodworth_clips_noise_past_the_physical_boundary():
    max_delay_us = woodworth_itd_us(90.0)
    assert invert_woodworth(max_delay_us + 2.0) == pytest.approx(90.0, abs=0.5)
    assert invert_woodworth(-(max_delay_us + 2.0)) == pytest.approx(-90.0, abs=0.5)


def test_fractional_delay_matches_linear_interpolation_definition():
    impulse = np.zeros(20)
    impulse[5] = 1.0
    delayed = apply_fractional_delay(impulse, 4.3)


    assert delayed[9] == pytest.approx(0.7, abs=1e-12)
    assert delayed[10] == pytest.approx(0.3, abs=1e-12)
    assert np.allclose(np.delete(delayed, [9, 10]), 0.0, atol=1e-12)


def test_fractional_delay_rejects_negative_delay():
    with pytest.raises(ValueError):
        apply_fractional_delay(np.zeros(10), -1.0)


def test_fractional_delay_zero_is_identity():
    signal = np.random.default_rng(0).standard_normal(500)
    assert np.allclose(apply_fractional_delay(signal, 0.0), signal, atol=1e-12)


def test_head_shadow_is_near_flat_at_low_frequency():
    gain_db = head_shadow_gain_db(np.array([50.0, 100.0]), angle_degrees=90.0)
    assert np.all(np.abs(gain_db) < 2.0)


def test_head_shadow_rises_toward_high_frequency():
    frequencies = np.array([100.0, 1_000.0, 4_000.0, 8_000.0, 16_000.0])
    gain_db = head_shadow_gain_db(frequencies, angle_degrees=90.0)
    assert np.all(np.diff(gain_db) < 0.0)


def test_head_shadow_lands_in_doc_target_range_at_full_offset():
    gain_db = head_shadow_gain_db(np.array([8_000.0]), angle_degrees=90.0)
    assert -20.0 <= gain_db[0] <= -15.0


def test_head_shadow_scales_with_how_far_off_center():
    small_angle_db = head_shadow_gain_db(np.array([8_000.0]), angle_degrees=10.0)
    large_angle_db = head_shadow_gain_db(np.array([8_000.0]), angle_degrees=90.0)
    assert abs(small_angle_db[0]) < abs(large_angle_db[0])


def test_head_shadow_zero_angle_is_near_identity():
    n = SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    mono = np.sin(2.0 * np.pi * 440.0 * t) + 0.3 * np.sin(2.0 * np.pi * 2_000.0 * t)
    out = apply_head_shadow(mono, SAMPLE_RATE, angle_degrees=0.0)
    assert np.allclose(out, mono, atol=1e-9)


def test_render_zero_angle_gives_identical_channels():
    n = SAMPLE_RATE
    mono = np.random.default_rng(0).standard_normal(n)
    stereo = binaural_render(mono, SAMPLE_RATE, angle_degrees=0.0)
    assert np.allclose(stereo[:, 0], stereo[:, 1], atol=1e-9)


def test_render_positive_angle_delays_right_channel():
    n = SAMPLE_RATE
    mono = np.random.default_rng(1).standard_normal(n)
    stereo = binaural_render(mono, SAMPLE_RATE, angle_degrees=30.0)


    delay_samples = int(
        math.floor(abs(woodworth_itd_us(30.0)) * SAMPLE_RATE / 1_000_000.0)
    )
    assert np.allclose(stereo[:5, 0], mono[:5], atol=1e-6)
    assert np.all(stereo[:delay_samples, 1] == 0.0)


def test_render_negative_angle_delays_left_channel():
    n = SAMPLE_RATE
    mono = np.random.default_rng(1).standard_normal(n)
    stereo = binaural_render(mono, SAMPLE_RATE, angle_degrees=-30.0)
    delay_samples = int(
        math.floor(abs(woodworth_itd_us(-30.0)) * SAMPLE_RATE / 1_000_000.0)
    )
    assert np.allclose(stereo[:5, 1], mono[:5], atol=1e-6)
    assert np.all(stereo[:delay_samples, 0] == 0.0)


def test_render_rejects_angle_outside_ninety_degrees():
    with pytest.raises(ValueError):
        binaural_render(np.zeros(100), SAMPLE_RATE, angle_degrees=120.0)


pytestmark = pytest.mark.skipif(
    not CELLO_PATH.exists(), reason="real cello fixture not present"
)


@pytest.mark.parametrize("angle_deg", TEST_ANGLES_DEGREES)
def test_classifier_calls_binaural_files_delayed(tmp_path, angle_deg):
    mono, sample_rate = _load_mono_excerpt()
    stereo = binaural_render(mono, sample_rate, angle_deg)
    path = _write(tmp_path, f"binaural_{angle_deg}.wav", stereo, sample_rate)

    result = estimate_regime(path)
    assert result.regime == "delayed"


@pytest.mark.parametrize("angle_deg", TEST_ANGLES_DEGREES)
def test_recovered_delay_within_tolerance(tmp_path, angle_deg):
    mono, sample_rate = _load_mono_excerpt()
    stereo = binaural_render(mono, sample_rate, angle_deg)
    path = _write(tmp_path, f"binaural_{angle_deg}.wav", stereo, sample_rate)

    planted_delay_us = woodworth_itd_us(angle_deg)
    result = estimate_binaural_angle(path)
    assert result.delay_us == pytest.approx(planted_delay_us, abs=DELAY_RECOVERY_TOL_US)
    assert abs(result.delay_us) > DELAY_THRESHOLD_US or angle_deg == 0.0


@pytest.mark.parametrize("angle_deg", TEST_ANGLES_DEGREES)
def test_recovered_angle_within_five_degrees(tmp_path, angle_deg):
    mono, sample_rate = _load_mono_excerpt()
    stereo = binaural_render(mono, sample_rate, angle_deg)
    path = _write(tmp_path, f"binaural_{angle_deg}.wav", stereo, sample_rate)

    result = estimate_binaural_angle(path)
    assert result.angle_degrees == pytest.approx(
        angle_deg, abs=ANGLE_RECOVERY_TOL_DEGREES
    )


@pytest.mark.parametrize("angle_deg", [-60.0, -30.0, 30.0, 60.0, 90.0])
def test_level_difference_is_not_flat_for_binaural_files(tmp_path, angle_deg):
    mono, sample_rate = _load_mono_excerpt()
    stereo = binaural_render(mono, sample_rate, angle_deg)
    path = _write(tmp_path, f"binaural_{angle_deg}.wav", stereo, sample_rate)

    result = estimate_regime(path)
    assert result.level_flatness_db > 0.5
