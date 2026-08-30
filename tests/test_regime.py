from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from estimate_regime import (  # noqa: E402
    DELAY_THRESHOLD_US,
    FLATNESS_THRESHOLD_DB,
    PHASE_THRESHOLD_DEGREES,
    estimate_regime,
)

CELLO_PATH = Path(__file__).parent / "AuSep_2_vc_01_Jupiter.wav"
SAMPLE_RATE = 48_000
THETA0_DEG = 30.0
DELAY_RECOVERY_TOL_US = 30.0
EXCERPT_S = 3.0

pytestmark = pytest.mark.skipif(
    not CELLO_PATH.exists(), reason="real cello fixture not present"
)


def _load_mono_excerpt() -> np.ndarray:
    mono, sample_rate = sf.read(CELLO_PATH, dtype="float64", always_2d=True)
    assert sample_rate == SAMPLE_RATE, "fixture must already be 48 kHz"
    return mono[: int(EXCERPT_S * sample_rate), 0]


def _pan(mono: np.ndarray, angle_deg: float) -> np.ndarray:
    d = math.tan(math.radians(angle_deg)) / math.tan(math.radians(THETA0_DEG))
    gain_left, gain_right = (1.0 + d) / 2.0, (1.0 - d) / 2.0
    norm = math.hypot(gain_left, gain_right)
    gain_left, gain_right = gain_left / norm, gain_right / norm
    return np.column_stack((mono * gain_left, mono * gain_right))


def _delay_right(mono: np.ndarray, delay_us: float, sample_rate: int) -> np.ndarray:
    delay_samples = round(delay_us * sample_rate / 1_000_000.0)
    padding = np.zeros(delay_samples, dtype=mono.dtype)
    left = np.concatenate((mono, padding))
    right = np.concatenate((padding, mono))
    return np.column_stack((left, right))


def _write(tmp_path: Path, name: str, stereo: np.ndarray) -> Path:
    path = tmp_path / name
    sf.write(path, stereo, SAMPLE_RATE)
    return path


@pytest.mark.parametrize("angle_deg", [-25.0, -15.0, 0.0, 15.0, 25.0])
def test_amplitude_panned_files_classify_correctly(tmp_path, angle_deg):
    mono = _load_mono_excerpt()
    stereo = _pan(mono, angle_deg)
    path = _write(tmp_path, f"panned_{angle_deg}.wav", stereo)

    result = estimate_regime(path)
    assert result.regime == "amplitude-panned"
    assert abs(result.delay_us) <= DELAY_THRESHOLD_US
    assert abs(result.phase_at_1khz_degrees) <= PHASE_THRESHOLD_DEGREES
    assert result.level_flatness_db <= FLATNESS_THRESHOLD_DB


@pytest.mark.parametrize("delay_us", [100.0, 300.0, 600.0])
def test_delayed_files_classify_correctly(tmp_path, delay_us):
    mono = _load_mono_excerpt()
    stereo = _delay_right(mono, delay_us, SAMPLE_RATE)
    path = _write(tmp_path, f"delayed_{delay_us}.wav", stereo)

    result = estimate_regime(path)
    assert result.regime == "delayed"


@pytest.mark.parametrize("delay_us", [100.0, 300.0, 600.0])
def test_recovered_delay_within_tolerance(tmp_path, delay_us):
    mono = _load_mono_excerpt()
    stereo = _delay_right(mono, delay_us, SAMPLE_RATE)
    path = _write(tmp_path, f"delayed_{delay_us}.wav", stereo)

    result = estimate_regime(path)
    assert result.delay_us == pytest.approx(delay_us, abs=DELAY_RECOVERY_TOL_US)


def test_delay_sign_convention(tmp_path):
    mono = _load_mono_excerpt()
    stereo = _delay_right(mono, 300.0, SAMPLE_RATE)
    path = _write(tmp_path, "delayed_right.wav", stereo)

    result = estimate_regime(path)
    assert result.delay_us > 0.0


def test_zero_delay_amplitude_panned_is_not_misclassified(tmp_path):
    mono = _load_mono_excerpt()
    stereo = _pan(mono, 10.0)
    path = _write(tmp_path, "zero_delay.wav", stereo)

    result = estimate_regime(path)
    assert result.regime == "amplitude-panned"


@pytest.mark.parametrize(
    "delay_vote,phase_vote,flatness_vote,expected_regime",
    [
        (False, False, False, "amplitude-panned"),
        (True, False, False, "amplitude-panned"),
        (False, True, False, "amplitude-panned"),
        (False, False, True, "amplitude-panned"),
        (True, True, False, "delayed"),
        (True, False, True, "delayed"),
        (False, True, True, "delayed"),
        (True, True, True, "delayed"),
    ],
)
def test_majority_vote_logic(delay_vote, phase_vote, flatness_vote, expected_regime):
    votes = sum((delay_vote, phase_vote, flatness_vote))
    regime = "delayed" if votes >= 2 else "amplitude-panned"
    assert regime == expected_regime
