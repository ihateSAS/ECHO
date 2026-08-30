from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from create_wav_for_angle import read_mono, spatialize  # noqa: E402
from estimate_regime import DELAY_THRESHOLD_US, estimate_regime  # noqa: E402
from spatialize.delay_widening import (  # noqa: E402
    delay_right_channel,
    samples_for_delay,
    widen_with_delay,
)

SAMPLE_RATE = 48_000
CELLO_PATH = Path(__file__).parent / "AuSep_2_vc_01_Jupiter.wav"
DELAY_RECOVERY_TOL_US = 30.0


def test_samples_for_delay_matches_definition():
    assert samples_for_delay(1_000.0, SAMPLE_RATE) == round(1_000.0 * SAMPLE_RATE / 1_000_000.0)
    assert samples_for_delay(0.0, SAMPLE_RATE) == 0


def test_rejects_negative_delay():
    with pytest.raises(ValueError):
        samples_for_delay(-1.0, SAMPLE_RATE)
    with pytest.raises(ValueError):
        delay_right_channel(np.zeros((100, 2)), -5)


def test_rejects_mono_input():
    with pytest.raises(ValueError):
        delay_right_channel(np.zeros((100, 1)), 10)


def test_zero_delay_is_identity():
    stereo = np.random.default_rng(0).standard_normal((1_000, 2))
    out = widen_with_delay(stereo, SAMPLE_RATE, delay_us=0.0)
    assert np.array_equal(out, stereo)


def test_delay_inserts_exact_sample_count():
    n = 2_000
    signal = np.sin(2.0 * np.pi * 200.0 * np.arange(n) / SAMPLE_RATE)
    stereo = np.column_stack((signal, signal))
    delay_us = 500.0
    delay_samples = samples_for_delay(delay_us, SAMPLE_RATE)

    out = widen_with_delay(stereo, SAMPLE_RATE, delay_us)
    assert out.shape[0] == n + delay_samples
    assert np.array_equal(out[:n, 0], signal)
    assert np.all(out[:delay_samples, 1] == 0.0)
    assert np.array_equal(out[delay_samples:, 1], signal)


@pytest.mark.skipif(not CELLO_PATH.exists(), reason="real cello fixture not present")
def test_dry_amplitude_panned_cello_is_not_misclassified():
    mono, sample_rate, _ = read_mono(CELLO_PATH)
    excerpt = mono[: int(3.0 * sample_rate)]
    stereo = spatialize(excerpt, 15.0)
    result = estimate_regime_from_array(stereo, sample_rate)
    assert result.regime == "amplitude-panned"


@pytest.mark.skipif(not CELLO_PATH.exists(), reason="real cello fixture not present")
@pytest.mark.parametrize("delay_us", [100.0, 300.0, 600.0])
def test_widening_flips_regime_classification(tmp_path, delay_us):
    mono, sample_rate, _ = read_mono(CELLO_PATH)
    excerpt = mono[: int(3.0 * sample_rate)]
    stereo = spatialize(excerpt, 15.0)

    widened = widen_with_delay(stereo, sample_rate, delay_us)
    path = tmp_path / f"widened_{delay_us}.wav"
    sf.write(path, widened, sample_rate)

    result = estimate_regime(path)
    assert result.regime == "delayed"
    assert abs(result.delay_us) > DELAY_THRESHOLD_US
    assert result.delay_us == pytest.approx(delay_us, abs=DELAY_RECOVERY_TOL_US)


def estimate_regime_from_array(stereo: np.ndarray, sample_rate: int):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "dry.wav"
        sf.write(path, stereo, sample_rate)
        return estimate_regime(path)
