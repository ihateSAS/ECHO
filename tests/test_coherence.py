from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import coherence as scipy_coherence

from create_wav_for_coherence import (
    decorrelated_noise,
    panned_with_noise,
    predicted_coherence,
    unrelated_pair,
)
from cues.coherence import (
    MAX_BAND_HZ,
    MIN_BAND_HZ,
    MIN_SEGMENTS,
    estimate_coherence,
    welch_segment_count,
    welch_segment_length,
)

SAMPLE_RATE = 48_000
COHERENCE_TOLERANCE = 0.10
CELLO_FILE = Path(__file__).with_name("AuSep_2_vc_01_Jupiter.wav")
NOISE_RATIOS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def _white(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(round(seconds * SAMPLE_RATE))


def _pan(mono: np.ndarray, left_gain: float, right_gain: float) -> np.ndarray:
    return np.column_stack((mono * left_gain, mono * right_gain))


def _read_cello() -> np.ndarray:
    if not CELLO_FILE.exists():
        pytest.skip("optional URMP cello fixture not present")
    audio, sample_rate = sf.read(CELLO_FILE, dtype="float64")
    assert sample_rate == SAMPLE_RATE
    return audio


def test_single_segment_coherence_is_identically_one():
    length = 8192
    rng = np.random.default_rng(11)
    left, right = rng.standard_normal(length), rng.standard_normal(length)

    _, gamma_sq = scipy_coherence(left, right, fs=SAMPLE_RATE, nperseg=length)

    assert np.allclose(gamma_sq, 1.0)


def test_estimator_averages_enough_segments_to_escape_the_trap():
    for num_samples in (5_000, 9_600, 48_000, 200_000, 3_000_000):
        segment_length = welch_segment_length(num_samples)
        assert welch_segment_count(num_samples, segment_length) >= MIN_SEGMENTS


def test_estimator_refuses_audio_too_short_to_average():
    with pytest.raises(ValueError, match="segments"):
        estimate_coherence(np.zeros((2_000, 2)), SAMPLE_RATE)


def test_independent_channels_are_not_dragged_to_one_by_the_estimator():
    rng = np.random.default_rng(11)
    stereo = np.column_stack(
        (rng.standard_normal(3 * SAMPLE_RATE), rng.standard_normal(3 * SAMPLE_RATE))
    )

    result = estimate_coherence(stereo, SAMPLE_RATE)

    assert result.segments >= MIN_SEGMENTS
    assert result.value == pytest.approx(0.0, abs=COHERENCE_TOLERANCE)


def test_panned_source_reads_one():
    mono = _white(3.0, seed=1)

    result = estimate_coherence(_pan(mono, 0.9, 0.4), SAMPLE_RATE)

    assert result.value == pytest.approx(1.0, abs=COHERENCE_TOLERANCE)


@pytest.mark.parametrize("angle_degrees", [-25.0, -15.0, 0.0, 15.0, 25.0])
def test_coherence_ignores_the_pan_angle(angle_degrees: float):
    mono = _white(3.0, seed=2)
    difference = math.tan(math.radians(angle_degrees)) / math.tan(math.radians(30.0))
    left, right = (1.0 + difference) / 2.0, (1.0 - difference) / 2.0

    result = estimate_coherence(_pan(mono, left, right), SAMPLE_RATE)

    assert result.value == pytest.approx(1.0, abs=COHERENCE_TOLERANCE)


def test_two_unrelated_recordings_read_zero():
    stereo = np.column_stack((_white(3.0, seed=3), _white(3.0, seed=4)))

    result = estimate_coherence(stereo, SAMPLE_RATE)

    assert result.value == pytest.approx(0.0, abs=COHERENCE_TOLERANCE)


@pytest.mark.parametrize("ratio", NOISE_RATIOS)
def test_noise_ratio_matches_the_closed_form(ratio: float):
    rng = np.random.default_rng(5)
    length = 3 * SAMPLE_RATE
    source = rng.standard_normal(length)
    scale = math.sqrt(ratio)
    left = 0.9 * (source + scale * rng.standard_normal(length))
    right = 0.4 * (source + scale * rng.standard_normal(length))

    result = estimate_coherence(np.column_stack((left, right)), SAMPLE_RATE)

    assert result.value == pytest.approx(
        1.0 / (1.0 + ratio) ** 2, abs=COHERENCE_TOLERANCE
    )


def test_coherence_falls_as_the_noise_knob_turns():
    rng = np.random.default_rng(6)
    length = 3 * SAMPLE_RATE
    source = rng.standard_normal(length)
    noise_left, noise_right = rng.standard_normal(length), rng.standard_normal(length)

    measured = [
        estimate_coherence(
            np.column_stack(
                (
                    source + math.sqrt(ratio) * noise_left,
                    source + math.sqrt(ratio) * noise_right,
                )
            ),
            SAMPLE_RATE,
        ).value
        for ratio in NOISE_RATIOS
    ]

    assert all(later < earlier for earlier, later in zip(measured, measured[1:]))


def test_predicted_coherence_matches_the_formula():
    assert predicted_coherence(0.0) == 1.0
    assert predicted_coherence(1.0) == 0.25
    assert predicted_coherence(3.0) == pytest.approx(1.0 / 16.0)


def test_decorrelated_noise_has_the_requested_power():
    rng = np.random.default_rng(8)
    source = _white(1.0, seed=9)

    noise = decorrelated_noise(source, 0.4, rng)

    assert float(np.mean(noise**2)) == pytest.approx(
        0.4 * float(np.mean(source**2)), rel=1e-9
    )


def test_decorrelated_noise_keeps_the_spectrum_but_loses_the_waveform():
    rng = np.random.default_rng(10)
    source = _white(1.0, seed=12)

    noise = decorrelated_noise(source, 1.0, rng)

    source_magnitude = np.abs(np.fft.rfft(source))
    noise_magnitude = np.abs(np.fft.rfft(noise))
    assert np.allclose(noise_magnitude, source_magnitude, rtol=1e-6, atol=1e-9)

    correlation = float(
        np.dot(source, noise) / (np.linalg.norm(source) * np.linalg.norm(noise))
    )
    assert abs(correlation) < 0.05


def test_decorrelated_noise_is_deterministic_for_a_seed():
    source = _white(0.5, seed=13)

    first = decorrelated_noise(source, 0.7, np.random.default_rng(2026))
    second = decorrelated_noise(source, 0.7, np.random.default_rng(2026))

    assert np.array_equal(first, second)


def test_unrelated_pair_matches_channel_levels():
    first, second = _white(1.0, seed=14), 0.01 * _white(1.0, seed=15)

    stereo = unrelated_pair(first, second)

    left_rms = float(np.sqrt(np.mean(stereo[:, 0] ** 2)))
    right_rms = float(np.sqrt(np.mean(stereo[:, 1] ** 2)))
    assert right_rms == pytest.approx(left_rms, rel=1e-9)


@pytest.mark.parametrize("ratio", [0.0, *NOISE_RATIOS])
def test_generated_files_measure_back_their_predicted_coherence(ratio: float):
    mono = _read_cello()

    stereo = panned_with_noise(mono, 15.0, ratio, seed=2026)
    result = estimate_coherence(stereo, SAMPLE_RATE)

    assert result.value == pytest.approx(
        predicted_coherence(ratio), abs=COHERENCE_TOLERANCE
    )


def test_real_recording_panned_reads_one():
    mono = _read_cello()

    result = estimate_coherence(_pan(mono, 0.9, 0.4), SAMPLE_RATE)

    assert result.value == pytest.approx(1.0, abs=COHERENCE_TOLERANCE)
    assert MIN_BAND_HZ < MAX_BAND_HZ
    assert result.active_bins > 100


def test_two_unrelated_passages_of_the_real_recording_read_zero():
    mono = _read_cello()
    half = mono.size // 2

    stereo = unrelated_pair(mono[:half], mono[half : 2 * half])
    result = estimate_coherence(stereo, SAMPLE_RATE)

    assert result.value == pytest.approx(0.0, abs=COHERENCE_TOLERANCE)


def test_generator_and_estimator_command_lines_agree(tmp_path: Path):
    tests_dir = Path(__file__).parent
    source = tmp_path / "cli_source.wav"
    rng = np.random.default_rng(17)
    sf.write(source, rng.standard_normal(3 * SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_24")

    generated = subprocess.run(
        [
            sys.executable,
            str(tests_dir / "create_wav_for_coherence.py"),
            str(source),
            "-a",
            "15",
            "-n",
            "1.0",
            "-o",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tests_dir.parent,
    )
    assert "expected coherence 0.2500" in generated.stdout

    files = sorted((tmp_path / "out").glob("*.wav"))
    assert len(files) == 2

    measured = subprocess.run(
        [sys.executable, str(tests_dir / "estimate_coherence.py"), *map(str, files)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tests_dir.parent,
    )
    values = [
        float(line.split("coherence=")[1].split(",")[0])
        for line in measured.stdout.splitlines()
        if "coherence=" in line
    ]
    assert len(values) == 2
    noisy, panned = sorted(values)
    assert panned == pytest.approx(1.0, abs=COHERENCE_TOLERANCE)
    assert noisy == pytest.approx(0.25, abs=COHERENCE_TOLERANCE)


def test_unrelated_recordings_command_line(tmp_path: Path):
    tests_dir = Path(__file__).parent
    rng = np.random.default_rng(23)
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    sf.write(first, rng.standard_normal(3 * SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_24")
    sf.write(second, rng.standard_normal(3 * SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_24")

    generated = subprocess.run(
        [
            sys.executable,
            str(tests_dir / "create_wav_for_coherence.py"),
            str(first),
            "-u",
            str(second),
            "-o",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tests_dir.parent,
    )
    assert "expected coherence 0.0000" in generated.stdout

    unrelated = sorted((tmp_path / "out").glob("*unrelated*.wav"))
    assert len(unrelated) == 1

    measured = subprocess.run(
        [sys.executable, str(tests_dir / "estimate_coherence.py"), str(unrelated[0])],
        capture_output=True,
        text=True,
        check=True,
        cwd=tests_dir.parent,
    )
    value = float(measured.stdout.split("coherence=")[1].split(",")[0])
    assert value == pytest.approx(0.0, abs=COHERENCE_TOLERANCE)


def test_unrelated_inputs_must_share_a_sample_rate(tmp_path: Path):
    tests_dir = Path(__file__).parent
    rng = np.random.default_rng(24)
    first, second = tmp_path / "first.wav", tmp_path / "slow.wav"
    sf.write(first, rng.standard_normal(3 * SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_24")
    sf.write(second, rng.standard_normal(3 * 44_100), 44_100, subtype="PCM_24")

    result = subprocess.run(
        [
            sys.executable,
            str(tests_dir / "create_wav_for_coherence.py"),
            str(first),
            "-u",
            str(second),
            "-o",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=tests_dir.parent,
    )

    assert result.returncode != 0
