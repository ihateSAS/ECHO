from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cues.delay import estimate_itd_us  # noqa: E402
from cues.level import estimate_level_cues  # noqa: E402
from cues.regime import classify_regime  # noqa: E402
from spatialize.amplitude import render_amplitude_pan  # noqa: E402
from spatialize.mixture import MixtureGroundTruth, StemPlacement, mix_stems  # noqa: E402

SAMPLE_RATE = 48_000
CELLO_PATH = Path(__file__).parent / "AuSep_2_vc_01_Jupiter.wav"


def _tone(frequency_hz: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    return 0.2 * np.sin(2.0 * np.pi * frequency_hz * t)


def _read_cello_excerpt(start_s: float, duration_s: float) -> np.ndarray:
    import soundfile as sf

    mono, sample_rate = sf.read(CELLO_PATH, dtype="float64", always_2d=True)
    assert sample_rate == SAMPLE_RATE
    start = int(start_s * sample_rate)
    frames = int(duration_s * sample_rate)
    return mono[start : start + frames, 0]


def test_single_stem_mixture_equals_plain_render():
    mono = _tone(440.0, 0.5)
    mix, ground_truth = mix_stems([StemPlacement(mono, 12.0, "solo")])

    expected = render_amplitude_pan(mono, 12.0)
    assert np.allclose(mix, expected, atol=1e-12)
    assert ground_truth == [MixtureGroundTruth("solo", 12.0)]


def test_mixture_equals_sum_of_independently_padded_renders():
    short = _tone(220.0, 0.3)
    long = _tone(330.0, 0.7)
    mix, _ = mix_stems(
        [StemPlacement(short, -15.0, "short"), StemPlacement(long, 20.0, "long")]
    )

    padded_short = np.pad(short, (0, long.size - short.size))
    expected = render_amplitude_pan(padded_short, -15.0) + render_amplitude_pan(long, 20.0)
    assert mix.shape == (long.size, 2)
    assert np.allclose(mix, expected, atol=1e-12)


def test_mixture_pads_shorter_stems_to_the_longest():
    short = _tone(440.0, 0.2)
    long = _tone(440.0, 0.9)
    mix, _ = mix_stems([StemPlacement(short, 0.0, "a"), StemPlacement(long, 0.0, "b")])
    assert mix.shape[0] == long.size


def test_mixture_ground_truth_preserves_order_and_labels():
    mono = _tone(500.0, 0.2)
    _, ground_truth = mix_stems(
        [
            StemPlacement(mono, -10.0, "violin"),
            StemPlacement(mono, 0.0, "viola"),
            StemPlacement(mono, 25.0, "cello"),
        ]
    )
    assert ground_truth == [
        MixtureGroundTruth("violin", -10.0),
        MixtureGroundTruth("viola", 0.0),
        MixtureGroundTruth("cello", 25.0),
    ]


def test_mixture_rejects_empty_placements():
    with pytest.raises(ValueError):
        mix_stems([])


def test_mixture_rejects_non_mono_stem():
    stereo_shaped = np.zeros((100, 2))
    with pytest.raises(ValueError):
        mix_stems([StemPlacement(stereo_shaped, 0.0, "bad")])


def _finite_azimuth_in_range(value: float) -> bool:
    return np.isfinite(value) and -30.0 <= value <= 30.0


def test_explore_two_synthetic_tones_summed():
    left_source = _tone(220.0, 1.0)
    right_source = _tone(660.0, 1.0)
    mix, ground_truth = mix_stems(
        [
            StemPlacement(left_source, 25.0, "220hz"),
            StemPlacement(right_source, -25.0, "660hz"),
        ]
    )
    level = estimate_level_cues(mix, SAMPLE_RATE)
    delay = estimate_itd_us(mix, SAMPLE_RATE)
    regime = classify_regime(level, delay)

    print(f"\n[mixture explore] planted: {ground_truth}")
    print(
        f"[mixture explore] measured: azimuth={level.azimuth_deg:.2f} deg, "
        f"ild={level.broadband_ild_db:.2f} dB, regime={regime.label}"
    )
    assert _finite_azimuth_in_range(level.azimuth_deg)


def test_explore_real_cello_mixed_with_itself_at_two_angles():
    if not CELLO_PATH.exists():
        pytest.skip("tests/AuSep_2_vc_01_Jupiter.wav not present")

    first = _read_cello_excerpt(0.0, 2.0)
    second = _read_cello_excerpt(10.0, 2.0)
    mix, ground_truth = mix_stems(
        [
            StemPlacement(first, 20.0, "cello_excerpt_a"),
            StemPlacement(second, -10.0, "cello_excerpt_b"),
        ]
    )
    level = estimate_level_cues(mix, SAMPLE_RATE)
    delay = estimate_itd_us(mix, SAMPLE_RATE)
    coherence_active_bins = level.active_bins

    print(f"\n[mixture explore, real audio] planted: {ground_truth}")
    print(
        f"[mixture explore, real audio] measured: azimuth={level.azimuth_deg:.2f} deg, "
        f"itd={delay.itd_us:.1f} us, active_bins={coherence_active_bins}"
    )
    assert _finite_azimuth_in_range(level.azimuth_deg)
    assert np.isfinite(delay.itd_us)
