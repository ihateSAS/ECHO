from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatialize.compression import (  # noqa: E402
    gain_reduction_db,
    stereo_linked_compressor,
)

SAMPLE_RATE = 48_000


def _reference_gain_reduction_db(level_db: float, threshold_db: float, ratio: float) -> float:
    if level_db <= threshold_db:
        return 0.0
    return (level_db - threshold_db) * (1.0 - 1.0 / ratio)


@pytest.mark.parametrize(
    "level_db,threshold_db,ratio",
    [(-10.0, -20.0, 4.0), (-5.0, -30.0, 2.0), (0.0, -18.0, 8.0), (-25.0, -20.0, 4.0)],
)
def test_gain_reduction_matches_independent_derivation(level_db, threshold_db, ratio):
    expected = _reference_gain_reduction_db(level_db, threshold_db, ratio)
    actual = gain_reduction_db(np.array([level_db]), threshold_db, ratio)[0]
    assert actual == pytest.approx(expected, abs=1e-9)


def test_rejects_ratio_below_one():
    with pytest.raises(ValueError):
        gain_reduction_db(np.array([0.0]), -20.0, ratio=0.5)


def test_rejects_mono_input():
    mono = np.zeros((1000, 1))
    with pytest.raises(ValueError):
        stereo_linked_compressor(mono, SAMPLE_RATE, threshold_db=-20.0, ratio=4.0)


def test_below_threshold_is_identity():
    n = int(1.0 * SAMPLE_RATE)
    quiet = 10 ** (-50 / 20.0) * np.sin(2 * np.pi * 300 * np.arange(n) / SAMPLE_RATE)
    stereo = np.column_stack((quiet, quiet))
    out = stereo_linked_compressor(stereo, SAMPLE_RATE, threshold_db=-20.0, ratio=4.0)
    assert np.array_equal(out, stereo)


@pytest.mark.parametrize("threshold_db,ratio", [(-20.0, 4.0), (-30.0, 2.0), (-40.0, 8.0)])
def test_ild_exactly_preserved(threshold_db, ratio):
    n = int(1.0 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    left, right = mono * 0.9, mono * 0.3
    stereo = np.column_stack((left, right))

    ild_before = 20 * np.log10(
        np.sqrt(np.mean(left**2)) / np.sqrt(np.mean(right**2))
    )
    out = stereo_linked_compressor(stereo, SAMPLE_RATE, threshold_db=threshold_db, ratio=ratio)
    ild_after = 20 * np.log10(
        np.sqrt(np.mean(out[:, 0] ** 2)) / np.sqrt(np.mean(out[:, 1] ** 2))
    )
    assert ild_after == pytest.approx(ild_before, abs=1e-9)


def test_static_gain_matches_planted_reduction():
    level_db = -10.0
    threshold_db, ratio = -20.0, 4.0
    amp = 10 ** (level_db / 20.0)
    n = int(2.0 * SAMPLE_RATE)
    square = amp * np.sign(np.sin(2 * np.pi * 5 * np.arange(n) / SAMPLE_RATE))
    stereo = np.column_stack((square, square))

    out = stereo_linked_compressor(
        stereo, SAMPLE_RATE, threshold_db=threshold_db, ratio=ratio, attack_s=0.005, release_s=0.05
    )
    settled_out = out[int(0.5 * n) :, 0]
    settled_in = stereo[int(0.5 * n) :, 0]
    measured_gain_db = 20 * np.log10(
        np.sqrt(np.mean(settled_out**2)) / np.sqrt(np.mean(settled_in**2))
    )
    expected_reduction = _reference_gain_reduction_db(level_db, threshold_db, ratio)
    assert measured_gain_db == pytest.approx(-expected_reduction, abs=0.05)


def test_attack_reaches_one_minus_1_over_e_after_one_time_constant():
    attack_s = 0.02
    threshold_db, ratio = -20.0, 4.0
    pre_roll = int(0.1 * SAMPLE_RATE)
    step = np.concatenate((np.zeros(pre_roll), np.ones(int(0.5 * SAMPLE_RATE))))
    stereo = np.column_stack((step, step))

    out = stereo_linked_compressor(
        stereo, SAMPLE_RATE, threshold_db=threshold_db, ratio=ratio, attack_s=attack_s, release_s=0.5
    )
    target_reduction = _reference_gain_reduction_db(0.0, threshold_db, ratio)
    index_at_one_tc = pre_roll + int(attack_s * SAMPLE_RATE)
    reduction_at_tc = -20 * np.log10(np.abs(out[index_at_one_tc, 0]))
    fraction_covered = reduction_at_tc / target_reduction
    assert fraction_covered == pytest.approx(1.0 - 1.0 / np.e, abs=0.02)


def test_deterministic_and_stable():
    n = int(0.5 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    mono = np.sin(2 * np.pi * 220 * t)
    stereo = np.column_stack((mono, mono * 0.4))
    first = stereo_linked_compressor(stereo, SAMPLE_RATE, threshold_db=-15.0, ratio=3.0)
    second = stereo_linked_compressor(stereo, SAMPLE_RATE, threshold_db=-15.0, ratio=3.0)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
