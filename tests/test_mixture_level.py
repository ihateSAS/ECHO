from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from cues.level import (  # noqa: E402
    azimuth_from_ild_db,
    estimate_level_cues,
    ild_db_for_azimuth,
)
from cues.mixture import (  # noqa: E402
    AZIMUTH_TOLERANCE_DEG,
    MIN_ENERGY_FRACTION,
    MIN_PEAK_SEPARATION_DB,
    estimate_mixture_level_cues,
    find_peaks,
    ild_density,
)
from spatialize.amplitude import render_amplitude_pan  # noqa: E402
from spatialize.mixture import StemPlacement, mix_stems  # noqa: E402

SAMPLE_RATE = 48_000
URMP_ROOT = Path(__file__).resolve().parent.parent / "data" / "URMP" / "01_Jupiter_vn_vc"
VIOLIN = URMP_ROOT / "AuSep_1_vn_01_Jupiter.wav"
CELLO = URMP_ROOT / "AuSep_2_vc_01_Jupiter.wav"


def _harmonic(
    fundamental_hz: float,
    seconds: float = 2.0,
    seed: int = 0,
    rhythm: tuple[float, float, float] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for harmonic in range(1, 7):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += np.sin(2.0 * np.pi * fundamental_hz * harmonic * t + phase) / harmonic
    signal = 0.3 * signal / np.max(np.abs(signal))

    if rhythm is not None:
        note_s, rest_s, offset_s = rhythm
        gate = (((t + offset_s) % (note_s + rest_s)) < note_s).astype(np.float64)


        taper = round(0.005 * SAMPLE_RATE)
        gate = np.convolve(gate, np.ones(taper) / taper, mode="same")
        signal = signal * gate
    return signal


def _recovered(mix: np.ndarray, **kwargs) -> list[float]:
    estimate = estimate_mixture_level_cues(mix, SAMPLE_RATE, **kwargs)
    return sorted(source.azimuth_deg for source in estimate.sources)


def _urmp_excerpt(path: Path, seconds: float = 8.0, skip: float = 3.0) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float64")
    assert sample_rate == SAMPLE_RATE
    start = round(skip * sample_rate)
    return audio[start : start + round(seconds * sample_rate)]


needs_urmp = pytest.mark.skipif(
    not (VIOLIN.is_file() and CELLO.is_file()),
    reason=f"URMP stems not present under {URMP_ROOT}",
)


@pytest.mark.parametrize("azimuth_deg", [-25.0, -10.0, 0.0, 5.0, 17.5, 29.0])
def test_the_level_difference_and_the_angle_are_exact_inverses(azimuth_deg: float):
    assert azimuth_from_ild_db(ild_db_for_azimuth(azimuth_deg)) == pytest.approx(
        azimuth_deg, abs=1e-9
    )


def test_a_pan_at_the_speaker_angle_has_no_finite_level_difference():
    with pytest.raises(ValueError, match="must stay under"):
        ild_db_for_azimuth(30.0)


def test_the_resolution_limit_is_the_papers_tolerance_not_a_tuned_number():
    assert MIN_PEAK_SEPARATION_DB == pytest.approx(
        ild_db_for_azimuth(AZIMUTH_TOLERANCE_DEG)
    )
    assert MIN_PEAK_SEPARATION_DB == pytest.approx(2.653, abs=0.001)


@pytest.mark.parametrize("azimuth_deg", [-25.0, -8.0, 0.0, 12.0, 25.0])
def test_a_single_source_gives_exactly_one_peak_at_the_single_source_answer(
    azimuth_deg: float,
):
    stereo = render_amplitude_pan(_harmonic(196.0), azimuth_deg)
    estimate = estimate_mixture_level_cues(stereo, SAMPLE_RATE)

    assert estimate.source_count == 1
    assert estimate.sources[0].azimuth_deg == pytest.approx(azimuth_deg, abs=0.5)
    assert estimate.sources[0].energy_fraction == pytest.approx(1.0, abs=0.01)
    assert estimate.sources[0].azimuth_deg == pytest.approx(
        estimate_level_cues(stereo, SAMPLE_RATE).azimuth_deg, abs=0.5
    )


def test_two_synthetic_sources_both_come_back():
    mix, ground_truth = mix_stems(
        [
            StemPlacement(_harmonic(220.0, seed=1), 25.0, "low"),
            StemPlacement(_harmonic(311.13, seed=2), -25.0, "high"),
        ]
    )
    planted = sorted(item.azimuth_deg for item in ground_truth)

    recovered = _recovered(mix)
    assert len(recovered) == 2
    for got, want in zip(recovered, planted):
        assert got == pytest.approx(want, abs=AZIMUTH_TOLERANCE_DEG)


    single = estimate_level_cues(mix, SAMPLE_RATE).azimuth_deg
    assert min(abs(single - want) for want in planted) > AZIMUTH_TOLERANCE_DEG


def test_three_synthetic_sources_all_come_back():
    mix, ground_truth = mix_stems(
        [
            StemPlacement(
                _harmonic(174.61, seed=1, rhythm=(0.25, 0.15, 0.0)), -22.0, "one"
            ),
            StemPlacement(
                _harmonic(246.94, seed=2, rhythm=(0.20, 0.13, 0.07)), 3.0, "two"
            ),
            StemPlacement(
                _harmonic(329.63, seed=3, rhythm=(0.17, 0.11, 0.13)), 21.0, "three"
            ),
        ]
    )
    planted = sorted(item.azimuth_deg for item in ground_truth)

    recovered = _recovered(mix)
    assert len(recovered) == 3
    for got, want in zip(recovered, planted):
        assert got == pytest.approx(want, abs=AZIMUTH_TOLERANCE_DEG)


@needs_urmp
@pytest.mark.parametrize(
    "violin_deg,cello_deg", [(20.0, -20.0), (25.0, -10.0), (15.0, 5.0)]
)
def test_two_real_instruments_both_come_back(violin_deg: float, cello_deg: float):
    mix, _ = mix_stems(
        [
            StemPlacement(_urmp_excerpt(VIOLIN), violin_deg, "violin"),
            StemPlacement(_urmp_excerpt(CELLO), cello_deg, "cello"),
        ]
    )
    planted = sorted((violin_deg, cello_deg))

    recovered = _recovered(mix)
    assert len(recovered) == 2, recovered
    for got, want in zip(recovered, planted):
        assert got == pytest.approx(want, abs=AZIMUTH_TOLERANCE_DEG)


@needs_urmp
def test_the_single_source_estimator_still_reports_one_wrong_number():
    mix, _ = mix_stems(
        [
            StemPlacement(_urmp_excerpt(VIOLIN), 20.0, "violin"),
            StemPlacement(_urmp_excerpt(CELLO), -20.0, "cello"),
        ]
    )
    single = estimate_level_cues(mix, SAMPLE_RATE).azimuth_deg
    assert min(abs(single - want) for want in (20.0, -20.0)) > AZIMUTH_TOLERANCE_DEG


def test_two_sources_closer_than_the_tolerance_are_reported_as_one():
    mix, _ = mix_stems(
        [
            StemPlacement(_harmonic(220.0, seed=1), 10.0, "a"),
            StemPlacement(_harmonic(311.13, seed=2), 12.0, "b"),
        ]
    )
    recovered = _recovered(mix)
    assert len(recovered) == 1
    assert 10.0 <= recovered[0] <= 12.0


def test_sources_sharing_their_harmonics_produce_a_peak_between_them():
    mix, _ = mix_stems(
        [
            StemPlacement(_harmonic(220.0, seed=1), 25.0, "low"),
            StemPlacement(_harmonic(660.0, seed=2), -25.0, "high"),
        ]
    )
    recovered = _recovered(mix)

    assert any(angle == pytest.approx(25.0, abs=1.0) for angle in recovered)
    assert any(angle == pytest.approx(-25.0, abs=1.0) for angle in recovered)
    spurious = [
        angle for angle in recovered if abs(abs(angle) - 25.0) > AZIMUTH_TOLERANCE_DEG
    ]
    assert spurious, "the overlap peak is the thing this test exists to document"
    assert all(-25.0 < angle < 25.0 for angle in spurious)


def test_a_source_too_quiet_to_matter_is_not_reported():
    mix, _ = mix_stems(
        [
            StemPlacement(_harmonic(220.0, seed=1), 25.0, "loud"),
            StemPlacement(0.005 * _harmonic(311.13, seed=2), -25.0, "whisper"),
        ]
    )
    estimate = estimate_mixture_level_cues(mix, SAMPLE_RATE)

    assert estimate.source_count == 1
    assert estimate.sources[0].azimuth_deg == pytest.approx(25.0, abs=1.0)
    assert estimate.sources[0].energy_fraction > MIN_ENERGY_FRACTION


def test_max_sources_keeps_the_strongest():
    mix, _ = mix_stems(
        [
            StemPlacement(
                _harmonic(174.61, seed=1, rhythm=(0.25, 0.15, 0.0)), -22.0, "one"
            ),
            StemPlacement(
                _harmonic(246.94, seed=2, rhythm=(0.20, 0.13, 0.07)), 3.0, "two"
            ),
            StemPlacement(
                _harmonic(329.63, seed=3, rhythm=(0.17, 0.11, 0.13)), 21.0, "three"
            ),
        ]
    )
    everything = estimate_mixture_level_cues(mix, SAMPLE_RATE)
    capped = estimate_mixture_level_cues(mix, SAMPLE_RATE, max_sources=2)

    assert everything.source_count == 3
    assert capped.source_count == 2
    assert capped.sources == everything.sources[:2]


def test_sources_come_back_strongest_first():
    mix, _ = mix_stems(
        [
            StemPlacement(0.3 * _harmonic(220.0, seed=1), 20.0, "quiet"),
            StemPlacement(_harmonic(311.13, seed=2), -20.0, "loud"),
        ]
    )
    estimate = estimate_mixture_level_cues(mix, SAMPLE_RATE)

    fractions = [source.energy_fraction for source in estimate.sources]
    assert fractions == sorted(fractions, reverse=True)
    assert estimate.sources[0].azimuth_deg == pytest.approx(-20.0, abs=2.0)


def test_the_density_puts_its_mass_where_the_level_differences_are():
    ild_db = np.array([-6.0, -6.0, -6.0, 12.0, 12.0])
    energy = np.ones(5)
    grid, density = ild_density(ild_db, energy)

    assert grid[int(np.argmax(density))] == pytest.approx(-6.0, abs=0.25)

    assert 12.0 in [pytest.approx(grid[index], abs=0.25) for index in find_peaks(grid, density)]


def test_the_density_clamps_rather_than_drops_a_hard_panned_bin():
    grid, density = ild_density(np.array([200.0]), np.array([1.0]))
    assert density.sum() > 0.0
    assert grid[int(np.argmax(density))] == pytest.approx(grid[-1], abs=1.0)


def test_peaks_closer_than_the_separation_are_thinned_tallest_first():
    grid = np.arange(-10.0, 10.0, 0.25)
    density = np.zeros_like(grid)
    density[int(np.argmin(abs(grid - 0.0)))] = 1.0
    density[int(np.argmin(abs(grid - 1.0)))] = 2.0
    density[int(np.argmin(abs(grid - 6.0)))] = 0.5

    kept = sorted(grid[index] for index in find_peaks(grid, density))
    assert kept == [pytest.approx(1.0), pytest.approx(6.0)]


def test_an_empty_set_of_bins_is_refused():
    with pytest.raises(ValueError, match="no active bins"):
        ild_density(np.array([]), np.array([]))


def test_the_density_is_returned_so_a_caller_can_look_at_it():
    stereo = render_amplitude_pan(_harmonic(196.0), 10.0)
    estimate = estimate_mixture_level_cues(stereo, SAMPLE_RATE)

    assert estimate.grid_ild_db.shape == estimate.density.shape
    assert estimate.active_bins > 0
    peak_ild = estimate.grid_ild_db[int(np.argmax(estimate.density))]
    assert math.isclose(azimuth_from_ild_db(peak_ild), 10.0, abs_tol=1.0)
