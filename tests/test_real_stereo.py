from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.measure import measure_render  # noqa: E402
from benchmark.process import apply_processing_chain  # noqa: E402
from benchmark.schemas import ProcessingStep  # noqa: E402
from eval.real_stereo import (  # noqa: E402
    ANALYSIS_SAMPLE_RATE,
    AZIMUTH_TOLERANCE_DEG,
    CONTROL_CHAINS,
    QUIET_WINDOW_FLOOR_DB,
    control_recovery,
    loud_windows,
    mix_pitch,
    read_stereo,
    side_to_total_db,
    survey_track,
    window_starts,
)
from spatialize.amplitude import gains_for_azimuth, render_amplitude_pan  # noqa: E402
from spatialize.midside import apply_width  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEAKER_TANGENT = math.tan(math.radians(30.0))


def _musical_mono(seconds: float = 5.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * ANALYSIS_SAMPLE_RATE)) / ANALYSIS_SAMPLE_RATE
    signal = np.zeros_like(t)
    for fundamental in (110.0, 164.81, 220.0, 329.63):
        vibrato = 1.0 + 0.004 * np.sin(2.0 * np.pi * 5.3 * t)
        for harmonic in range(1, 7):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal += (
                np.sin(2.0 * np.pi * fundamental * harmonic * vibrato * t + phase)
                / harmonic
            )
    signal += 0.05 * rng.standard_normal(t.size)
    return 0.3 * signal / np.max(np.abs(signal))


def _planted_d(azimuth_deg: float) -> float:
    left, right = gains_for_azimuth(azimuth_deg)
    return (left - right) / (left + right)


def test_dual_mono_has_no_side_energy_at_all():
    mono = _musical_mono(1.0)
    doubled = np.column_stack((mono, mono))
    assert side_to_total_db(doubled) == float("-inf")


def test_a_panned_source_has_the_side_energy_its_gains_imply():
    left, right = gains_for_azimuth(25.0)
    expected_db = 10.0 * math.log10(
        2.0 * (left - right) ** 2 / (left**2 + right**2)
    )
    panned = render_amplitude_pan(_musical_mono(1.0), 25.0)
    assert abs(side_to_total_db(panned) - expected_db) < 0.05


def test_windows_are_contiguous_and_do_not_overlap():
    starts = window_starts(10 * ANALYSIS_SAMPLE_RATE, ANALYSIS_SAMPLE_RATE, 2.5)
    assert starts == [
        0,
        round(2.5 * ANALYSIS_SAMPLE_RATE),
        5 * ANALYSIS_SAMPLE_RATE,
        round(7.5 * ANALYSIS_SAMPLE_RATE),
    ]


def test_a_partial_final_window_is_dropped_rather_than_measured_short():
    starts = window_starts(7 * ANALYSIS_SAMPLE_RATE, ANALYSIS_SAMPLE_RATE, 2.0)
    assert starts == [0, 2 * ANALYSIS_SAMPLE_RATE, 4 * ANALYSIS_SAMPLE_RATE]


def test_a_silent_lead_in_is_left_out_of_the_windows():
    loud = render_amplitude_pan(_musical_mono(4.0), 0.0)
    silence = np.zeros((2 * ANALYSIS_SAMPLE_RATE, 2))
    track = np.vstack((silence, loud))
    selected = loud_windows(track, ANALYSIS_SAMPLE_RATE, window_seconds=2.0)
    assert [start for start, _ in selected] == [
        2 * ANALYSIS_SAMPLE_RATE,
        4 * ANALYSIS_SAMPLE_RATE,
    ]


def test_a_window_just_above_the_floor_is_kept():
    loud = render_amplitude_pan(_musical_mono(2.0), 0.0)
    quiet = loud * 10.0 ** ((QUIET_WINDOW_FLOOR_DB + 1.0) / 20.0)
    selected = loud_windows(
        np.vstack((loud, quiet)), ANALYSIS_SAMPLE_RATE, window_seconds=2.0
    )
    assert len(selected) == 2


def test_reading_a_file_resamples_without_touching_the_balance(tmp_path: Path):
    source_rate = 44_100
    t = np.arange(2 * source_rate) / source_rate
    mono = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    path = tmp_path / "planted.wav"
    sf.write(path, render_amplitude_pan(mono, 15.0), source_rate, subtype="PCM_24")

    stereo = read_stereo(path)
    assert stereo.shape[1] == 2
    assert abs(stereo.shape[0] - 2 * ANALYSIS_SAMPLE_RATE) <= 1
    evidence = measure_render(stereo, ANALYSIS_SAMPLE_RATE)
    assert abs(evidence.estimated_azimuth_deg - 15.0) < AZIMUTH_TOLERANCE_DEG


def test_a_mono_file_is_refused_rather_than_measured(tmp_path: Path):
    path = tmp_path / "mono.wav"
    sf.write(path, _musical_mono(1.0), ANALYSIS_SAMPLE_RATE, subtype="PCM_24")
    with pytest.raises(ValueError, match="expected stereo"):
        read_stereo(path)


def test_mix_pitch_reports_nan_rather_than_raising_on_a_short_clip():
    short = render_amplitude_pan(_musical_mono(0.05), 0.0)
    f0_hz, voiced_fraction = mix_pitch(short, ANALYSIS_SAMPLE_RATE)
    assert math.isnan(f0_hz)
    assert math.isnan(voiced_fraction)


def test_mix_pitch_returns_the_chord_root_nobody_is_playing():
    mono = _musical_mono(3.0)
    f0_hz, voiced_fraction = mix_pitch(
        np.column_stack((mono, mono)), ANALYSIS_SAMPLE_RATE
    )
    assert abs(f0_hz - 55.0) / 55.0 < 0.03
    assert 0.0 < voiced_fraction <= 1.0


def test_the_dry_control_recovers_every_planted_angle():
    stereo = render_amplitude_pan(_musical_mono(5.0), 0.0)
    readings = control_recovery(
        stereo,
        ANALYSIS_SAMPLE_RATE,
        start=0,
        angles_deg=(-25.0, -10.0, 0.0, 10.0, 25.0),
        chains={"dry": ()},
    )
    assert len(readings) == 5
    for reading in readings:
        assert reading.azimuth_error_deg < 1e-6
        assert reading.regime == "amplitude"


def test_stereo_linked_compression_leaves_the_planted_angle_alone():
    stereo = render_amplitude_pan(_musical_mono(5.0), 20.0)
    (reading,) = control_recovery(
        stereo,
        ANALYSIS_SAMPLE_RATE,
        start=0,
        angles_deg=(20.0,),
        chains={"compression": CONTROL_CHAINS["compression"]},
    )
    assert reading.azimuth_error_deg < AZIMUTH_TOLERANCE_DEG


def test_reverb_pulls_a_planted_angle_past_tolerance():
    stereo = render_amplitude_pan(_musical_mono(5.0), 25.0)
    (reading,) = control_recovery(
        stereo,
        ANALYSIS_SAMPLE_RATE,
        start=0,
        angles_deg=(25.0,),
        chains={"reverb": CONTROL_CHAINS["reverb"]},
    )
    assert reading.azimuth_error_deg > 4 * AZIMUTH_TOLERANCE_DEG
    assert abs(reading.recovered_azimuth_deg) < 5.0
    assert reading.coherence < 0.2


def test_mid_side_width_moves_the_angle_by_exactly_the_documented_amount():
    for width, angle in ((1.3, 10.0), (1.5, 10.0), (1.3, 20.0), (0.5, 25.0)):
        stereo = render_amplitude_pan(_musical_mono(5.0), angle)
        widened = apply_width(stereo, width)
        measured = measure_render(widened, ANALYSIS_SAMPLE_RATE)
        predicted = math.degrees(
            math.atan(width * _planted_d(angle) * SPEAKER_TANGENT)
        )
        assert width * _planted_d(angle) < 1.0, "case must stay in range"
        assert abs(measured.estimated_azimuth_deg - predicted) < 0.05


def test_a_polarity_inversion_reads_as_a_dead_gcc_phat_peak():
    angle, width = 25.0, 1.5
    assert width * _planted_d(angle) > 1.0, "case must be out of range"
    stereo = apply_width(render_amplitude_pan(_musical_mono(5.0), angle), width)
    measured = measure_render(stereo, ANALYSIS_SAMPLE_RATE)

    assert measured.itd_peak_strength < 0.05
    assert measured.coherence > 0.99

    assert abs(measured.estimated_azimuth_deg - angle) < AZIMUTH_TOLERANCE_DEG


def test_coherence_separates_a_reverberant_pair_from_a_clean_pan():
    mono = _musical_mono(5.0)
    panned = render_amplitude_pan(mono, 10.0)
    reverberant = apply_processing_chain(
        panned, ANALYSIS_SAMPLE_RATE, CONTROL_CHAINS["reverb"]
    )
    assert measure_render(panned, ANALYSIS_SAMPLE_RATE).coherence > 0.99
    assert measure_render(reverberant, ANALYSIS_SAMPLE_RATE).coherence < 0.5


def test_survey_track_measures_every_loud_window_and_runs_the_controls(
    tmp_path: Path,
):
    path = tmp_path / "planted.wav"
    stereo = render_amplitude_pan(_musical_mono(12.0), 10.0)
    sf.write(path, stereo, ANALYSIS_SAMPLE_RATE, subtype="PCM_24")

    survey = survey_track(
        path,
        window_seconds=4.0,
        control_angles_deg=(10.0,),
        control_chains={"dry": (), "width": CONTROL_CHAINS["width"]},
    )
    assert len(survey.windows) == 3
    assert survey.channels == 2
    mean, std, peak_to_peak = survey.spread("estimated_azimuth_deg")
    assert abs(mean - 10.0) < 0.5
    assert peak_to_peak < 0.5, "one steady pan should not wander between windows"
    assert std >= 0.0

    by_chain = survey.worst_control_error_by_chain()
    assert set(by_chain) == {"dry", "width"}
    assert by_chain["dry"] < 1e-6

    assert 4.0 < by_chain["width"] < 5.5


def test_the_survey_command_line_runs_and_writes_its_json(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(
        audio_dir / "one.wav",
        render_amplitude_pan(_musical_mono(6.0), -15.0),
        ANALYSIS_SAMPLE_RATE,
        subtype="PCM_24",
    )
    output = tmp_path / "out" / "result.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "survey_real_stereo.py"),
            str(audio_dir),
            "--window-seconds",
            "3",
            "--control-angles",
            "-15",
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "one.wav" in completed.stdout

    payload = json.loads(output.read_text())
    (track,) = payload["tracks"]
    assert track["windows_measured"] == 2
    assert abs(track["summary"]["azimuth_deg"]["mean"] + 15.0) < 0.5
    assert track["summary"]["worst_control_error_deg_by_chain"]["dry"] < 1e-6
    assert payload["azimuth_tolerance_deg"] == AZIMUTH_TOLERANCE_DEG


def test_the_command_line_says_so_rather_than_crashing_on_an_empty_directory(
    tmp_path: Path,
):
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "survey_real_stereo.py"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode != 0
    assert "no audio found" in completed.stderr


def test_the_processing_recipes_the_survey_uses_are_ones_the_benchmark_knows():
    for steps in CONTROL_CHAINS.values():
        for step in steps:
            assert isinstance(step, ProcessingStep)
            assert step.kind in {
                "reverb",
                "compression",
                "eq",
                "midside",
                "delay_widening",
            }
