from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from create_wav_for_pitch import (
    note_sequence,
    read_f0_annotation,
    write_f0_annotation,
)
from cues.pitch import (
    F0_TOLERANCE_PERCENT,
    INSTRUMENT_RANGE_HZ,
    PITCH_SAMPLE_RATE,
    PitchTrack,
    align_to_times,
    compare_to_reference,
    frame_length_for,
    instrument_range_hz,
    percent_error,
    track_instrument_pitch,
    track_pitch,
)

SAMPLE_RATE = 48_000
CELLO_RANGE_HZ = INSTRUMENT_RANGE_HZ["vc"]
FLUTE_RANGE_HZ = INSTRUMENT_RANGE_HZ["fl"]
CELLO_FILE = Path(__file__).with_name("AuSep_2_vc_01_Jupiter.wav")
URMP_F0_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "URMP"
    / "01_Jupiter_vn_vc"
    / "F0s_2_vc_01_Jupiter.txt"
)

OPEN_STRINGS_HZ = (65.41, 98.00, 146.83, 220.00)


def _tone(f0_hz: float, seconds: float, harmonics: int = 6) -> np.ndarray:
    times = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = np.zeros_like(times)
    for partial in range(1, harmonics + 1):
        tone += np.sin(2.0 * np.pi * partial * f0_hz * times) / partial
    return 0.5 * tone / float(np.max(np.abs(tone)))


def _median_tracked_hz(track: PitchTrack) -> float:
    voiced = track.f0_hz[track.voiced]
    assert voiced.size > 0, "nothing was tracked as voiced"
    return float(np.median(voiced))


@pytest.mark.parametrize("f0_hz", OPEN_STRINGS_HZ)
def test_held_note_recovers_its_planted_frequency(f0_hz: float):
    track = track_pitch(_tone(f0_hz, 1.5), SAMPLE_RATE, *CELLO_RANGE_HZ)

    recovered = _median_tracked_hz(track)

    assert abs(recovered - f0_hz) / f0_hz * 100.0 < F0_TOLERANCE_PERCENT


def test_glide_is_followed_frame_by_frame():
    start_hz, end_hz, seconds = 110.0, 220.0, 3.0
    times = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    ratio = end_hz / start_hz
    instantaneous = start_hz * ratio ** (times / seconds)
    phase = np.cumsum(instantaneous) * (2.0 * np.pi / SAMPLE_RATE)
    audio = 0.5 * sum(np.sin(partial * phase) / partial for partial in (1, 2, 3, 4))

    track = track_pitch(audio, SAMPLE_RATE, *CELLO_RANGE_HZ)

    expected = start_hz * ratio ** (track.times_seconds / seconds)
    inside = track.voiced & (track.times_seconds > 0.1) & (
        track.times_seconds < seconds - 0.1
    )
    assert np.count_nonzero(inside) > 100
    errors = percent_error(track.f0_hz[inside], expected[inside])
    assert float(np.median(errors)) < F0_TOLERANCE_PERCENT
    assert float(np.mean(errors <= F0_TOLERANCE_PERCENT)) > 0.9


def test_frame_times_land_exactly_on_the_requested_hop():
    track = track_pitch(_tone(146.83, 1.0), SAMPLE_RATE, *CELLO_RANGE_HZ)

    assert track.sample_rate == PITCH_SAMPLE_RATE
    assert track.hop_seconds == pytest.approx(0.01, abs=1e-12)
    assert np.allclose(
        track.times_seconds, np.arange(track.f0_hz.size) * 0.01, atol=1e-12
    )


def test_tracking_works_from_the_projects_48_kilohertz_audio():
    track = track_pitch(_tone(196.0, 1.5), SAMPLE_RATE, *CELLO_RANGE_HZ)

    assert _median_tracked_hz(track) == pytest.approx(196.0, rel=0.03)


def test_silence_is_reported_unvoiced_instead_of_garbage():
    audio = np.concatenate(
        (_tone(146.83, 1.0), np.zeros(SAMPLE_RATE), _tone(146.83, 1.0))
    )

    track = track_pitch(audio, SAMPLE_RATE, *CELLO_RANGE_HZ)

    silent = (track.times_seconds > 1.1) & (track.times_seconds < 1.9)
    assert np.count_nonzero(silent) > 50
    assert not np.any(track.voiced[silent])
    assert np.all(np.isnan(track.f0_hz[silent]))


def test_a_file_of_pure_silence_tracks_nothing():
    track = track_pitch(np.zeros(2 * SAMPLE_RATE), SAMPLE_RATE, *CELLO_RANGE_HZ)

    assert not np.any(track.voiced)
    assert np.all(np.isnan(track.f0_hz))


def test_quiet_room_tone_is_below_the_silence_floor():
    rng = np.random.default_rng(21)
    room = 1e-5 * rng.standard_normal(SAMPLE_RATE)
    audio = np.concatenate((_tone(98.0, 1.0), room))

    track = track_pitch(audio, SAMPLE_RATE, *CELLO_RANGE_HZ)

    quiet = track.times_seconds > 1.1
    assert not np.any(track.voiced[quiet])


def test_a_pitch_above_the_search_range_comes_back_an_octave_down():
    audio = _tone(880.0, 2.0)

    right = track_pitch(audio, SAMPLE_RATE, *FLUTE_RANGE_HZ)
    wrong = track_pitch(audio, SAMPLE_RATE, *INSTRUMENT_RANGE_HZ["db"])

    assert _median_tracked_hz(right) == pytest.approx(880.0, rel=0.03)
    assert _median_tracked_hz(wrong) == pytest.approx(220.0, rel=0.03)
    assert np.all(wrong.voiced)


def test_a_pitch_below_the_search_range_is_pinned_to_its_bottom_edge():
    audio = _tone(110.0, 2.0)

    right = track_pitch(audio, SAMPLE_RATE, *CELLO_RANGE_HZ)
    wrong = track_pitch(audio, SAMPLE_RATE, *FLUTE_RANGE_HZ)

    assert _median_tracked_hz(right) == pytest.approx(110.0, rel=0.03)
    assert _median_tracked_hz(wrong) == pytest.approx(FLUTE_RANGE_HZ[0], rel=0.01)


def test_instrument_ranges_cover_their_instruments():
    assert instrument_range_hz("vc")[0] < min(OPEN_STRINGS_HZ)
    assert instrument_range_hz("VC") == instrument_range_hz("vc")

    assert instrument_range_hz("fl")[0] > OPEN_STRINGS_HZ[0]
    for code, (low, high) in INSTRUMENT_RANGE_HZ.items():
        assert 0.0 < low < high < PITCH_SAMPLE_RATE / 2.0, code


def test_unknown_instrument_code_is_rejected():
    with pytest.raises(ValueError, match="unknown instrument code"):
        instrument_range_hz("kazoo")


def test_frame_length_holds_several_periods_of_the_lowest_pitch():
    for _, (low, _) in INSTRUMENT_RANGE_HZ.items():
        frame_length = frame_length_for(low, PITCH_SAMPLE_RATE)
        assert frame_length >= 4.0 * PITCH_SAMPLE_RATE / low
        assert frame_length & (frame_length - 1) == 0


def test_track_instrument_pitch_picks_the_range_by_code():
    audio = _tone(98.0, 1.5)

    track = track_instrument_pitch(audio, SAMPLE_RATE, "vc")

    assert (track.fmin_hz, track.fmax_hz) == CELLO_RANGE_HZ
    assert _median_tracked_hz(track) == pytest.approx(98.0, rel=0.03)


def _ramp_track(num_frames: int, hop: float = 0.01) -> PitchTrack:
    f0 = 100.0 + np.arange(num_frames, dtype=np.float64)
    return PitchTrack(
        times_seconds=np.arange(num_frames) * hop,
        f0_hz=f0,
        voiced=np.ones(num_frames, dtype=bool),
        voiced_probability=np.ones(num_frames),
        hop_seconds=hop,
        sample_rate=PITCH_SAMPLE_RATE,
        fmin_hz=50.0,
        fmax_hz=500.0,
    )


def test_align_to_times_snaps_to_the_nearest_frame():
    track = _ramp_track(10)

    aligned = align_to_times(track, np.array([0.0, 0.0049, 0.0051, 0.023, 0.09]))

    assert np.allclose(aligned, [100.0, 100.0, 101.0, 102.0, 109.0])


def test_align_to_times_handles_urmps_offset_grid():
    track = _ramp_track(100)
    reference_times = 0.023 + np.arange(90) * 0.01

    aligned = align_to_times(track, reference_times)


    assert np.allclose(aligned, 102.0 + np.arange(90))


def test_align_to_times_reports_missing_frames_rather_than_borrowing():
    track = _ramp_track(10)

    aligned = align_to_times(track, np.array([-0.5, 0.05, 5.0]))

    assert math.isnan(aligned[0])
    assert aligned[1] == 105.0
    assert math.isnan(aligned[2])


def test_a_one_frame_shift_is_what_makes_a_good_tracker_look_bad():
    seconds = 3.0
    start_hz, end_hz = 110.0, 220.0
    times = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    ratio = end_hz / start_hz
    phase = np.cumsum(start_hz * ratio ** (times / seconds)) * (
        2.0 * np.pi / SAMPLE_RATE
    )
    audio = 0.5 * sum(np.sin(partial * phase) / partial for partial in (1, 2, 3, 4))
    track = track_pitch(audio, SAMPLE_RATE, *CELLO_RANGE_HZ)

    reference_times = np.arange(20, 280) * 0.01
    reference_f0 = start_hz * ratio ** (reference_times / seconds)

    aligned = compare_to_reference(track, reference_times, reference_f0)
    shifted = compare_to_reference(track, reference_times + 0.5, reference_f0)

    assert aligned.median_abs_percent_error < F0_TOLERANCE_PERCENT
    assert shifted.median_abs_percent_error > aligned.median_abs_percent_error


def test_percent_error_matches_its_definition():
    estimated = np.array([110.0, 216.0])
    reference = np.array([100.0, 220.0])

    assert np.allclose(percent_error(estimated, reference), [10.0, 100.0 * 4.0 / 220.0])


def test_percent_error_rejects_a_zero_reference():
    with pytest.raises(ValueError, match="positive"):
        percent_error(np.array([100.0]), np.array([0.0]))


def test_compare_to_reference_scores_a_planted_annotation():
    audio, times, f0 = note_sequence(OPEN_STRINGS_HZ, note_seconds=1.0, gap_seconds=0.3)

    track = track_pitch(audio, SAMPLE_RATE, *CELLO_RANGE_HZ)
    result = compare_to_reference(track, times, f0)

    assert result.median_abs_percent_error < F0_TOLERANCE_PERCENT
    assert result.within_tolerance == 1.0

    assert result.frames_compared == result.reference_voiced_frames > 300


    assert result.voicing_agreement > 0.85


def test_annotation_round_trips_through_the_urmp_text_format(tmp_path: Path):
    times = np.array([0.023, 0.033, 0.043])
    f0 = np.array([0.0, 138.925, 139.194])
    path = tmp_path / "F0s_planted.txt"

    write_f0_annotation(path, times, f0)
    read_times, read_f0 = read_f0_annotation(path)

    assert np.allclose(read_times, times)
    assert np.allclose(read_f0, f0)


def test_compare_to_reference_needs_overlapping_voiced_frames():
    track = _ramp_track(10)

    with pytest.raises(ValueError, match="both the track and the reference"):
        compare_to_reference(track, np.arange(10) * 0.01, np.zeros(10))


def test_compare_to_reference_rejects_mismatched_arrays():
    track = _ramp_track(10)

    with pytest.raises(ValueError, match="same shape"):
        compare_to_reference(track, np.arange(10) * 0.01, np.ones(9) * 110.0)


@pytest.mark.skipif(
    not URMP_F0_FILE.is_file(),
    reason=f"URMP annotation not present at {URMP_F0_FILE}",
)
def test_matches_urmp_annotation_on_the_real_cello_stem():
    start_seconds, end_seconds = 10.0, 30.0
    audio, sample_rate = sf.read(CELLO_FILE, dtype="float64")
    excerpt = audio[round(start_seconds * sample_rate) : round(end_seconds * sample_rate)]
    times, f0 = read_f0_annotation(URMP_F0_FILE)
    inside = (times >= start_seconds) & (times < end_seconds)

    track = track_instrument_pitch(excerpt, sample_rate, "vc")
    result = compare_to_reference(track, times[inside] - start_seconds, f0[inside])

    assert result.frames_compared > 1_000
    assert result.median_abs_percent_error < F0_TOLERANCE_PERCENT
    assert result.within_tolerance > 0.9


def test_pitch_command_lines_agree(tmp_path: Path):
    tests_dir = Path(__file__).parent

    subprocess.run(
        [
            sys.executable,
            str(tests_dir / "create_wav_for_pitch.py"),
            "-f",
            "146.83",
            "-f",
            "220",
            "--note-seconds",
            "1.0",
            "-o",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tests_dir.parent,
    )
    audio_files = sorted(tmp_path.glob("*.wav"))
    assert len(audio_files) == 1

    measured = subprocess.run(
        [
            sys.executable,
            str(tests_dir / "estimate_pitch.py"),
            str(audio_files[0]),
            "--instrument",
            "vc",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tests_dir.parent,
    )

    assert "instrument=vc" in measured.stdout
    assert "F0s_" in measured.stdout
    median_error = float(measured.stdout.split("median error=")[1].split("%")[0])
    assert median_error < F0_TOLERANCE_PERCENT


def test_a_urmp_stem_finds_its_own_annotation_and_instrument(tmp_path: Path):
    tests_dir = Path(__file__).parent
    audio_path = tmp_path / "AuSep_1_vc_09_Jesus.wav"
    annotation_path = tmp_path / "F0s_1_vc_09_Jesus.txt"

    audio, times, f0 = note_sequence((98.0, 146.83), note_seconds=1.0, gap_seconds=0.2)
    sf.write(audio_path, audio, SAMPLE_RATE, subtype="PCM_24")
    write_f0_annotation(annotation_path, times, f0)

    measured = subprocess.run(
        [sys.executable, str(tests_dir / "estimate_pitch.py"), str(audio_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tests_dir.parent,
    )

    assert "instrument=vc" in measured.stdout
    assert annotation_path.name in measured.stdout
    median_error = float(measured.stdout.split("median error=")[1].split("%")[0])
    assert median_error < F0_TOLERANCE_PERCENT


def test_an_unidentifiable_file_asks_for_the_instrument(tmp_path: Path):
    tests_dir = Path(__file__).parent
    audio_path = tmp_path / "mystery.wav"
    sf.write(audio_path, _tone(146.83, 0.5), SAMPLE_RATE, subtype="PCM_24")

    result = subprocess.run(
        [sys.executable, str(tests_dir / "estimate_pitch.py"), str(audio_path)],
        capture_output=True,
        text=True,
        cwd=tests_dir.parent,
    )

    assert result.returncode != 0
    assert "pass --instrument" in result.stdout
