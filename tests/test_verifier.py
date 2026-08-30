from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.measure import measure_render  # noqa: E402
from benchmark.schemas import CueMeasurements  # noqa: E402
from create_wav_for_angle import read_mono  # noqa: E402
from spatialize.amplitude import render_amplitude_pan  # noqa: E402
from verifier import verify, verify_ablation_pair, verify_perturbation_pair  # noqa: E402

CELLO_PATH = Path(__file__).parent / "AuSep_2_vc_01_Jupiter.wav"


def _cues(**overrides) -> CueMeasurements:
    values = dict(
        ild_db=19.5,
        high_frequency_ild_db=19.5,
        ild_flatness_db=0.0,
        itd_us=0.0,
        itd_peak_strength=0.9,
        coherence=1.0,
        estimated_azimuth_deg=25.0,
        estimated_regime="amplitude",
        active_bins=500,
        f0_hz=float("nan"),
    )
    values.update(overrides)
    return CueMeasurements(**values)


def test_correct_azimuth_and_ild_claims_pass():
    transcript = (
        "The instrument is at 25 degrees to the left because the level "
        "difference is 19.5 dB with the left channel being louder."
    )
    result = verify(transcript, _cues())

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["estimated_azimuth_deg_side"].passed
    assert by_quantity["estimated_azimuth_deg_magnitude"].passed
    assert by_quantity["ild_db"].passed
    assert result.all_passed
    assert result.inference_passed


def test_correct_regime_and_coherence_claims_pass():
    transcript = "This is an amplitude-panned source with a coherence of 0.97."
    result = verify(transcript, _cues(coherence=0.98))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["estimated_regime"].passed
    assert by_quantity["coherence"].passed


def test_correct_delay_claim_passes():
    transcript = "There is a 400 microsecond delay, with the right channel later."
    result = verify(transcript, _cues(itd_us=395.0, estimated_regime="delayed"))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["itd_us"].passed


def test_correct_f0_claim_passes():
    transcript = "The fundamental frequency is about 146.8 Hz."
    result = verify(transcript, _cues(f0_hz=146.5))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["f0_hz"].passed


def test_wrong_azimuth_magnitude_fails():
    transcript = "The instrument is at 5 degrees to the left."
    result = verify(transcript, _cues(estimated_azimuth_deg=25.0))

    by_quantity = {c.quantity: c for c in result.checks}
    assert not by_quantity["estimated_azimuth_deg_magnitude"].passed
    assert by_quantity["estimated_azimuth_deg_side"].passed
    assert not result.all_passed
    assert not result.inference_passed


def test_wrong_azimuth_side_fails_even_though_a_naive_check_might_not():
    transcript = "The instrument is at 2 degrees to the right."
    result = verify(transcript, _cues(estimated_azimuth_deg=2.0))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["estimated_azimuth_deg_magnitude"].passed
    assert not by_quantity["estimated_azimuth_deg_side"].passed
    assert not result.inference_passed


def test_wrong_regime_fails():
    transcript = "This is an amplitude-panned source."
    result = verify(transcript, _cues(estimated_regime="delayed"))

    by_quantity = {c.quantity: c for c in result.checks}
    assert not by_quantity["estimated_regime"].passed
    assert not result.inference_passed


def test_fabricated_premise_r_fails_even_though_everything_else_passes():
    transcript = (
        "The inter-channel level difference is -14.8 dB, so the right "
        "channel is louder. Converting with the panning law: "
        "r = 10^(-14.8/20) = 0.139; d = (r - 1)/(r + 1) = -0.756; "
        "azimuth = arctan(d * tan 30deg) = -23.6 degrees. "
        "The trumpet is at 23.6 degrees to the right."
    )
    result = verify(transcript, _cues(ild_db=-14.8, estimated_azimuth_deg=-21.8))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["ild_db"].passed
    assert by_quantity["estimated_azimuth_deg_magnitude"].passed
    assert by_quantity["estimated_azimuth_deg_side"].passed
    assert not by_quantity["premise_r_matches_its_own_stated_cue"].passed
    assert not result.inference_passed
    assert not result.all_passed


def test_correct_premise_r_passes():
    transcript = (
        "The inter-channel level difference is -14.8 dB, so the right "
        "channel is louder. Converting with the panning law: "
        "r = 10^(-14.8/20) = 0.182; d = (r - 1)/(r + 1) = -0.692; "
        "azimuth = arctan(d * tan 30deg) = -21.4 degrees. "
        "The trumpet is at 21.4 degrees to the right."
    )
    result = verify(transcript, _cues(ild_db=-14.8, estimated_azimuth_deg=-21.8))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["premise_r_matches_its_own_stated_cue"].passed
    assert result.inference_passed
    assert result.all_passed


def test_premise_r_check_is_skipped_when_the_transcript_has_no_formula():
    transcript = "The instrument is at 25 degrees to the left."
    result = verify(transcript, _cues())

    assert "premise_r_matches_its_own_stated_cue" not in {
        c.quantity for c in result.checks
    }


def test_premise_r_tolerates_ordinary_display_rounding():
    transcript = (
        "Converting with the panning law: r = 10^(-14.8/20) = 0.182; "
        "azimuth = arctan(d * tan 30deg) = -21.4 degrees."
    )
    result = verify(transcript, _cues(ild_db=-14.8, estimated_azimuth_deg=-21.4))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["premise_r_matches_its_own_stated_cue"].passed


def test_read_out_passes_while_inference_fails():
    transcript = (
        "The level difference is 19.5 dB with the left channel being "
        "louder, so the instrument must be at 25 degrees to the right."
    )
    result = verify(transcript, _cues(ild_db=19.5, estimated_azimuth_deg=25.0))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["ild_db"].passed
    assert not by_quantity["estimated_azimuth_deg_side"].passed
    assert not result.inference_passed
    assert not result.all_passed


def test_qualitative_no_coherence_claim_is_extracted_and_fails_against_high_coherence():
    transcript = "There is no coherence between the two channels."
    result = verify(transcript, _cues(coherence=1.0))

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["coherence"].claimed == 0.0
    assert not by_quantity["coherence"].passed


def test_flags_the_real_part_6_transcript():
    if not CELLO_PATH.exists():
        pytest.skip("tests/AuSep_2_vc_01_Jupiter.wav not present")

    mono, sample_rate, _ = read_mono(CELLO_PATH)
    excerpt = mono[: 8 * sample_rate]
    stereo = render_amplitude_pan(excerpt, 25.0)
    evidence = measure_render(stereo, sample_rate)

    real_reply = (
        "The instrument is positioned at an angle of 25 degrees to the "
        "left because the inter-channel level difference is 19.5 dB with "
        "the left channel being louder and there is no coherence between "
        "the two channels."
    )
    result = verify(real_reply, evidence)

    by_quantity = {c.quantity: c for c in result.checks}
    assert by_quantity["coherence"].passed is False
    assert by_quantity["estimated_azimuth_deg_side"].passed is True
    assert by_quantity["estimated_azimuth_deg_magnitude"].passed is True
    assert by_quantity["ild_db"].passed is True


    assert result.all_passed is False


def test_unrecognized_numeric_sentence_is_logged_not_dropped():
    transcript = "The instrument is at 25 degrees to the left. It was recorded in 1987."
    result = verify(transcript, _cues())

    assert any("1987" in sentence for sentence in result.unparseable_claims)


def test_no_claims_at_all_is_not_reported_as_passing():
    result = verify("It sounds nice.", _cues())

    assert result.checks == ()
    assert result.all_passed is False
    assert result.inference_passed is False


def test_perturbation_tracked_when_claim_moves_with_the_render():
    transcript_a = "The instrument is at 10 degrees to the left."
    transcript_b = "The instrument is at 25 degrees to the left."
    result = verify_perturbation_pair(
        transcript_a, _cues(estimated_azimuth_deg=10.0),
        transcript_b, _cues(estimated_azimuth_deg=25.0),
    )

    assert result is not None
    assert result.tracked is True


def test_perturbation_not_tracked_when_claim_ignores_the_render():
    transcript_a = "The instrument is at 10 degrees to the left."
    transcript_b = "The instrument is at 10 degrees to the left."
    result = verify_perturbation_pair(
        transcript_a, _cues(estimated_azimuth_deg=10.0),
        transcript_b, _cues(estimated_azimuth_deg=25.0),
    )

    assert result is not None
    assert result.tracked is False


def test_perturbation_not_tracked_when_claim_moves_the_wrong_way():
    transcript_a = "The instrument is at 10 degrees to the left."
    transcript_b = "The instrument is at 10 degrees to the right."
    result = verify_perturbation_pair(
        transcript_a, _cues(estimated_azimuth_deg=10.0),
        transcript_b, _cues(estimated_azimuth_deg=25.0),
    )

    assert result is not None
    assert result.tracked is False


def test_perturbation_returns_none_without_extractable_claims():
    result = verify_perturbation_pair(
        "It sounds nice.", _cues(),
        "The instrument is at 25 degrees to the left.", _cues(),
    )
    assert result is None


def test_ablation_relied_on_numbers_when_answer_only_correct_with_them():
    with_numbers = "The level difference is 19.5 dB, left louder, so it's at 25 degrees left."
    without_numbers = "It's probably around 10 degrees left."
    result = verify_ablation_pair(
        with_numbers, _cues(estimated_azimuth_deg=25.0), without_numbers
    )

    assert result is not None
    assert result.with_numbers_passed is True
    assert result.without_numbers_passed is False
    assert result.relied_on_numbers is True


def test_ablation_not_relied_on_when_correct_either_way():
    with_numbers = "The level difference is 19.5 dB, so it's at 25 degrees left."
    without_numbers = "It's at 25 degrees left."
    result = verify_ablation_pair(
        with_numbers, _cues(estimated_azimuth_deg=25.0), without_numbers
    )

    assert result is not None
    assert result.with_numbers_passed is True
    assert result.without_numbers_passed is True
    assert result.relied_on_numbers is False


def test_ablation_returns_none_without_a_with_numbers_claim():
    result = verify_ablation_pair("It sounds nice.", _cues(), "Also nice.")
    assert result is None
