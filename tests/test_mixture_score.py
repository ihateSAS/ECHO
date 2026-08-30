from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.mixture_render import MIN_ANGLE_SEPARATION_DEG  # noqa: E402
from benchmark.schemas import MixtureCueMeasurements  # noqa: E402
from eval.mixture_score import (  # noqa: E402
    MixtureLabelledItem,
    constant_count_baseline,
    front_end_baseline,
    read_mixture_labels,
    refusal_baseline,
    score_mixture_answers,
    score_mixture_item,
    signed_azimuths,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _evidence(azimuths_deg: tuple[float, ...]) -> MixtureCueMeasurements:
    return MixtureCueMeasurements(
        source_azimuths_deg=azimuths_deg,
        source_energy_fractions=tuple(1.0 / len(azimuths_deg) for _ in azimuths_deg),
        active_bins=1000,
    )


def _item(item_id: str, azimuths_deg: tuple[float, ...], evidence: bool = True):
    return MixtureLabelledItem(
        item_id=item_id,
        true_azimuths_deg=azimuths_deg,
        evidence=_evidence(azimuths_deg) if evidence else None,
    )


def test_extracts_every_azimuth_in_order():
    text = "I hear one source at 15.0 degrees to the left and another at 20.0 degrees to the right."
    assert signed_azimuths(text) == (15.0, -20.0)


def test_extracts_three_claims_including_the_loose_pattern():
    text = (
        "One source is 10.0 degrees to the left. Leaning strongly right, "
        "the second source sits around the 15.0-degree mark. The third is "
        "dead centre, 0 degrees to the left."
    )
    result = signed_azimuths(text)
    assert result == (10.0, -15.0, 0.0)


def test_no_claims_found_is_an_empty_tuple_not_a_crash():
    assert signed_azimuths("I have no idea where anything is.") == ()


def test_exact_match_when_every_position_is_found():
    item = _item("a", (15.0, -20.0))
    score = score_mixture_item(item, "one at 15.0 degrees to the left, one at 20.0 degrees to the right")
    assert score.true_count == 2
    assert score.predicted_count == 2
    assert score.matched_count == 2
    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.exact_match is True


def test_missing_a_source_costs_recall_not_precision():
    item = _item("a", (15.0, -20.0))
    score = score_mixture_item(item, "one source at 15.0 degrees to the left")
    assert score.matched_count == 1
    assert score.recall == pytest.approx(0.5)
    assert score.precision == 1.0
    assert score.exact_match is False
    assert score.count_correct is False


def test_hallucinated_extra_source_costs_precision_not_recall():
    item = _item("a", (15.0,))
    score = score_mixture_item(
        item, "one at 15.0 degrees to the left, another at 10.0 degrees to the right"
    )
    assert score.matched_count == 1
    assert score.recall == 1.0
    assert score.precision == pytest.approx(0.5)
    assert score.exact_match is False


def test_unanswered_item_scores_zero_not_nan():
    item = _item("a", (15.0, -20.0))
    score = score_mixture_item(item, "")
    assert score.answered is False
    assert score.recall == 0.0
    assert score.precision is None
    assert score.exact_match is False


def test_a_position_just_outside_tolerance_does_not_match():
    item = _item("a", (15.0,))
    score = score_mixture_item(item, "one source at 20.1 degrees to the left")
    assert score.matched_count == 0
    assert score.recall == 0.0


def test_a_position_just_inside_tolerance_matches():
    item = _item("a", (15.0,))
    score = score_mixture_item(item, "one source at 19.9 degrees to the left")
    assert score.matched_count == 1
    assert score.recall == 1.0


def test_refusal_baseline_scores_exactly_zero():
    labels = [_item(f"item_{i}", (15.0, -15.0)) for i in range(5)]
    report = refusal_baseline(labels)
    assert report.coverage == 0.0
    assert report.exact_match_rate == 0.0
    assert report.count_correct_rate == 0.0
    assert report.mean_recall == 0.0
    import math

    assert math.isnan(report.mean_precision)


def test_front_end_baseline_is_the_ceiling():
    labels = [
        _item("a", (15.0, -15.0)),
        _item("b", (20.0, 0.0, -10.0)),
    ]
    report = front_end_baseline(labels)
    assert report.exact_match_rate == 1.0
    assert report.count_correct_rate == 1.0
    assert report.mean_recall == 1.0
    assert report.mean_precision == 1.0


def test_front_end_baseline_skips_items_with_no_evidence():
    labels = [_item("a", (15.0,), evidence=False)]
    report = front_end_baseline(labels)
    assert report.coverage == 0.0


def test_constant_count_baseline_uses_the_real_separation_constant():
    labels = [_item("a", (15.0, -15.0))]
    report = constant_count_baseline(labels, count=2)


    only_score = report.scores[0]
    assert only_score.predicted_count == 2
    gap = abs(only_score.predicted_azimuths_deg[0] - only_score.predicted_azimuths_deg[1])
    assert gap == pytest.approx(MIN_ANGLE_SEPARATION_DEG)


def test_mean_recall_averages_across_items_not_pools_positions():
    labels = [_item("a", (15.0, -15.0)), _item("b", (10.0, 0.0, -10.0))]
    answers = {
        "a": "one at 15.0 degrees to the left, one at 15.0 degrees to the right",
        "b": "nothing here",
    }
    report = score_mixture_answers(labels, answers)
    assert report.mean_recall == pytest.approx(0.5)


def test_reads_and_scores_the_real_built_mixture_labels():
    path = REPO_ROOT / "artifacts" / "stereomusicqa_v0.2" / "mixture_test_private_labels.jsonl"
    if not path.is_file():
        pytest.skip(f"{path} not built")
    labels = read_mixture_labels(path)
    assert labels
    for item in labels:
        assert len(item.true_azimuths_deg) >= 2

        assert list(item.true_azimuths_deg) == sorted(item.true_azimuths_deg, reverse=True)

    report = front_end_baseline(labels)


    assert report.exact_match_rate == 1.0
