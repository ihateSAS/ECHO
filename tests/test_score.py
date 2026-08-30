from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.schemas import CueMeasurements  # noqa: E402
from eval.score import (  # noqa: E402
    AZIMUTH_TOLERANCE_DEG,
    LabelledItem,
    constant_baseline,
    front_end_baseline,
    random_baseline,
    read_answers,
    read_labels,
    refusal_baseline,
    score_answers,
    score_item,
    signed_azimuth,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANGLES = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def _evidence(azimuth_deg: float) -> CueMeasurements:
    from cues.level import ild_db_for_azimuth

    ild = ild_db_for_azimuth(azimuth_deg)
    return CueMeasurements(
        ild_db=ild,
        high_frequency_ild_db=ild,
        ild_flatness_db=0.0,
        itd_us=0.0,
        itd_peak_strength=1.0,
        coherence=1.0,
        estimated_azimuth_deg=azimuth_deg,
        estimated_regime="amplitude",
        active_bins=500,
        f0_hz=float("nan"),
    )


def _labels(angles: tuple[float, ...] = ANGLES, evidence: bool = True):
    from benchmark.questions import side_for_angle

    return [
        LabelledItem(
            item_id=f"item_{index:03d}",
            azimuth_deg=angle,
            side=side_for_angle(angle),
            instrument="cello",
            evidence=_evidence(angle) if evidence else None,
        )
        for index, angle in enumerate(angles)
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The cello is at 25 degrees to the left.", 25.0),
        ("The cello is at 12.5 degrees to the right.", -12.5),
        ("It sits 0 degrees to the left, dead centre.", 0.0),
        ("...towards the left, as the 18.0-degree azimuth shows.", 18.0),
        ("The trumpet is at 0.0 degrees to the center.", 0.0),
        ("The flute is centred.", 0.0),
        ("azimuth = 0.0 degrees. The cello is in the centre.", 0.0),
        ("left at 27.5°", 27.5),
        ("right, 23°", -23.0),
        ("The trumpet is on the right side, at an azimuth of 12.9°.", -12.9),
    ],
)
def test_a_signed_answer_is_read_out_of_prose(text: str, expected: float):
    assert signed_azimuth(text) == pytest.approx(expected)


def test_a_verbose_derivation_is_read_by_its_final_claim_not_its_first():
    text = (
        "We have the formula for the panning coordinate:\n"
        "d = (r - 1) / (r + 1)\n"
        "where \nr = 10^(level difference in dB / 20).\n\n"
        "We also have the formula for the angle:\n"
        "theta = arctan(d * tan 30 degrees).\n\n"
        "Let's go step-by-step.\n\n"
        "1. Compute r:\nr = 10^(12.9 / 20)\n   = 10^(0.645)\n   ~= 4.42\n\n"
        "2. Compute the panning coordinate d:\n"
        "d = (4.42 - 1) / (4.42 + 1)\n   = 3.42 / 5.42\n   ~= 0.631\n\n"
        "3. Plug d into the angle formula:\n"
        "theta = arctan(0.631 * tan 30 degrees)\n\n"
        "We know tan 30 degrees ~= 0.577.\nSo:\n"
        "theta = arctan(0.631 * 0.577)\n      = arctan(0.364)\n\n"
        "Find the arctan:\ntheta ~= 20.06 degrees.\n\n"
        "This is the angle we get from the level difference.\n\n"
        "Positive azimuth means left, and since the left channel is louder, "
        "the source is positioned toward the left.\n\n"
        "Answer: The trumpet is positioned to the left in the stereo image, "
        "at approximately +20 degrees azimuth."
    )
    assert signed_azimuth(text) == pytest.approx(20.0)


@pytest.mark.parametrize(
    "text", ["It is somewhere in the middle.", "", "I cannot tell from this audio."]
)
def test_prose_with_no_angle_in_it_is_not_an_answer(text: str):
    assert signed_azimuth(text) is None


def test_the_scorer_and_the_verifier_read_a_transcript_the_same_way():
    from verifier.extract import extract_azimuth

    text = "The cello is at 20 degrees to the right."
    found = extract_azimuth(text)
    assert found is not None
    magnitude, side, _ = found
    assert signed_azimuth(text) == pytest.approx(
        magnitude if side == "left" else -magnitude
    )


def test_a_correct_answer_scores_correct():
    (item,) = _labels((15.0,))
    score = score_item(item, "The cello is at 15 degrees to the left.")

    assert score.answered
    assert score.abs_error_deg == pytest.approx(0.0)
    assert score.within_tolerance
    assert score.side_correct


def test_the_right_magnitude_on_the_wrong_side_is_not_nearly_right():
    (item,) = _labels((15.0,))
    score = score_item(item, "The cello is at 15 degrees to the right.")

    assert score.abs_error_deg == pytest.approx(30.0)
    assert not score.within_tolerance
    assert not score.side_correct


def test_an_answer_just_inside_the_tolerance_counts_and_just_outside_does_not():
    (item,) = _labels((10.0,))
    inside = score_item(item, "The cello is at 14.9 degrees to the left.")
    outside = score_item(item, "The cello is at 15.1 degrees to the left.")

    assert inside.within_tolerance
    assert not outside.within_tolerance


def test_a_centre_label_accepts_zero_degrees_on_either_side():
    (item,) = _labels((0.0,))
    for text in ("0 degrees to the left", "0 degrees to the right"):
        score = score_item(item, f"The cello is at {text}.")
        assert score.side_correct
        assert score.within_tolerance


def test_an_unanswered_item_is_wrong_not_missing():
    (item,) = _labels((15.0,))
    score = score_item(item, "I am not able to determine that.")

    assert not score.answered
    assert score.abs_error_deg is None
    assert not score.within_tolerance
    assert not score.side_correct


def test_an_item_without_evidence_is_scored_but_not_verified():
    (item,) = _labels((15.0,), evidence=False)
    score = score_item(item, "The cello is at 15 degrees to the left.")

    assert score.within_tolerance
    assert score.all_passed is None
    assert score.inference_passed is None


def test_refusing_everything_scores_zero_not_undefined():
    report = refusal_baseline(_labels())

    assert report.within_tolerance_rate == 0.0
    assert report.side_accuracy == 0.0
    assert report.coverage == 0.0
    assert math.isnan(report.mean_abs_error_deg)


def test_answering_only_the_easy_items_does_not_buy_a_good_score():
    labels = _labels()
    answers = {
        item.item_id: f"The cello is at {abs(item.azimuth_deg):.1f} degrees to the "
        f"{'left' if item.azimuth_deg > 0 else 'right'}."
        for item in labels
        if abs(item.azimuth_deg) <= 5.0
    }
    report = score_answers(labels, answers, name="selective")

    assert report.mean_abs_error_deg == pytest.approx(0.0)
    assert report.coverage == pytest.approx(3 / 11)
    assert report.within_tolerance_rate == pytest.approx(3 / 11)


def test_always_centre_scores_exactly_what_the_grid_implies():
    report = constant_baseline(_labels(), 0.0)

    assert report.within_tolerance_rate == pytest.approx(3 / 11)
    assert report.mean_abs_error_deg == pytest.approx(150 / 11)
    assert report.coverage == 1.0
    assert report.side_accuracy == pytest.approx(1 / 11)


def test_the_front_end_baseline_is_the_ceiling():
    report = front_end_baseline(_labels())

    assert report.within_tolerance_rate == 1.0
    assert report.side_accuracy == 1.0
    assert report.mean_abs_error_deg == pytest.approx(0.0, abs=1e-9)
    assert report.inference_passed_rate == 1.0


def test_the_random_baseline_is_reproducible_and_beatable():
    labels = _labels()
    first = random_baseline(labels, ANGLES, seed=7)
    second = random_baseline(labels, ANGLES, seed=7)

    assert first.to_dict() == second.to_dict()
    assert first.within_tolerance_rate < 0.5


def test_the_by_angle_breakdown_exposes_a_centre_prior():
    breakdown = constant_baseline(_labels(), 0.0).by_angle()

    assert breakdown[0.0] == 1.0
    assert breakdown[5.0] == 1.0
    assert breakdown[25.0] == 0.0
    assert breakdown[-25.0] == 0.0


def test_faithfulness_is_reported_beside_accuracy_not_inside_it():
    labels = _labels((25.0,))
    right_and_checkable = score_answers(
        labels,
        {
            labels[0]
            .item_id: "The level difference is 19.5 dB with the left channel "
            "louder, the regime is amplitude, so the cello is at 25 degrees "
            "to the left."
        },
        name="checkable",
    )
    assert right_and_checkable.within_tolerance_rate == 1.0
    assert right_and_checkable.inference_passed_rate == 1.0

    bare = score_answers(
        labels, {labels[0].item_id: "25 degrees to the left."}, name="bare"
    )
    assert bare.within_tolerance_rate == 1.0
    assert 0.0 <= bare.inference_passed_rate <= 1.0


def test_private_test_labels_and_train_items_both_load(tmp_path: Path):
    private = tmp_path / "test_private_labels.jsonl"
    private.write_text(
        json.dumps(
            {
                "item_id": "smq_test_00001_absolute",
                "answer_side": "left",
                "answer_azimuth_deg": 15.0,
                "render": {"instrument": "cello"},
                "measured_evidence": {
                    key: getattr(_evidence(15.0), key)
                    for key in CueMeasurements.__dataclass_fields__
                },
            }
        )
        + "\n"
    )
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "item_id": "stem_amp_p15_0ms_absolute",
                "question": {"answer_side": "left", "answer_azimuth_deg": 15.0},
                "render": {"instrument": "cello"},
                "measured_evidence": None,
            }
        )
        + "\n"
    )

    (from_private,) = read_labels(private)
    (from_train,) = read_labels(train)
    assert from_private.azimuth_deg == from_train.azimuth_deg == 15.0
    assert from_private.evidence is not None
    assert from_train.evidence is None


def test_answers_accept_either_field_name(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"item_id": "a", "answer": "10 degrees to the left"})
        + "\n"
        + json.dumps({"item_id": "b", "transcript": "5 degrees to the right"})
        + "\n"
    )
    answers = read_answers(path)
    assert set(answers) == {"a", "b"}


def test_an_answer_file_missing_the_text_says_so(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"item_id": "a", "azimuth": 10.0}) + "\n")
    with pytest.raises(ValueError, match="needs an 'answer'"):
        read_answers(path)


def _write_labels(path: Path, angles: tuple[float, ...]) -> None:
    from benchmark.questions import side_for_angle

    with path.open("w", encoding="utf-8") as file:
        for index, angle in enumerate(angles):
            json.dump(
                {
                    "item_id": f"smq_test_{index:05d}_absolute",
                    "answer_side": side_for_angle(angle),
                    "answer_azimuth_deg": angle,
                    "render": {"instrument": "cello"},
                    "measured_evidence": {
                        key: getattr(_evidence(angle), key)
                        for key in CueMeasurements.__dataclass_fields__
                    },
                },
                file,
            )
            file.write("\n")


def test_the_scorer_command_line_runs_the_baselines_and_writes_json(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, ANGLES)
    output = tmp_path / "out" / "score.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "score_benchmark.py"),
            "--labels",
            str(labels),
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "always centre" in completed.stdout
    assert "front end (ceiling)" in completed.stdout

    payload = json.loads(output.read_text())
    by_name = {report["name"]: report for report in payload["reports"]}
    assert by_name["always centre"]["within_tolerance_rate"] == pytest.approx(3 / 11)
    assert by_name["front end (ceiling)"]["within_tolerance_rate"] == 1.0
    assert payload["items"] == 11


def test_the_scorer_command_line_grades_an_answers_file(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, ANGLES)
    answers = tmp_path / "answers.jsonl"
    with answers.open("w", encoding="utf-8") as file:
        for index, angle in enumerate(ANGLES):
            side = "left" if angle > 0 else "right"
            json.dump(
                {
                    "item_id": f"smq_test_{index:05d}_absolute",
                    "answer": f"The cello is at {abs(angle):.1f} degrees to the {side}.",
                },
                file,
            )
            file.write("\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "score_benchmark.py"),
            "--labels",
            str(labels),
            "--answers",
            str(answers),
            "--name",
            "perfect",
            "--baselines",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "perfect" in completed.stdout
    assert "100.0%" in completed.stdout


def test_the_scorer_says_what_to_do_when_the_benchmark_is_not_built(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "score_benchmark.py"),
            "--labels",
            str(tmp_path / "absent.jsonl"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode != 0
    assert "build_benchmark.py" in completed.stderr


def test_the_tolerance_the_scorer_uses_is_the_papers():
    assert AZIMUTH_TOLERANCE_DEG == 5.0
