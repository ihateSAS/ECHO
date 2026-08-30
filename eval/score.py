from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.questions import side_for_angle
from benchmark.schemas import CueMeasurements
from verifier import verify
from verifier.extract import extract_azimuth

AZIMUTH_TOLERANCE_DEG = 5.0


@dataclass(frozen=True)
class LabelledItem:
    item_id: str
    azimuth_deg: float
    side: str
    instrument: str
    evidence: CueMeasurements | None

    @classmethod
    def from_record(cls, record: dict) -> LabelledItem:
        raw = record.get("measured_evidence")
        evidence = None
        if raw is not None:
            evidence = CueMeasurements(
                **{
                    key: raw[key]
                    for key in CueMeasurements.__dataclass_fields__
                    if key in raw
                }
            )
        return cls(
            item_id=record["item_id"],
            azimuth_deg=float(record["answer_azimuth_deg"]),
            side=record["answer_side"],
            instrument=record["render"]["instrument"],
            evidence=evidence,
        )


@dataclass(frozen=True)
class ItemScore:
    item_id: str
    true_azimuth_deg: float
    predicted_azimuth_deg: float | None
    abs_error_deg: float | None
    within_tolerance: bool
    side_correct: bool
    all_passed: bool | None
    inference_passed: bool | None

    @property
    def answered(self) -> bool:
        return self.predicted_azimuth_deg is not None


def signed_azimuth(transcript: str) -> float | None:
    found = extract_azimuth(transcript)
    if found is None:
        return None
    magnitude, side, _ = found
    if side == "center":
        return 0.0
    return magnitude if side == "left" else -magnitude


def score_item(item: LabelledItem, transcript: str) -> ItemScore:
    predicted = signed_azimuth(transcript)
    error = None if predicted is None else abs(predicted - item.azimuth_deg)

    verification = None
    if item.evidence is not None:
        verification = verify(transcript, item.evidence)

    return ItemScore(
        item_id=item.item_id,
        true_azimuth_deg=item.azimuth_deg,
        predicted_azimuth_deg=predicted,
        abs_error_deg=error,
        within_tolerance=error is not None and error <= AZIMUTH_TOLERANCE_DEG,
        side_correct=(
            predicted is not None and side_for_angle(predicted) == item.side
        ),
        all_passed=None if verification is None else verification.all_passed,
        inference_passed=(
            None if verification is None else verification.inference_passed
        ),
    )


@dataclass(frozen=True)
class ScoreReport:
    name: str
    scores: list[ItemScore] = field(default_factory=list)

    @property
    def items(self) -> int:
        return len(self.scores)

    @property
    def answered(self) -> int:
        return sum(1 for score in self.scores if score.answered)

    @property
    def coverage(self) -> float:
        return self.answered / self.items if self.items else float("nan")

    @property
    def mean_abs_error_deg(self) -> float:
        errors = [
            score.abs_error_deg
            for score in self.scores
            if score.abs_error_deg is not None
        ]
        return sum(errors) / len(errors) if errors else float("nan")

    @property
    def median_abs_error_deg(self) -> float:
        errors = sorted(
            score.abs_error_deg
            for score in self.scores
            if score.abs_error_deg is not None
        )
        if not errors:
            return float("nan")
        middle = len(errors) // 2
        if len(errors) % 2:
            return errors[middle]
        return 0.5 * (errors[middle - 1] + errors[middle])

    @property
    def within_tolerance_rate(self) -> float:
        if not self.items:
            return float("nan")
        return sum(1 for score in self.scores if score.within_tolerance) / self.items

    @property
    def side_accuracy(self) -> float:
        if not self.items:
            return float("nan")
        return sum(1 for score in self.scores if score.side_correct) / self.items

    def _verified_rate(self, attribute: str) -> float:
        checked = [
            getattr(score, attribute)
            for score in self.scores
            if getattr(score, attribute) is not None
        ]
        return sum(checked) / len(checked) if checked else float("nan")

    @property
    def all_passed_rate(self) -> float:
        return self._verified_rate("all_passed")

    @property
    def inference_passed_rate(self) -> float:
        return self._verified_rate("inference_passed")

    def by_angle(self) -> dict[float, float]:
        buckets: dict[float, list[ItemScore]] = {}
        for score in self.scores:
            buckets.setdefault(score.true_azimuth_deg, []).append(score)
        return {
            angle: sum(1 for score in group if score.within_tolerance) / len(group)
            for angle, group in sorted(buckets.items())
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "items": self.items,
            "answered": self.answered,
            "coverage": self.coverage,
            "mean_abs_error_deg": self.mean_abs_error_deg,
            "median_abs_error_deg": self.median_abs_error_deg,
            "within_tolerance_rate": self.within_tolerance_rate,
            "azimuth_tolerance_deg": AZIMUTH_TOLERANCE_DEG,
            "side_accuracy": self.side_accuracy,
            "all_passed_rate": self.all_passed_rate,
            "inference_passed_rate": self.inference_passed_rate,
            "within_tolerance_by_angle": {
                str(angle): rate for angle, rate in self.by_angle().items()
            },
        }

    def summary(self) -> str:
        faithful = (
            "n/a"
            if math.isnan(self.inference_passed_rate)
            else f"{self.inference_passed_rate:6.1%}"
        )
        return (
            f"{self.name:<24} "
            f"within {AZIMUTH_TOLERANCE_DEG:g} deg {self.within_tolerance_rate:6.1%}  "
            f"side {self.side_accuracy:6.1%}  "
            f"MAE {self.mean_abs_error_deg:6.2f} deg over {self.coverage:5.1%} answered  "
            f"inference {faithful}"
        )


def score_answers(
    labels: list[LabelledItem], answers: dict[str, str], name: str = "model"
) -> ScoreReport:
    return ScoreReport(
        name=name,
        scores=[score_item(item, answers.get(item.item_id, "")) for item in labels],
    )


def _as_prose(azimuth_deg: float, instrument: str) -> str:
    side = side_for_angle(azimuth_deg)
    if side == "center":
        return f"The {instrument} is centred, 0 degrees to the left."
    return f"The {instrument} is at {abs(azimuth_deg):.1f} degrees to the {side}."


def constant_baseline(
    labels: list[LabelledItem], azimuth_deg: float = 0.0
) -> ScoreReport:
    name = "always centre" if azimuth_deg == 0.0 else f"always {azimuth_deg:+g} deg"
    return score_answers(
        labels,
        {item.item_id: _as_prose(azimuth_deg, item.instrument) for item in labels},
        name=name,
    )


def random_baseline(
    labels: list[LabelledItem],
    angles_deg: tuple[float, ...],
    seed: int = 0,
) -> ScoreReport:
    rng = random.Random(seed)
    return score_answers(
        labels,
        {
            item.item_id: _as_prose(rng.choice(angles_deg), item.instrument)
            for item in labels
        },
        name="random from grid",
    )


def front_end_baseline(labels: list[LabelledItem]) -> ScoreReport:
    answers: dict[str, str] = {}
    for item in labels:
        if item.evidence is None:
            continue
        answers[item.item_id] = _as_prose(
            item.evidence.estimated_azimuth_deg, item.instrument
        )
    return score_answers(labels, answers, name="front end (ceiling)")


def refusal_baseline(labels: list[LabelledItem]) -> ScoreReport:
    return score_answers(labels, {}, name="no answer")


def read_labels(path: Path) -> list[LabelledItem]:
    items: list[LabelledItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "answer_azimuth_deg" not in record:
            question = record["question"]
            record = {
                "item_id": record["item_id"],
                "answer_azimuth_deg": question["answer_azimuth_deg"],
                "answer_side": question["answer_side"],
                "render": record["render"],
                "measured_evidence": record.get("measured_evidence"),
            }
        items.append(LabelledItem.from_record(record))
    return items


def read_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = record.get("answer", record.get("transcript"))
        if text is None:
            raise ValueError(
                f"{path}: each line needs an 'answer' (or 'transcript') field"
            )
        answers[record["item_id"]] = str(text)
    return answers


def read_build_id(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        found = json.loads(line).get("build_id")
        if found is not None:
            return str(found)


        return None
    return None
