from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.mixture_render import MIN_ANGLE_SEPARATION_DEG
from benchmark.questions import side_for_angle
from benchmark.schemas import MixtureCueMeasurements
from verifier.extract import extract_all_azimuths

AZIMUTH_TOLERANCE_DEG = 5.0


@dataclass(frozen=True)
class MixtureLabelledItem:
    item_id: str
    true_azimuths_deg: tuple[float, ...]
    evidence: MixtureCueMeasurements | None

    @classmethod
    def from_record(cls, record: dict) -> MixtureLabelledItem:
        raw = record.get("measured_evidence")
        evidence = None
        if raw is not None:
            evidence = MixtureCueMeasurements(
                source_azimuths_deg=tuple(raw["source_azimuths_deg"]),
                source_energy_fractions=tuple(raw["source_energy_fractions"]),
                active_bins=raw["active_bins"],
            )
        return cls(
            item_id=record["item_id"],
            true_azimuths_deg=tuple(record["answer_azimuths_deg"]),
            evidence=evidence,
        )


def signed_azimuths(transcript: str) -> tuple[float, ...]:
    return tuple(
        magnitude if side == "left" else -magnitude
        for magnitude, side, _ in extract_all_azimuths(transcript)
    )


def _match(
    true_azimuths: tuple[float, ...], predicted_azimuths: tuple[float, ...]
) -> int:
    remaining = list(predicted_azimuths)
    matched = 0
    for true_angle in true_azimuths:
        for index, predicted_angle in enumerate(remaining):
            if abs(predicted_angle - true_angle) <= AZIMUTH_TOLERANCE_DEG:
                matched += 1
                del remaining[index]
                break
    return matched


@dataclass(frozen=True)
class MixtureItemScore:
    item_id: str
    true_azimuths_deg: tuple[float, ...]
    predicted_azimuths_deg: tuple[float, ...]
    matched_count: int

    @property
    def answered(self) -> bool:
        return len(self.predicted_azimuths_deg) > 0

    @property
    def true_count(self) -> int:
        return len(self.true_azimuths_deg)

    @property
    def predicted_count(self) -> int:
        return len(self.predicted_azimuths_deg)

    @property
    def count_correct(self) -> bool:
        return self.true_count == self.predicted_count

    @property
    def recall(self) -> float:
        return self.matched_count / self.true_count if self.true_count else float("nan")

    @property
    def precision(self) -> float | None:
        if not self.answered:
            return None
        return self.matched_count / self.predicted_count

    @property
    def exact_match(self) -> bool:
        return self.count_correct and self.matched_count == self.true_count


def score_mixture_item(item: MixtureLabelledItem, transcript: str) -> MixtureItemScore:
    predicted = signed_azimuths(transcript)
    matched = _match(item.true_azimuths_deg, predicted)
    return MixtureItemScore(
        item_id=item.item_id,
        true_azimuths_deg=item.true_azimuths_deg,
        predicted_azimuths_deg=predicted,
        matched_count=matched,
    )


@dataclass(frozen=True)
class MixtureScoreReport:
    name: str
    scores: list[MixtureItemScore] = field(default_factory=list)

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
    def exact_match_rate(self) -> float:
        if not self.items:
            return float("nan")
        return sum(1 for score in self.scores if score.exact_match) / self.items

    @property
    def count_correct_rate(self) -> float:
        if not self.items:
            return float("nan")
        return sum(1 for score in self.scores if score.count_correct) / self.items

    @property
    def mean_recall(self) -> float:
        if not self.items:
            return float("nan")
        return sum(score.recall for score in self.scores) / self.items

    @property
    def mean_precision(self) -> float:
        precisions = [
            score.precision for score in self.scores if score.precision is not None
        ]
        return sum(precisions) / len(precisions) if precisions else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "items": self.items,
            "answered": self.answered,
            "coverage": self.coverage,
            "exact_match_rate": self.exact_match_rate,
            "count_correct_rate": self.count_correct_rate,
            "mean_recall": self.mean_recall,
            "mean_precision": self.mean_precision,
            "azimuth_tolerance_deg": AZIMUTH_TOLERANCE_DEG,
        }

    def summary(self) -> str:
        precision = (
            "n/a" if math.isnan(self.mean_precision) else f"{self.mean_precision:6.1%}"
        )
        return (
            f"{self.name:<24} "
            f"exact match {self.exact_match_rate:6.1%}  "
            f"count correct {self.count_correct_rate:6.1%}  "
            f"recall {self.mean_recall:6.1%}  "
            f"precision {precision} over {self.coverage:5.1%} answered"
        )


def score_mixture_answers(
    labels: list[MixtureLabelledItem], answers: dict[str, str], name: str = "model"
) -> MixtureScoreReport:
    return MixtureScoreReport(
        name=name,
        scores=[
            score_mixture_item(item, answers.get(item.item_id, "")) for item in labels
        ],
    )


def _as_mixture_prose(azimuths_deg: tuple[float, ...]) -> str:
    if not azimuths_deg:
        return "There is only one instrument, centred."
    parts = []
    for angle in azimuths_deg:
        side = side_for_angle(angle)
        if side == "center":
            parts.append("one source centred, 0 degrees to the left")
        else:
            parts.append(f"one source at {abs(angle):.1f} degrees to the {side}")
    return f"I hear {len(azimuths_deg)} sources: " + "; ".join(parts) + "."


def refusal_baseline(labels: list[MixtureLabelledItem]) -> MixtureScoreReport:
    return score_mixture_answers(labels, {}, name="no answer")


def constant_count_baseline(
    labels: list[MixtureLabelledItem], count: int = 2
) -> MixtureScoreReport:
    azimuths = tuple(
        MIN_ANGLE_SEPARATION_DEG * ((count - 1) / 2.0 - index) for index in range(count)
    )
    prose = _as_mixture_prose(azimuths)
    return score_mixture_answers(
        labels, {item.item_id: prose for item in labels}, name=f"always {count} sources"
    )


def front_end_baseline(labels: list[MixtureLabelledItem]) -> MixtureScoreReport:
    answers: dict[str, str] = {}
    for item in labels:
        if item.evidence is None:
            continue
        azimuths = tuple(
            sorted(item.evidence.source_azimuths_deg, reverse=True)
        )
        answers[item.item_id] = _as_mixture_prose(azimuths)
    return score_mixture_answers(labels, answers, name="front end (ceiling)")


def read_mixture_labels(path: Path) -> list[MixtureLabelledItem]:
    items: list[MixtureLabelledItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "answer_azimuths_deg" not in record:
            question = record["question"]
            record = {
                "item_id": record["item_id"],
                "answer_azimuths_deg": question["answer_azimuths_deg"],
                "measured_evidence": record.get("measured_evidence"),
            }
        items.append(MixtureLabelledItem.from_record(record))
    return items
