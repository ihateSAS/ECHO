from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cues.delay import DelayEstimate
from cues.level import LevelEstimate

RegimeLabel = Literal["amplitude", "delayed", "ambiguous"]


@dataclass(frozen=True)
class RegimeEstimate:
    label: RegimeLabel
    confidence: float


def classify_regime(
    level: LevelEstimate,
    delay: DelayEstimate,
    delay_threshold_us: float = 30.0,
    flatness_threshold_db: float = 2.0,
) -> RegimeEstimate:
    delayed = abs(delay.itd_us) > delay_threshold_us
    frequency_dependent = level.ild_flatness_db > flatness_threshold_db
    if not delayed and not frequency_dependent:
        return RegimeEstimate("amplitude", 1.0)
    if delayed:
        return RegimeEstimate("delayed", 1.0 if frequency_dependent else 0.75)
    return RegimeEstimate("ambiguous", 0.5)
