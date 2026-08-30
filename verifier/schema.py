from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConsistencyKind = Literal["read_out", "inference"]


@dataclass(frozen=True)
class ClaimCheck:
    quantity: str
    claimed: float | str
    measured: float | str
    tolerance: float | None
    passed: bool
    consistency_kind: ConsistencyKind


@dataclass(frozen=True)
class VerificationResult:
    checks: tuple[ClaimCheck, ...]
    all_passed: bool
    inference_passed: bool


    unparseable_claims: tuple[str, ...]


@dataclass(frozen=True)
class PerturbationResult:
    quantity: str
    claimed_a: float
    claimed_b: float
    measured_a: float
    measured_b: float
    measured_delta: float
    claimed_delta: float
    tracked: bool


@dataclass(frozen=True)
class AblationResult:
    quantity: str
    claimed_with_numbers: float
    claimed_without_numbers: float | None

    measured: float
    tolerance: float
    with_numbers_passed: bool
    without_numbers_passed: bool


    relied_on_numbers: bool
