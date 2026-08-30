from __future__ import annotations

import math

from benchmark.schemas import CueMeasurements
from verifier.extract import (
    extract_azimuth,
    extract_coherence,
    extract_f0_hz,
    extract_ild_db,
    extract_itd_us,
    extract_premise_r,
    extract_regime,
    find_unparseable_sentences,
)
from verifier.schema import (
    AblationResult,
    ClaimCheck,
    PerturbationResult,
    VerificationResult,
)

AZIMUTH_TOLERANCE_DEG = 5.0
ILD_TOLERANCE_DB = 1.5
ITD_TOLERANCE_US = 30.0
COHERENCE_TOLERANCE = 0.10
F0_TOLERANCE_FRACTION = 0.03


PREMISE_R_TOLERANCE_DB = 0.5


def _side_for(value: float) -> str:
    if value > 0.0:
        return "left"
    if value < 0.0:
        return "right"
    return "center"


def verify(transcript: str, evidence: CueMeasurements) -> VerificationResult:
    checks: list[ClaimCheck] = []
    spans: list[tuple[int, int]] = []

    azimuth_claim = extract_azimuth(transcript)
    if azimuth_claim is not None:
        magnitude, side, span = azimuth_claim
        spans.append(span)
        measured = evidence.estimated_azimuth_deg
        measured_side = _side_for(measured)


        side_ok = measured_side == "center" or side == measured_side
        checks.append(
            ClaimCheck(
                "estimated_azimuth_deg_side", side, measured_side, None, side_ok, "inference"
            )
        )
        magnitude_ok = abs(magnitude - abs(measured)) <= AZIMUTH_TOLERANCE_DEG
        checks.append(
            ClaimCheck(
                "estimated_azimuth_deg_magnitude",
                magnitude,
                abs(measured),
                AZIMUTH_TOLERANCE_DEG,
                magnitude_ok,
                "inference",
            )
        )

    ild_claim = extract_ild_db(transcript)
    if ild_claim is not None:
        value, span = ild_claim
        spans.append(span)
        passed = abs(value - evidence.ild_db) <= ILD_TOLERANCE_DB
        checks.append(
            ClaimCheck("ild_db", value, evidence.ild_db, ILD_TOLERANCE_DB, passed, "read_out")
        )

    itd_claim = extract_itd_us(transcript)
    if itd_claim is not None:
        value, span = itd_claim
        spans.append(span)
        passed = abs(value - evidence.itd_us) <= ITD_TOLERANCE_US
        checks.append(
            ClaimCheck("itd_us", value, evidence.itd_us, ITD_TOLERANCE_US, passed, "read_out")
        )

    coherence_claim = extract_coherence(transcript)
    if coherence_claim is not None:
        value, span = coherence_claim
        spans.append(span)
        passed = abs(value - evidence.coherence) <= COHERENCE_TOLERANCE
        checks.append(
            ClaimCheck(
                "coherence", value, evidence.coherence, COHERENCE_TOLERANCE, passed, "read_out"
            )
        )

    regime_claim = extract_regime(transcript)
    if regime_claim is not None:
        value, span = regime_claim
        spans.append(span)
        passed = value == evidence.estimated_regime
        checks.append(
            ClaimCheck("estimated_regime", value, evidence.estimated_regime, None, passed, "inference")
        )

    f0_claim = extract_f0_hz(transcript)
    if f0_claim is not None and evidence.f0_hz == evidence.f0_hz:
        value, span = f0_claim
        spans.append(span)
        tolerance = F0_TOLERANCE_FRACTION * evidence.f0_hz
        passed = abs(value - evidence.f0_hz) <= tolerance
        checks.append(
            ClaimCheck("f0_hz", value, evidence.f0_hz, tolerance, passed, "read_out")
        )


    premise_claim = extract_premise_r(transcript)
    if premise_claim is not None:
        cue_in_formula, r_claimed, span = premise_claim
        spans.append(span)
        if r_claimed > 0.0:
            implied_ild_db = 20.0 * math.log10(r_claimed)
            passed = abs(implied_ild_db - cue_in_formula) <= PREMISE_R_TOLERANCE_DB
        else:
            passed = False
        checks.append(
            ClaimCheck(
                "premise_r_matches_its_own_stated_cue",
                r_claimed,
                10.0 ** (cue_in_formula / 20.0),
                PREMISE_R_TOLERANCE_DB,
                passed,
                "inference",
            )
        )

    unparseable = find_unparseable_sentences(transcript, spans)


    all_passed = bool(checks) and all(check.passed for check in checks)
    inference_checks = [check for check in checks if check.consistency_kind == "inference"]
    inference_passed = bool(inference_checks) and all(check.passed for check in inference_checks)

    return VerificationResult(
        checks=tuple(checks),
        all_passed=all_passed,
        inference_passed=inference_passed,
        unparseable_claims=unparseable,
    )


def _claimed_azimuth(transcript: str) -> float | None:
    claim = extract_azimuth(transcript)
    if claim is None:
        return None
    magnitude, side, _ = claim
    return magnitude if side == "left" else -magnitude


def verify_perturbation_pair(
    transcript_a: str,
    evidence_a: CueMeasurements,
    transcript_b: str,
    evidence_b: CueMeasurements,
) -> PerturbationResult | None:
    claimed_a = _claimed_azimuth(transcript_a)
    claimed_b = _claimed_azimuth(transcript_b)
    if claimed_a is None or claimed_b is None:
        return None

    measured_delta = evidence_b.estimated_azimuth_deg - evidence_a.estimated_azimuth_deg
    claimed_delta = claimed_b - claimed_a

    if measured_delta == 0.0:
        tracked = abs(claimed_delta) < AZIMUTH_TOLERANCE_DEG
    else:
        same_direction = (claimed_delta * measured_delta) > 0.0
        covered_half = abs(claimed_delta) >= 0.5 * abs(measured_delta)
        tracked = same_direction and covered_half

    return PerturbationResult(
        quantity="estimated_azimuth_deg",
        claimed_a=claimed_a,
        claimed_b=claimed_b,
        measured_a=evidence_a.estimated_azimuth_deg,
        measured_b=evidence_b.estimated_azimuth_deg,
        measured_delta=measured_delta,
        claimed_delta=claimed_delta,
        tracked=tracked,
    )


def verify_ablation_pair(
    transcript_with_numbers: str,
    evidence: CueMeasurements,
    transcript_without_numbers: str,
) -> AblationResult | None:
    claimed_with = _claimed_azimuth(transcript_with_numbers)
    if claimed_with is None:
        return None

    measured = evidence.estimated_azimuth_deg
    measured_side = _side_for(measured)

    def _passes(claimed: float) -> bool:
        side_ok = measured_side == "center" or _side_for(claimed) == measured_side
        magnitude_ok = abs(abs(claimed) - abs(measured)) <= AZIMUTH_TOLERANCE_DEG
        return side_ok and magnitude_ok

    with_passed = _passes(claimed_with)

    claimed_without = _claimed_azimuth(transcript_without_numbers)
    without_passed = claimed_without is not None and _passes(claimed_without)

    return AblationResult(
        quantity="estimated_azimuth_deg",
        claimed_with_numbers=claimed_with,
        claimed_without_numbers=claimed_without,
        measured=measured,
        tolerance=AZIMUTH_TOLERANCE_DEG,
        with_numbers_passed=with_passed,
        without_numbers_passed=without_passed,
        relied_on_numbers=with_passed and not without_passed,
    )
