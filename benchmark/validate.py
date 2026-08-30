from __future__ import annotations

import numpy as np

from benchmark.config import BenchmarkConfig
from benchmark.process import expected_after_processing
from benchmark.schemas import CueMeasurements, RenderPlan
from spatialize.binaural import woodworth_itd_us


ITD_TOLERANCE_US = 30.0
MINIMUM_COHERENCE = 0.90


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    if rms <= np.finfo(np.float64).eps:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def _binaural_failures(
    plan: RenderPlan, evidence: CueMeasurements, config: BenchmarkConfig
) -> list[str]:
    failures: list[str] = []
    expected_itd_us = woodworth_itd_us(plan.azimuth_deg)
    if abs(evidence.itd_us - expected_itd_us) > ITD_TOLERANCE_US:
        failures.append("binaural_itd_recovery_failed")

    if not np.isfinite(evidence.binaural_azimuth_deg):
        failures.append("binaural_azimuth_not_recoverable")
    elif abs(evidence.binaural_azimuth_deg - plan.azimuth_deg) > (
        config.azimuth_tolerance_degrees
    ):
        failures.append("binaural_azimuth_recovery_failed")


    if abs(expected_itd_us) > ITD_TOLERANCE_US:
        if evidence.estimated_regime != "delayed":
            failures.append("binaural_render_does_not_read_as_delayed")
    if evidence.coherence < MINIMUM_COHERENCE:
        failures.append("compact_source_has_low_coherence")
    return failures


def validate_render(
    stereo: np.ndarray,
    plan: RenderPlan,
    evidence: CueMeasurements,
    config: BenchmarkConfig,
) -> list[str]:
    failures: list[str] = []
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        failures.append("output_is_not_stereo")
        return failures
    if not np.all(np.isfinite(stereo)):
        failures.append("non_finite_samples")
    if float(np.max(np.abs(stereo))) > config.maximum_peak:
        failures.append("peak_too_high")
    if rms_db(stereo) < config.minimum_rms_db:
        failures.append("audio_too_quiet")

    if plan.regime == "binaural":
        return failures + _binaural_failures(plan, evidence, config)

    expected = expected_after_processing(plan.azimuth_deg, plan.processing)

    if expected.azimuth_deg is None:
        failures.append("azimuth_not_recoverable_after_processing")
    elif abs(evidence.estimated_azimuth_deg - expected.azimuth_deg) > (
        config.azimuth_tolerance_degrees
    ):
        failures.append("azimuth_recovery_failed")

    if expected.itd_us is not None:
        if abs(evidence.itd_us - expected.itd_us) > ITD_TOLERANCE_US:
            failures.append("itd_does_not_match_the_processing")
    if expected.regime is not None and evidence.estimated_regime != expected.regime:
        failures.append("wrong_regime")
    if expected.coherence_stays_high and evidence.coherence < MINIMUM_COHERENCE:
        failures.append("compact_source_has_low_coherence")
    return failures
