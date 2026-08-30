from __future__ import annotations

import numpy as np

from benchmark.config import BenchmarkConfig
from benchmark.schemas import MixtureCueMeasurements, MixtureRenderPlan
from benchmark.validate import rms_db


def validate_mixture_render(
    stereo: np.ndarray,
    plan: MixtureRenderPlan,
    evidence: MixtureCueMeasurements,
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

    planted = sorted(stem.azimuth_deg for stem in plan.stems)
    measured = sorted(evidence.source_azimuths_deg)

    if len(measured) != len(planted):
        failures.append("source_count_mismatch")
        return failures


    for measured_angle, planted_angle in zip(measured, planted):
        if abs(measured_angle - planted_angle) > config.azimuth_tolerance_degrees:
            failures.append("azimuth_recovery_failed")
            break

    return failures
