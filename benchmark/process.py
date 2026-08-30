from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from benchmark.schemas import ProcessingStep
from spatialize.amplitude import SPEAKER_ANGLE_DEG, gains_for_azimuth
from spatialize.compression import stereo_linked_compressor
from spatialize.delay_widening import widen_with_delay
from spatialize.eq import apply_peaking_eq
from spatialize.midside import apply_width
from spatialize.reverb import schroeder_reverb


def apply_processing_step(
    stereo: np.ndarray, sample_rate: int, step: ProcessingStep
) -> np.ndarray:
    if step.kind == "reverb":
        return schroeder_reverb(stereo, sample_rate, **step.params)
    if step.kind == "compression":
        return stereo_linked_compressor(stereo, sample_rate, **step.params)
    if step.kind == "eq":
        return apply_peaking_eq(stereo, sample_rate, **step.params)
    if step.kind == "midside":
        return apply_width(stereo, **step.params)
    if step.kind == "delay_widening":
        return widen_with_delay(stereo, sample_rate, **step.params)
    raise ValueError(f"unknown processing kind: {step.kind!r}")


def apply_processing_chain(
    stereo: np.ndarray, sample_rate: int, steps: tuple[ProcessingStep, ...]
) -> np.ndarray:
    for step in steps:
        stereo = apply_processing_step(stereo, sample_rate, step)
    return stereo


@dataclass(frozen=True)
class ExpectedCues:
    azimuth_deg: float | None
    itd_us: float | None
    regime: str | None
    coherence_stays_high: bool

    notes: tuple[str, ...] = ()


def _azimuth_after_width(azimuth_deg: float, width: float) -> float | None:
    left, right = gains_for_azimuth(azimuth_deg)
    d = (left - right) / (left + right)
    widened = width * d
    if abs(widened) > 1.0:
        return None
    return math.degrees(
        math.atan(widened * math.tan(math.radians(SPEAKER_ANGLE_DEG)))
    )


def expected_after_processing(
    azimuth_deg: float, steps: tuple[ProcessingStep, ...]
) -> ExpectedCues:
    azimuth: float | None = azimuth_deg
    itd_us: float | None = 0.0
    regime: str | None = "amplitude"
    coherence_stays_high = True
    notes: list[str] = []

    for step in steps:
        if step.kind in ("compression", "eq"):
            continue
        if step.kind == "midside":
            if azimuth is None:
                continue
            width = float(step.params["width"])
            widened = _azimuth_after_width(azimuth, width)
            if widened is None:
                azimuth = None
                notes.append(
                    f"midside width {width:g} drives the panning coordinate past "
                    "1, which inverts polarity rather than widening"
                )
            else:
                azimuth = widened
        elif step.kind == "delay_widening":
            itd_us = float(step.params["delay_us"])
            regime = "delayed" if itd_us != 0.0 else regime
        elif step.kind == "reverb":
            if float(step.params.get("wet", 0.0)) == 0.0:
                continue
            azimuth = None
            regime = None
            coherence_stays_high = False
            notes.append(
                "reverb pulls the level ratio toward centre by a "
                "source-dependent amount, so no azimuth can be re-derived"
            )
        else:
            raise ValueError(f"unknown processing kind: {step.kind!r}")

    return ExpectedCues(
        azimuth_deg=azimuth,
        itd_us=itd_us,
        regime=regime,
        coherence_stays_high=coherence_stays_high,
        notes=tuple(notes),
    )


def processing_label(steps: tuple[ProcessingStep, ...]) -> str:
    if not steps:
        return "dry"

    def _step_label(step: ProcessingStep) -> str:
        params = "_".join(
            f"{key}{value:g}".replace(".", "p").replace("-", "neg")
            for key, value in sorted(step.params.items())
        )
        return f"{step.kind}_{params}" if params else step.kind

    return "_".join(_step_label(step) for step in steps)
