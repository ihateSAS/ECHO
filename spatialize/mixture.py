from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatialize.amplitude import render_amplitude_pan


@dataclass(frozen=True)
class StemPlacement:
    mono: np.ndarray
    azimuth_deg: float
    label: str


@dataclass(frozen=True)
class MixtureGroundTruth:
    label: str
    azimuth_deg: float


def mix_stems(
    placements: list[StemPlacement],
) -> tuple[np.ndarray, list[MixtureGroundTruth]]:
    if not placements:
        raise ValueError("placements must not be empty")

    longest = max(placement.mono.shape[0] for placement in placements)
    mix = np.zeros((longest, 2), dtype=np.float64)
    ground_truth: list[MixtureGroundTruth] = []

    for placement in placements:
        mono = np.asarray(placement.mono, dtype=np.float64)
        if mono.ndim != 1:
            raise ValueError(f"stem {placement.label!r} must be one-dimensional")
        padded = np.pad(mono, (0, longest - mono.shape[0]))
        mix += render_amplitude_pan(padded, placement.azimuth_deg)
        ground_truth.append(MixtureGroundTruth(placement.label, placement.azimuth_deg))

    return mix, ground_truth
