from __future__ import annotations

import numpy as np

from benchmark.schemas import MixtureCueMeasurements
from cues.mixture import estimate_mixture_level_cues


def measure_mixture(stereo: np.ndarray, sample_rate: int) -> MixtureCueMeasurements:
    estimate = estimate_mixture_level_cues(stereo, sample_rate)
    return MixtureCueMeasurements(
        source_azimuths_deg=estimate.azimuths_deg,
        source_energy_fractions=tuple(
            source.energy_fraction for source in estimate.sources
        ),
        active_bins=estimate.active_bins,
    )
