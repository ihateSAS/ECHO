from __future__ import annotations

import numpy as np

from benchmark.schemas import CueMeasurements
from cues.binaural import invert_woodworth
from cues.coherence import estimate_coherence
from cues.delay import estimate_itd_us
from cues.level import estimate_level_cues
from cues.pitch import instrument_range_hz, track_pitch
from cues.regime import classify_regime


def _binaural_azimuth_deg(itd_us: float) -> float:
    if not np.isfinite(itd_us):
        return float("nan")
    try:
        return invert_woodworth(float(itd_us))
    except ValueError:
        return float("nan")


def _median_f0_hz(mono: np.ndarray, sample_rate: int, instrument_code: str) -> float:
    fmin_hz, fmax_hz = instrument_range_hz(instrument_code)
    try:
        track = track_pitch(mono, sample_rate, fmin_hz, fmax_hz)
    except ValueError:
        return float("nan")
    voiced = track.f0_hz[np.isfinite(track.f0_hz)]
    if voiced.size == 0:
        return float("nan")
    return float(np.median(voiced))


def measure_render(
    stereo: np.ndarray, sample_rate: int, instrument_code: str | None = None
) -> CueMeasurements:
    level = estimate_level_cues(stereo, sample_rate)
    delay = estimate_itd_us(stereo, sample_rate)
    coherence = estimate_coherence(stereo, sample_rate)
    regime = classify_regime(level, delay)

    f0_hz = float("nan")
    if instrument_code is not None:
        mono = np.asarray(stereo, dtype=np.float64).mean(axis=1)
        f0_hz = _median_f0_hz(mono, sample_rate, instrument_code)

    return CueMeasurements(
        ild_db=level.broadband_ild_db,
        high_frequency_ild_db=level.high_frequency_ild_db,
        ild_flatness_db=level.ild_flatness_db,
        itd_us=delay.itd_us,
        itd_peak_strength=delay.peak_strength,
        coherence=coherence.value,
        estimated_azimuth_deg=level.azimuth_deg,
        estimated_regime=regime.label,
        active_bins=level.active_bins,
        f0_hz=f0_hz,
        binaural_azimuth_deg=_binaural_azimuth_deg(delay.itd_us),
    )
