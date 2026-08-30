from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import coherence

MIN_BAND_HZ = 80.0
MAX_BAND_HZ = 8_000.0
MIN_SEGMENTS = 32
MIN_NPERSEG = 256
MAX_NPERSEG = 4096


@dataclass(frozen=True)
class CoherenceEstimate:
    value: float
    active_bins: int
    segment_length: int = 0
    segments: int = 0


def welch_segment_length(num_samples: int) -> int:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    longest = 2 * num_samples // (MIN_SEGMENTS + 1)
    segment_length = min(MAX_NPERSEG, longest)
    if segment_length < MIN_NPERSEG:
        shortest = MIN_NPERSEG * (MIN_SEGMENTS + 1) // 2
        raise ValueError(
            f"{num_samples} samples cannot be split into {MIN_SEGMENTS} "
            f"segments of at least {MIN_NPERSEG} samples (needs {shortest}); "
            "coherence averaged over too few segments is biased toward 1.0 "
            "and does not mean anything"
        )
    return segment_length


def welch_segment_count(num_samples: int, segment_length: int) -> int:
    if segment_length <= 1:
        raise ValueError("segment_length must be greater than one")
    if num_samples < segment_length:
        return 0
    return (num_samples - segment_length) // (segment_length // 2) + 1


def estimate_coherence(stereo: np.ndarray, sample_rate: int) -> CoherenceEstimate:
    audio = np.asarray(stereo, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("stereo must have shape (samples, 2)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains non-finite samples")

    segment_length = welch_segment_length(audio.shape[0])
    frequencies, gamma_sq = coherence(
        audio[:, 0], audio[:, 1], fs=sample_rate, nperseg=segment_length
    )
    in_band = (frequencies >= MIN_BAND_HZ) & (frequencies <= MAX_BAND_HZ)
    finite = in_band & np.isfinite(gamma_sq)
    if not np.any(finite):
        raise ValueError("no usable coherence bins")

    return CoherenceEstimate(
        value=float(np.median(gamma_sq[finite])),
        active_bins=int(np.count_nonzero(finite)),
        segment_length=segment_length,
        segments=welch_segment_count(audio.shape[0], segment_length),
    )
