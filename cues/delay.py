from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len


@dataclass(frozen=True)
class DelayEstimate:
    itd_us: float
    peak_strength: float
    reliable: bool


def estimate_itd_us(
    stereo: np.ndarray,
    sample_rate: int,
    max_delay_us: float = 700.0,
) -> DelayEstimate:
    audio = np.asarray(stereo, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("stereo must have shape (samples, 2)")
    if audio.shape[0] < 16:
        raise ValueError("audio is too short to analyze")
    if sample_rate <= 0 or max_delay_us <= 0:
        raise ValueError("sample_rate and max_delay_us must be positive")

    left = audio[:, 0] - np.mean(audio[:, 0])
    right = audio[:, 1] - np.mean(audio[:, 1])
    nfft = next_fast_len(2 * audio.shape[0] - 1)
    cross = np.fft.rfft(left, nfft) * np.conj(np.fft.rfft(right, nfft))
    magnitude = np.abs(cross)
    if float(np.max(magnitude)) <= np.finfo(np.float64).eps:
        raise ValueError("audio has insufficient energy for delay estimation")

    normalized = np.zeros_like(cross)
    usable = magnitude > float(np.max(magnitude)) * 1e-12
    normalized[usable] = cross[usable] / magnitude[usable]
    correlation = np.fft.irfft(normalized, nfft)

    max_shift = int(np.ceil(max_delay_us * sample_rate / 1_000_000.0))
    lags = np.arange(-max_shift, max_shift + 1)
    values = correlation[np.mod(lags, nfft)]
    peak_index = int(np.argmax(values))
    peak_lag = float(lags[peak_index])

    if 0 < peak_index < values.size - 1:
        before, peak, after = values[peak_index - 1 : peak_index + 2]
        denominator = before - 2.0 * peak + after
        if abs(float(denominator)) > np.finfo(np.float64).eps:
            peak_lag += float(0.5 * (before - after) / denominator)

    absolute_values = np.abs(values)
    peak_strength = float(absolute_values[peak_index] / (np.sum(absolute_values) + 1e-15))
    return DelayEstimate(
        itd_us=-peak_lag / sample_rate * 1_000_000.0,
        peak_strength=peak_strength,
        reliable=bool(peak_strength > 0.05),
    )
