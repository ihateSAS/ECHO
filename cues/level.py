from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

SPEAKER_ANGLE_DEG = 30.0
ACTIVE_LEVEL_DB = -60.0


@dataclass(frozen=True)
class LevelEstimate:
    broadband_ild_db: float
    high_frequency_ild_db: float
    ild_flatness_db: float
    azimuth_deg: float
    active_bins: int


def _validate_stereo(stereo: np.ndarray) -> np.ndarray:
    audio = np.asarray(stereo, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("stereo must have shape (samples, 2)")
    if audio.shape[0] < 16:
        raise ValueError("audio is too short to analyze")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains non-finite samples")
    if float(np.max(np.abs(audio))) <= np.finfo(np.float64).eps:
        raise ValueError("audio contains only silence")
    return audio


def _channel_spectrogram(channel: np.ndarray, sample_rate: int, nperseg: int):
    frequencies, _, spectrum = stft(
        channel,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        boundary=None,
        padded=False,
    )
    return frequencies, np.abs(spectrum)


def _channel_spectrum(channel: np.ndarray, sample_rate: int, nperseg: int):
    frequencies, magnitude = _channel_spectrogram(channel, sample_rate, nperseg)
    return frequencies, np.sqrt(np.mean(magnitude**2, axis=1))


def _angle_from_ild(ild_db: float) -> float:
    ratio = 10.0 ** (ild_db / 20.0)
    difference = (ratio - 1.0) / (ratio + 1.0)
    return math.degrees(
        math.atan(difference * math.tan(math.radians(SPEAKER_ANGLE_DEG)))
    )


def azimuth_from_ild_db(ild_db: float) -> float:
    return _angle_from_ild(ild_db)


def ild_db_for_azimuth(azimuth_deg: float) -> float:
    if abs(azimuth_deg) >= SPEAKER_ANGLE_DEG:
        raise ValueError(
            f"|azimuth| must stay under {SPEAKER_ANGLE_DEG:g} degrees; "
            f"got {azimuth_deg:g}"
        )
    difference = math.tan(math.radians(azimuth_deg)) / math.tan(
        math.radians(SPEAKER_ANGLE_DEG)
    )
    return 20.0 * math.log10((1.0 + difference) / (1.0 - difference))


@dataclass(frozen=True)
class ActiveBins:
    frequencies: np.ndarray
    left: np.ndarray
    right: np.ndarray
    active: np.ndarray

    @property
    def ild_db(self) -> np.ndarray:
        return 20.0 * np.log10(self.left[self.active] / self.right[self.active])

    @property
    def energy(self) -> np.ndarray:
        return (
            self.left[self.active] ** 2 + self.right[self.active] ** 2
        )

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.active))


def active_bins(
    stereo: np.ndarray, sample_rate: int, per_frame: bool = False
) -> ActiveBins:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    audio = _validate_stereo(stereo)
    nperseg = min(4096, audio.shape[0])
    spectrum = _channel_spectrogram if per_frame else _channel_spectrum
    frequencies, left = spectrum(audio[:, 0], sample_rate, nperseg)
    _, right = spectrum(audio[:, 1], sample_rate, nperseg)

    combined = np.hypot(left, right)
    threshold = float(np.max(combined)) * 10.0 ** (ACTIVE_LEVEL_DB / 20.0)
    floor = np.finfo(np.float64).tiny
    active = (combined >= threshold) & (left > floor) & (right > floor)
    if not np.any(active):
        raise ValueError("audio has no usable frequency bins")
    return ActiveBins(frequencies=frequencies, left=left, right=right, active=active)


def estimate_level_cues(stereo: np.ndarray, sample_rate: int) -> LevelEstimate:
    bins = active_bins(stereo, sample_rate)
    frequencies, left, right, active = (
        bins.frequencies,
        bins.left,
        bins.right,
        bins.active,
    )

    ild = bins.ild_db
    broadband = float(np.median(ild))

    high = active & (frequencies >= 2_000.0)
    high_frequency = (
        float(np.median(20.0 * np.log10(left[high] / right[high])))
        if np.any(high)
        else broadband
    )

    band_ilds: list[float] = []
    half_octave = math.sqrt(2.0)
    for center in (125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0):
        in_band = active & (frequencies >= center / half_octave) & (
            frequencies < center * half_octave
        )
        if np.any(in_band):
            band_ilds.append(
                float(np.median(20.0 * np.log10(left[in_band] / right[in_band])))
            )
    flatness = float(np.std(band_ilds)) if len(band_ilds) >= 2 else 0.0

    return LevelEstimate(
        broadband_ild_db=broadband,
        high_frequency_ild_db=high_frequency,
        ild_flatness_db=flatness,
        azimuth_deg=_angle_from_ild(broadband),
        active_bins=bins.count,
    )
