from __future__ import annotations

import math

import numpy as np
from scipy.signal import istft, stft

HEAD_RADIUS_M = 0.0875
SPEED_OF_SOUND_MPS = 343.0
MAX_ANGLE_DEGREES = 90.0
MAX_HEAD_SHADOW_DB = 20.0
HEAD_SHADOW_CORNER_HZ = 1_500.0
STFT_NPERSEG = 1024


def woodworth_itd_seconds(angle_radians: float) -> float:
    return (HEAD_RADIUS_M / SPEED_OF_SOUND_MPS) * (
        angle_radians + math.sin(angle_radians)
    )


def woodworth_itd_us(angle_degrees: float) -> float:
    return woodworth_itd_seconds(math.radians(angle_degrees)) * 1_000_000.0


def apply_fractional_delay(mono: np.ndarray, delay_samples: float) -> np.ndarray:
    if delay_samples < 0.0:
        raise ValueError("delay_samples must be non-negative")

    whole = int(math.floor(delay_samples))
    frac = delay_samples - whole
    extended = np.concatenate((np.zeros(whole + 1, dtype=mono.dtype), mono))
    return (1.0 - frac) * extended[1:] + frac * extended[:-1]


def head_shadow_gain_db(frequencies: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle_scale = abs(math.sin(math.radians(angle_degrees)))
    shadow_curve = frequencies / (frequencies + HEAD_SHADOW_CORNER_HZ)
    return -MAX_HEAD_SHADOW_DB * angle_scale * shadow_curve


def apply_head_shadow(mono: np.ndarray, sample_rate: int, angle_degrees: float) -> np.ndarray:
    nperseg = min(STFT_NPERSEG, mono.size)
    frequencies, _, spectrum = stft(
        mono, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=nperseg // 2
    )
    gain = 10.0 ** (head_shadow_gain_db(frequencies, angle_degrees) / 20.0)
    _, filtered = istft(
        spectrum * gain[:, None],
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    if filtered.size < mono.size:
        filtered = np.concatenate((filtered, np.zeros(mono.size - filtered.size)))
    return filtered[: mono.size]


def binaural_render(mono: np.ndarray, sample_rate: int, angle_degrees: float) -> np.ndarray:
    if not -MAX_ANGLE_DEGREES <= angle_degrees <= MAX_ANGLE_DEGREES:
        raise ValueError(
            f"angle must be between {-MAX_ANGLE_DEGREES:g} and "
            f"{MAX_ANGLE_DEGREES:g} degrees"
        )

    delay_us = woodworth_itd_us(angle_degrees)
    delay_samples = abs(delay_us) * sample_rate / 1_000_000.0

    far_ear = apply_head_shadow(mono, sample_rate, angle_degrees)
    far_ear_delayed = apply_fractional_delay(far_ear, delay_samples)

    pad = far_ear_delayed.size - mono.size
    near_ear = np.concatenate((mono, np.zeros(pad, dtype=mono.dtype)))

    if delay_us >= 0.0:
        return np.column_stack((near_ear, far_ear_delayed))
    return np.column_stack((far_ear_delayed, near_ear))
