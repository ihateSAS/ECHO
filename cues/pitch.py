from __future__ import annotations

import math
from dataclasses import dataclass

import librosa
import numpy as np

PITCH_SAMPLE_RATE = 24_000
DEFAULT_HOP_SECONDS = 0.01
DEFAULT_SILENCE_FLOOR_DB = -60.0
F0_TOLERANCE_PERCENT = 3.0


INSTRUMENT_RANGE_HZ: dict[str, tuple[float, float]] = {
    "vn": (185.00, 2637.02),
    "va": (123.47, 1396.91),
    "vc": (61.74, 1046.50),
    "db": (36.71, 246.94),
    "fl": (246.94, 2489.02),
    "ob": (220.00, 1864.66),
    "cl": (138.59, 1975.53),
    "sax": (123.47, 932.33),
    "bn": (55.00, 659.26),
    "tpt": (155.56, 1244.51),
    "hn": (58.27, 739.99),
    "tbn": (77.78, 698.46),
    "tba": (34.65, 369.99),
}


@dataclass(frozen=True)
class PitchTrack:
    times_seconds: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    voiced_probability: np.ndarray
    hop_seconds: float
    sample_rate: int
    fmin_hz: float
    fmax_hz: float


@dataclass(frozen=True)
class PitchComparison:
    frames_compared: int
    reference_voiced_frames: int
    median_abs_percent_error: float
    mean_abs_percent_error: float
    within_tolerance: float
    tolerance_percent: float
    voicing_agreement: float


def instrument_range_hz(instrument_code: str) -> tuple[float, float]:
    key = instrument_code.strip().lower()
    if key not in INSTRUMENT_RANGE_HZ:
        known = ", ".join(sorted(INSTRUMENT_RANGE_HZ))
        raise ValueError(f"unknown instrument code {instrument_code!r}; known: {known}")
    return INSTRUMENT_RANGE_HZ[key]


def frame_length_for(fmin_hz: float, sample_rate: int) -> int:
    if fmin_hz <= 0.0 or sample_rate <= 0:
        raise ValueError("fmin_hz and sample_rate must be positive")

    needed = 4.0 * sample_rate / fmin_hz
    exponent = max(11, math.ceil(math.log2(needed)))
    return 2**exponent


def track_pitch(
    mono: np.ndarray,
    sample_rate: int,
    fmin_hz: float,
    fmax_hz: float,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    silence_floor_db: float = DEFAULT_SILENCE_FLOOR_DB,
) -> PitchTrack:
    signal = np.asarray(mono, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("mono must be a one-dimensional array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 < fmin_hz < fmax_hz:
        raise ValueError("need 0 < fmin_hz < fmax_hz")
    if hop_seconds <= 0.0:
        raise ValueError("hop_seconds must be positive")
    if not np.all(np.isfinite(signal)):
        raise ValueError("audio contains non-finite samples")

    if sample_rate != PITCH_SAMPLE_RATE:
        signal = librosa.resample(
            signal, orig_sr=sample_rate, target_sr=PITCH_SAMPLE_RATE
        )
    hop_length = max(1, round(hop_seconds * PITCH_SAMPLE_RATE))
    frame_length = frame_length_for(fmin_hz, PITCH_SAMPLE_RATE)
    if signal.size < frame_length:
        raise ValueError(
            f"audio is {signal.size} samples at {PITCH_SAMPLE_RATE} Hz; pitch "
            f"tracking down to {fmin_hz:g} Hz needs at least {frame_length}"
        )
    if fmax_hz >= PITCH_SAMPLE_RATE / 2.0:
        raise ValueError(
            f"fmax_hz must stay below the {PITCH_SAMPLE_RATE / 2:g} Hz "
            "analysis Nyquist frequency"
        )

    f0_hz, voiced, voiced_probability = librosa.pyin(
        signal,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=PITCH_SAMPLE_RATE,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    voiced_probability = np.asarray(voiced_probability, dtype=np.float64)

    loud_enough = _loud_enough(signal, frame_length, hop_length, silence_floor_db)
    voiced &= loud_enough[: voiced.size]
    f0_hz = np.where(voiced, f0_hz, np.nan)

    hop = hop_length / PITCH_SAMPLE_RATE
    return PitchTrack(
        times_seconds=np.arange(f0_hz.size, dtype=np.float64) * hop,
        f0_hz=f0_hz,
        voiced=voiced,
        voiced_probability=voiced_probability,
        hop_seconds=hop,
        sample_rate=PITCH_SAMPLE_RATE,
        fmin_hz=float(fmin_hz),
        fmax_hz=float(fmax_hz),
    )


def _loud_enough(
    signal: np.ndarray, frame_length: int, hop_length: int, silence_floor_db: float
) -> np.ndarray:
    rms = librosa.feature.rms(
        y=signal, frame_length=frame_length, hop_length=hop_length
    )[0]
    loudest = float(np.max(rms))
    if loudest <= np.finfo(np.float64).tiny:
        return np.zeros(rms.size, dtype=bool)
    return rms >= loudest * 10.0 ** (silence_floor_db / 20.0)


def track_instrument_pitch(
    mono: np.ndarray,
    sample_rate: int,
    instrument_code: str,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> PitchTrack:
    fmin_hz, fmax_hz = instrument_range_hz(instrument_code)
    return track_pitch(mono, sample_rate, fmin_hz, fmax_hz, hop_seconds=hop_seconds)


def align_to_times(track: PitchTrack, target_times_seconds: np.ndarray) -> np.ndarray:
    targets = np.asarray(target_times_seconds, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("target_times_seconds must be one-dimensional")
    if track.times_seconds.size == 0:
        raise ValueError("track has no frames")

    times = track.times_seconds
    right = np.searchsorted(times, targets)
    left = np.clip(right - 1, 0, times.size - 1)
    right = np.clip(right, 0, times.size - 1)
    take_left = np.abs(times[left] - targets) <= np.abs(times[right] - targets)
    nearest = np.where(take_left, left, right)

    aligned = track.f0_hz[nearest]
    too_far = np.abs(times[nearest] - targets) > track.hop_seconds / 2.0
    return np.where(too_far, np.nan, aligned)


def percent_error(estimated_hz: np.ndarray, reference_hz: np.ndarray) -> np.ndarray:
    estimated = np.asarray(estimated_hz, dtype=np.float64)
    reference = np.asarray(reference_hz, dtype=np.float64)
    if estimated.shape != reference.shape:
        raise ValueError("estimated_hz and reference_hz must have the same shape")
    if np.any(reference <= 0.0):
        raise ValueError("reference_hz must be positive everywhere")
    return 100.0 * np.abs(estimated - reference) / reference


def compare_to_reference(
    track: PitchTrack,
    reference_times_seconds: np.ndarray,
    reference_f0_hz: np.ndarray,
    tolerance_percent: float = F0_TOLERANCE_PERCENT,
) -> PitchComparison:
    reference_times = np.asarray(reference_times_seconds, dtype=np.float64)
    reference_f0 = np.asarray(reference_f0_hz, dtype=np.float64)
    if reference_times.shape != reference_f0.shape:
        raise ValueError("reference times and f0 arrays must have the same shape")
    if reference_times.size == 0:
        raise ValueError("reference annotation is empty")
    if tolerance_percent <= 0.0:
        raise ValueError("tolerance_percent must be positive")

    aligned = align_to_times(track, reference_times)
    reference_voiced = reference_f0 > 0.0
    estimated_voiced = np.isfinite(aligned)
    both_voiced = reference_voiced & estimated_voiced
    if not np.any(both_voiced):
        raise ValueError("no frames where both the track and the reference are voiced")

    errors = percent_error(aligned[both_voiced], reference_f0[both_voiced])
    return PitchComparison(
        frames_compared=int(np.count_nonzero(both_voiced)),
        reference_voiced_frames=int(np.count_nonzero(reference_voiced)),
        median_abs_percent_error=float(np.median(errors)),
        mean_abs_percent_error=float(np.mean(errors)),
        within_tolerance=float(np.mean(errors <= tolerance_percent)),
        tolerance_percent=float(tolerance_percent),
        voicing_agreement=float(np.mean(reference_voiced == estimated_voiced)),
    )
