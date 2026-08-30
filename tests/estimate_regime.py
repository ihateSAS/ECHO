#!/usr/bin/env python3


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import next_fast_len

REQUIRED_SAMPLE_RATE = 48_000
ANALYSIS_SAMPLES = 65_536
MAX_DELAY_US = 700.0
DELAY_THRESHOLD_US = 30.0
PHASE_THRESHOLD_DEGREES = 10.0
PHASE_REFERENCE_HZ = 1_000.0
FLATNESS_THRESHOLD_DB = 2.0
ACTIVE_LEVEL_DB = -60.0
OCTAVE_CENTERS_HZ = (125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0)


@dataclass(frozen=True)
class RegimeEstimate:
    regime: str
    delay_us: float
    phase_delay_us: float
    phase_at_1khz_degrees: float
    level_flatness_db: float
    octave_bands: int
    delayed_votes: int


def read_stereo(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected stereo")
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sample_rate} Hz; delay estimation requires "
            f"{REQUIRED_SAMPLE_RATE} Hz"
        )
    if audio.shape[0] < 16:
        raise ValueError(f"{path} is too short to analyze")
    if float(np.max(np.abs(audio))) <= np.finfo(np.float64).eps:
        raise ValueError(f"{path} contains only silence")
    return audio, sample_rate


def highest_energy_segment(audio: np.ndarray) -> np.ndarray:
    frame_length = min(ANALYSIS_SAMPLES, audio.shape[0])
    if frame_length == audio.shape[0]:
        segment = audio.copy()
    else:
        hop = frame_length // 2
        starts = range(0, audio.shape[0] - frame_length + 1, hop)
        start = max(
            starts,
            key=lambda index: float(
                np.sum(audio[index : index + frame_length] ** 2)
            ),
        )
        segment = audio[start : start + frame_length].copy()

    segment -= np.mean(segment, axis=0, keepdims=True)
    channel_rms = np.sqrt(np.mean(segment**2, axis=0))
    if float(np.max(channel_rms)) <= np.finfo(np.float64).eps:
        raise ValueError("the selected analysis frame is silent")
    if float(np.min(channel_rms)) < float(np.max(channel_rms)) * 1e-6:
        raise ValueError("one channel has too little energy for regime detection")
    return segment


def gcc_phat_delay_us(
    left: np.ndarray, right: np.ndarray, sample_rate: int
) -> float:
    nfft = next_fast_len(2 * left.size - 1)
    left_spectrum = np.fft.rfft(left, n=nfft)
    right_spectrum = np.fft.rfft(right, n=nfft)
    cross_spectrum = left_spectrum * np.conj(right_spectrum)
    magnitude = np.abs(cross_spectrum)

    normalized = np.zeros_like(cross_spectrum)
    usable = magnitude > float(np.max(magnitude)) * 1e-12
    normalized[usable] = cross_spectrum[usable] / magnitude[usable]
    correlation = np.fft.irfft(normalized, n=nfft)

    max_shift = int(np.ceil(MAX_DELAY_US * sample_rate / 1_000_000.0))
    lags = np.arange(-max_shift, max_shift + 1)
    values = correlation[np.mod(lags, nfft)]
    peak_index = int(np.argmax(values))
    peak_lag = float(lags[peak_index])

    if 0 < peak_index < values.size - 1:
        before, peak, after = values[peak_index - 1 : peak_index + 2]
        denominator = before - 2.0 * peak + after
        if abs(float(denominator)) > np.finfo(np.float64).eps:
            peak_lag += float(0.5 * (before - after) / denominator)

    return -peak_lag / sample_rate * 1_000_000.0


def spectra(
    left: np.ndarray, right: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window = np.hanning(left.size)
    left_spectrum = np.fft.rfft(left * window)
    right_spectrum = np.fft.rfft(right * window)
    frequencies = np.fft.rfftfreq(left.size, d=1.0 / sample_rate)
    cross_spectrum = left_spectrum * np.conj(right_spectrum)
    return frequencies, np.abs(left_spectrum), np.abs(right_spectrum), cross_spectrum


def phase_slope_delay_us(
    frequencies: np.ndarray,
    left_magnitude: np.ndarray,
    right_magnitude: np.ndarray,
    cross_spectrum: np.ndarray,
) -> float:
    combined = np.hypot(left_magnitude, right_magnitude)
    threshold = float(np.max(combined)) * 10.0 ** (ACTIVE_LEVEL_DB / 20.0)
    usable = (
        (combined >= threshold)
        & (np.abs(cross_spectrum) > np.finfo(np.float64).tiny)
        & (frequencies >= 80.0)
        & (frequencies <= 8_000.0)
    )
    if np.count_nonzero(usable) < 16:
        raise ValueError("not enough active bins to measure phase slope")

    active_frequencies = frequencies[usable]
    measured_phase = np.angle(cross_spectrum[usable])
    candidates_us = np.arange(-MAX_DELAY_US, MAX_DELAY_US + 1.0, 1.0)
    scores = np.array(
        [
            np.mean(
                np.cos(
                    measured_phase
                    - 2.0
                    * np.pi
                    * active_frequencies
                    * candidate_us
                    * 1e-6
                )
            )
            for candidate_us in candidates_us
        ]
    )
    peak_index = int(np.argmax(scores))
    fitted_delay_us = float(candidates_us[peak_index])

    if 0 < peak_index < scores.size - 1:
        before, peak, after = scores[peak_index - 1 : peak_index + 2]
        denominator = before - 2.0 * peak + after
        if abs(float(denominator)) > np.finfo(np.float64).eps:
            fitted_delay_us += float(0.5 * (before - after) / denominator)
    return fitted_delay_us


def level_difference_flatness_db(
    frequencies: np.ndarray,
    left_magnitude: np.ndarray,
    right_magnitude: np.ndarray,
) -> tuple[float, int]:
    combined = np.hypot(left_magnitude, right_magnitude)
    threshold = float(np.max(combined)) * 10.0 ** (ACTIVE_LEVEL_DB / 20.0)
    floor = np.finfo(np.float64).tiny
    active = (
        (combined >= threshold)
        & (left_magnitude > floor)
        & (right_magnitude > floor)
    )

    band_ilds = []
    half_octave = np.sqrt(2.0)
    for center_hz in OCTAVE_CENTERS_HZ:
        in_band = (
            active
            & (frequencies >= center_hz / half_octave)
            & (frequencies < center_hz * half_octave)
        )
        if np.any(in_band):
            ild = 20.0 * np.log10(
                left_magnitude[in_band] / right_magnitude[in_band]
            )
            band_ilds.append(float(np.median(ild)))

    if len(band_ilds) < 3:
        raise ValueError("not enough active octave bands to measure ILD flatness")
    return float(np.std(band_ilds)), len(band_ilds)


def estimate_regime(path: Path) -> RegimeEstimate:
    audio, sample_rate = read_stereo(path)
    segment = highest_energy_segment(audio)
    left, right = segment[:, 0], segment[:, 1]

    delay_us = gcc_phat_delay_us(left, right, sample_rate)
    frequencies, left_magnitude, right_magnitude, cross_spectrum = spectra(
        left, right, sample_rate
    )
    phase_delay_us = phase_slope_delay_us(
        frequencies, left_magnitude, right_magnitude, cross_spectrum
    )
    phase_at_1khz_degrees = (
        phase_delay_us * 1e-6 * PHASE_REFERENCE_HZ * 360.0
    )
    flatness_db, octave_bands = level_difference_flatness_db(
        frequencies, left_magnitude, right_magnitude
    )

    delayed_votes = sum(
        (
            abs(delay_us) > DELAY_THRESHOLD_US,
            abs(phase_at_1khz_degrees) > PHASE_THRESHOLD_DEGREES,
            flatness_db > FLATNESS_THRESHOLD_DB,
        )
    )
    regime = "delayed" if delayed_votes >= 2 else "amplitude-panned"
    return RegimeEstimate(
        regime=regime,
        delay_us=delay_us,
        phase_delay_us=phase_delay_us,
        phase_at_1khz_degrees=phase_at_1khz_degrees,
        level_flatness_db=flatness_db,
        octave_bands=octave_bands,
        delayed_votes=delayed_votes,
    )


def vote_label(is_delayed: bool) -> str:
    return "delayed" if is_delayed else "amplitude-panned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify 48 kHz stereo WAV files using GCC-PHAT delay, "
            "phase-versus-frequency behavior, and level-difference flatness."
        )
    )
    parser.add_argument("files", type=Path, nargs="+", help="stereo files to analyze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for path in args.files:
        try:
            result = estimate_regime(path)
            print(
                f"{path}: regime={result.regime}, "
                f"delay={result.delay_us:+.2f} us "
                f"[{vote_label(abs(result.delay_us) > DELAY_THRESHOLD_US)}], "
                f"phase@1kHz={result.phase_at_1khz_degrees:+.2f}° "
                f"[{vote_label(abs(result.phase_at_1khz_degrees) > PHASE_THRESHOLD_DEGREES)}], "
                f"ILD-flatness={result.level_flatness_db:.3f} dB "
                f"[{vote_label(result.level_flatness_db > FLATNESS_THRESHOLD_DB)}], "
                f"delayed-votes={result.delayed_votes}/3, "
                f"octave-bands={result.octave_bands}"
            )
        except (OSError, ValueError) as error:
            failed = True
            print(f"{path}: error: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
