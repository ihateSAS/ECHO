#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf

REQUIRED_SAMPLE_RATE = 48_000
SPEAKER_ANGLE_DEGREES = 30.0


def read_mono(path: Path) -> tuple[np.ndarray, int, str]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(
            f"{path} has {audio.shape[1]} channels; the input must be mono"
        )
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sample_rate} Hz; regime delay tests require "
            f"{REQUIRED_SAMPLE_RATE} Hz"
        )
    return audio[:, 0], sample_rate, info.subtype


def gains_for_angle(angle_degrees: float) -> tuple[float, float]:
    if not -SPEAKER_ANGLE_DEGREES <= angle_degrees <= SPEAKER_ANGLE_DEGREES:
        raise ValueError(
            f"angle must be between {-SPEAKER_ANGLE_DEGREES:g} and "
            f"{SPEAKER_ANGLE_DEGREES:g} degrees"
        )

    difference = math.tan(math.radians(angle_degrees)) / math.tan(
        math.radians(SPEAKER_ANGLE_DEGREES)
    )
    left = (1.0 + difference) / 2.0
    right = (1.0 - difference) / 2.0
    norm = math.hypot(left, right)
    return left / norm, right / norm


def amplitude_pan(mono: np.ndarray, angle_degrees: float) -> np.ndarray:
    left_gain, right_gain = gains_for_angle(angle_degrees)
    return np.column_stack((mono * left_gain, mono * right_gain))


def samples_for_delay(delay_us: float, sample_rate: int) -> int:
    if delay_us <= 0.0:
        raise ValueError("delay must be positive")

    exact_samples = delay_us * sample_rate / 1_000_000.0
    delay_samples = round(exact_samples)
    if not math.isclose(exact_samples, delay_samples, abs_tol=1e-9):
        nearest_us = delay_samples / sample_rate * 1_000_000.0
        raise ValueError(
            f"{delay_us:g} us is {exact_samples:g} samples at {sample_rate} Hz; "
            f"start with a whole-sample delay such as {nearest_us:g} us"
        )
    return delay_samples


def delay_right_channel(mono: np.ndarray, delay_samples: int) -> np.ndarray:
    padding = np.zeros(delay_samples, dtype=mono.dtype)
    left = np.concatenate((mono, padding))
    right = np.concatenate((padding, mono))
    return np.column_stack((left, right))


def number_label(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(abs(int(round(value))))
    return f"{abs(value):g}".replace(".", "p")


def angle_label(angle_degrees: float) -> str:
    number = number_label(angle_degrees)
    if angle_degrees < 0:
        return f"neg{number}deg"
    if angle_degrees > 0:
        return f"p{number}deg"
    return f"{number}deg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create known-regime stereo files at 48 kHz. Positive angles are "
            "left; positive delays mean the right channel arrives later."
        )
    )
    parser.add_argument("input", type=Path, help="input 48 kHz mono audio file")
    parser.add_argument(
        "-a",
        "--angle",
        type=float,
        action="append",
        dest="angles",
        default=[],
        metavar="DEGREES",
        help=(
            "create an amplitude-panned file at this angle; repeat for multiple "
            "angles"
        ),
    )
    parser.add_argument(
        "-d",
        "--delay-us",
        type=float,
        action="append",
        dest="delays_us",
        default=[],
        metavar="MICROSECONDS",
        help=(
            "create a file with this whole-sample right-channel delay; repeat "
            "for multiple delays"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_regime"),
        help=(
            "directory for generated WAV files "
            "(default: tests/generated_regime)"
        ),
    )
    args = parser.parse_args()
    if not args.angles and not args.delays_us:
        parser.error("provide at least one --angle or --delay-us")
    return args


def main() -> None:
    args = parse_args()
    mono, sample_rate, subtype = read_mono(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for angle in args.angles:
        stereo = amplitude_pan(mono, angle)
        destination = args.output_dir / (
            f"{args.input.stem}_regime_amplitude_angle_{angle_label(angle)}.wav"
        )
        sf.write(destination, stereo, sample_rate, subtype=subtype)
        left_gain, right_gain = gains_for_angle(angle)
        print(
            f"Wrote {destination} (regime=amplitude-panned, "
            f"angle={angle:+g}°, left={left_gain:.6f}, right={right_gain:.6f})"
        )

    for delay_us in args.delays_us:
        delay_samples = samples_for_delay(delay_us, sample_rate)
        stereo = delay_right_channel(mono, delay_samples)
        destination = args.output_dir / (
            f"{args.input.stem}_regime_delayed_right_"
            f"{number_label(delay_us)}us.wav"
        )
        sf.write(destination, stereo, sample_rate, subtype=subtype)
        print(
            f"Wrote {destination} (regime=delayed, right delay={delay_us:g} us, "
            f"samples={delay_samples})"
        )


if __name__ == "__main__":
    main()
