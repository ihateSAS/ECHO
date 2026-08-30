#!/usr/bin/env python3


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from create_wav_for_regime import amplitude_pan, angle_label, number_label, read_mono

DEFAULT_SEED = 2026
DEFAULT_ANGLE_DEGREES = 15.0


def decorrelated_noise(
    source: np.ndarray, ratio: float, rng: np.random.Generator
) -> np.ndarray:
    if ratio < 0.0:
        raise ValueError("ratio must be non-negative")
    if source.ndim != 1 or source.size < 2:
        raise ValueError("source must be a one-dimensional signal")
    if ratio == 0.0:
        return np.zeros_like(source)

    spectrum = np.fft.rfft(source)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.size)


    phases[0] = 0.0
    if source.size % 2 == 0:
        phases[-1] = 0.0
    noise = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=source.size)

    source_power = float(np.mean(source**2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= np.finfo(np.float64).tiny:
        raise ValueError("source has no energy to build noise from")
    return noise * np.sqrt(ratio * source_power / noise_power)


def predicted_coherence(ratio: float) -> float:
    if ratio < 0.0:
        raise ValueError("ratio must be non-negative")
    return 1.0 / (1.0 + ratio) ** 2


def panned_with_noise(
    mono: np.ndarray, angle_degrees: float, ratio: float, seed: int
) -> np.ndarray:
    stereo = amplitude_pan(mono, angle_degrees)
    rng = np.random.default_rng(seed)
    left = stereo[:, 0] + decorrelated_noise(stereo[:, 0], ratio, rng)
    right = stereo[:, 1] + decorrelated_noise(stereo[:, 1], ratio, rng)
    return np.column_stack((left, right))


def unrelated_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    length = min(first.size, second.size)
    left, right = first[:length].copy(), second[:length].copy()
    left_rms = float(np.sqrt(np.mean(left**2)))
    right_rms = float(np.sqrt(np.mean(right**2)))
    if min(left_rms, right_rms) <= np.finfo(np.float64).tiny:
        raise ValueError("both recordings must contain audio")
    return np.column_stack((left, right * (left_rms / right_rms)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create stereo files with a known interchannel coherence: a "
            "clean amplitude pan (coherence 1), the same pan with a known "
            "amount of independent per-channel noise (coherence "
            "1/(1+ratio)^2), and optionally two unrelated recordings "
            "(coherence 0)."
        )
    )
    parser.add_argument("input", type=Path, help="input 48 kHz mono audio file")
    parser.add_argument(
        "-a",
        "--angle",
        type=float,
        default=DEFAULT_ANGLE_DEGREES,
        metavar="DEGREES",
        help=(
            "pan angle for the coherent files, positive is left "
            f"(default: {DEFAULT_ANGLE_DEGREES:g})"
        ),
    )
    parser.add_argument(
        "-n",
        "--noise-ratio",
        type=float,
        action="append",
        dest="noise_ratios",
        default=[],
        metavar="RATIO",
        help=(
            "noise-to-signal power ratio for one noisy file; repeat to sweep "
            "the knob (e.g. -n 0.1 -n 0.5 -n 2)"
        ),
    )
    parser.add_argument(
        "-u",
        "--unrelated",
        type=Path,
        default=None,
        metavar="OTHER.wav",
        help="second 48 kHz mono file to put in the right channel on its own",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for the noise streams (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_coherence"),
        help="directory for generated WAV files (default: tests/generated_coherence)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mono, sample_rate, subtype = read_mono(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    label = angle_label(args.angle)

    clean = amplitude_pan(mono, args.angle)
    destination = args.output_dir / f"{stem}_coherence_panned_{label}.wav"
    sf.write(destination, clean, sample_rate, subtype=subtype)
    print(f"Wrote {destination} (panned {args.angle:+g}°, expected coherence 1.0000)")

    for ratio in args.noise_ratios:
        noisy = panned_with_noise(mono, args.angle, ratio, args.seed)
        destination = args.output_dir / (
            f"{stem}_coherence_noise_{number_label(ratio)}_{label}.wav"
        )
        sf.write(destination, noisy, sample_rate, subtype=subtype)
        print(
            f"Wrote {destination} (panned {args.angle:+g}° + noise ratio "
            f"{ratio:g}, seed {args.seed}, expected coherence "
            f"{predicted_coherence(ratio):.4f})"
        )

    if args.unrelated is not None:
        other, other_rate, _ = read_mono(args.unrelated)
        if other_rate != sample_rate:
            raise ValueError("both recordings must share a sample rate")
        stereo = unrelated_pair(mono, other)
        destination = args.output_dir / (
            f"{stem}_coherence_unrelated_{args.unrelated.stem}.wav"
        )
        sf.write(destination, stereo, sample_rate, subtype=subtype)
        print(f"Wrote {destination} (unrelated pair, expected coherence 0.0000)")


if __name__ == "__main__":
    main()
