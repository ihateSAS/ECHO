#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SPEAKER_ANGLE_DEGREES = 30.0


def gains_for_angle(angle_degrees: float) -> tuple[float, float]:
    if not -SPEAKER_ANGLE_DEGREES <= angle_degrees <= SPEAKER_ANGLE_DEGREES:
        raise ValueError(
            f"angle must be between {-SPEAKER_ANGLE_DEGREES:g} and "
            f"{SPEAKER_ANGLE_DEGREES:g} degrees"
        )

    angle = math.radians(angle_degrees)
    speaker_angle = math.radians(SPEAKER_ANGLE_DEGREES)
    difference = math.tan(angle) / math.tan(speaker_angle)

    left = (1.0 + difference) / 2.0
    right = (1.0 - difference) / 2.0
    norm = math.hypot(left, right)
    return left / norm, right / norm


def read_mono(path: Path) -> tuple[np.ndarray, int, str]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(
            f"{path} has {audio.shape[1]} channels; the input must be mono"
        )
    return audio[:, 0], sample_rate, info.subtype


def spatialize(mono: np.ndarray, angle_degrees: float) -> np.ndarray:
    left_gain, right_gain = gains_for_angle(angle_degrees)
    return np.column_stack((mono * left_gain, mono * right_gain))


def angle_label(angle_degrees: float) -> str:
    if math.isclose(angle_degrees, round(angle_degrees), abs_tol=1e-9):
        number = str(abs(int(round(angle_degrees))))
    else:
        number = f"{abs(angle_degrees):g}".replace(".", "p")
    if angle_degrees < 0:
        return f"neg{number}deg"
    if angle_degrees > 0:
        return f"p{number}deg"
    return f"{number}deg"


def output_path(input_path: Path, output_dir: Path, angle: float) -> Path:
    return output_dir / f"{input_path.stem}_angle_{angle_label(angle)}.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pan a mono file to one or more known angles. Positive angles are "
            "left and negative angles are right."
        )
    )
    parser.add_argument("input", type=Path, help="input mono audio file")
    parser.add_argument(
        "-a",
        "--angle",
        type=float,
        action="append",
        dest="angles",
        required=True,
        metavar="DEGREES",
        help=(
            "panning angle in degrees, between -30 and +30; "
            "repeat for multiple angles (e.g. -a -25 -a -15 -a 0 -a 15 -a 25)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated"),
        help="directory for generated WAV files (default: tests/generated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mono, sample_rate, subtype = read_mono(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for angle in args.angles:
        stereo = spatialize(mono, angle)
        destination = output_path(args.input, args.output_dir, angle)
        sf.write(destination, stereo, sample_rate, subtype=subtype)
        left_gain, right_gain = gains_for_angle(angle)
        print(
            f"Wrote {destination} "
            f"(angle={angle:+g}°, left={left_gain:.6f}, right={right_gain:.6f})"
        )


if __name__ == "__main__":
    main()
