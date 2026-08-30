#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialize.binaural import binaural_render, woodworth_itd_us  # noqa: E402

REQUIRED_SAMPLE_RATE = 48_000


def read_mono(path: Path) -> tuple[np.ndarray, int, str]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(
            f"{path} has {audio.shape[1]} channels; the input must be mono"
        )
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sample_rate} Hz; binaural delay tests require "
            f"{REQUIRED_SAMPLE_RATE} Hz"
        )
    return audio[:, 0], sample_rate, info.subtype


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
            "Render known-angle binaural (Woodworth-delayed, head-shadowed) "
            "stereo files at 48 kHz. Positive angles are left."
        )
    )
    parser.add_argument("input", type=Path, help="input 48 kHz mono audio file")
    parser.add_argument(
        "-a",
        "--angle",
        type=float,
        action="append",
        dest="angles",
        required=True,
        metavar="DEGREES",
        help=(
            "binaural angle in degrees, between -90 and +90; repeat for "
            "multiple angles (e.g. -a -60 -a -30 -a -10 -a 10 -a 30 -a 60 -a 90)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_binaural"),
        help="directory for generated WAV files (default: tests/generated_binaural)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mono, sample_rate, subtype = read_mono(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for angle in args.angles:
        stereo = binaural_render(mono, sample_rate, angle)
        destination = args.output_dir / (
            f"{args.input.stem}_binaural_angle_{angle_label(angle)}.wav"
        )
        sf.write(destination, stereo, sample_rate, subtype=subtype)
        delay_us = woodworth_itd_us(angle)
        print(
            f"Wrote {destination} "
            f"(angle={angle:+g}°, ITD={delay_us:+.2f} us)"
        )


if __name__ == "__main__":
    main()
