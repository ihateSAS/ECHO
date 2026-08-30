#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialize.eq import apply_peaking_eq  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a parametric peaking EQ, identically to both channels, "
            "to a stereo WAV file."
        )
    )
    parser.add_argument("input", type=Path, help="input stereo WAV file")
    parser.add_argument(
        "--freq", type=float, required=True, metavar="HZ", help="center frequency in Hz"
    )
    parser.add_argument(
        "--gain", type=float, required=True, metavar="DB", help="boost (+) or cut (-) in dB"
    )
    parser.add_argument("--q", type=float, default=1.0, help="filter Q (default: 1.0)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_eq"),
        help="directory for the output WAV file (default: tests/generated_eq)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, sample_rate = sf.read(args.input, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise SystemExit(f"{args.input} has {audio.shape[1]} channels; expected stereo")

    equalized = apply_peaking_eq(audio, sample_rate, args.freq, args.gain, args.q)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_label = f"{args.freq:g}"
    gain_label = f"{args.gain:g}".replace(".", "p").replace("-", "neg")
    destination = args.output_dir / f"{args.input.stem}_eq_{freq_label}Hz_{gain_label}dB.wav"
    sf.write(destination, equalized, sample_rate)
    print(f"Wrote {destination} (freq={args.freq:g}Hz, gain={args.gain:g}dB, Q={args.q:g})")


if __name__ == "__main__":
    main()
