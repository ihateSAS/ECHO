#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialize.midside import apply_width  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply mid/side stereo width processing to a stereo WAV file. "
            "width=1 is unchanged, width=0 collapses to mono, width>1 "
            "exaggerates the stereo image."
        )
    )
    parser.add_argument("input", type=Path, help="input stereo WAV file")
    parser.add_argument("--width", type=float, required=True, help="side-channel scale factor")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_midside"),
        help="directory for the output WAV file (default: tests/generated_midside)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, sample_rate = sf.read(args.input, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise SystemExit(f"{args.input} has {audio.shape[1]} channels; expected stereo")

    widened = apply_width(audio, args.width)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    width_label = f"{args.width:g}".replace(".", "p").replace("-", "neg")
    destination = args.output_dir / f"{args.input.stem}_width_{width_label}.wav"
    sf.write(destination, widened, sample_rate)
    print(f"Wrote {destination} (width={args.width:g})")


if __name__ == "__main__":
    main()
