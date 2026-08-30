#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialize.delay_widening import samples_for_delay, widen_with_delay  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply delay-based stereo widening (right channel delayed "
            "relative to left) to a stereo WAV file."
        )
    )
    parser.add_argument("input", type=Path, help="input stereo WAV file")
    parser.add_argument(
        "--delay-us",
        type=float,
        required=True,
        metavar="MICROSECONDS",
        help="right-channel delay in microseconds",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_delay_widening"),
        help="directory for the output WAV file (default: tests/generated_delay_widening)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, sample_rate = sf.read(args.input, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise SystemExit(f"{args.input} has {audio.shape[1]} channels; expected stereo")

    widened = widen_with_delay(audio, sample_rate, args.delay_us)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    delay_label = f"{args.delay_us:g}".replace(".", "p")
    destination = args.output_dir / f"{args.input.stem}_delaywiden_{delay_label}us.wav"
    sf.write(destination, widened, sample_rate)
    delay_samples = samples_for_delay(args.delay_us, sample_rate)
    print(f"Wrote {destination} (delay={args.delay_us:g}us, samples={delay_samples})")


if __name__ == "__main__":
    main()
