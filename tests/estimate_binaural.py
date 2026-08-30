#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cues.binaural import invert_woodworth  # noqa: E402
from estimate_regime import gcc_phat_delay_us, highest_energy_segment, read_stereo  # noqa: E402


@dataclass(frozen=True)
class BinauralEstimate:
    delay_us: float
    angle_degrees: float


def estimate_binaural_angle(path: Path) -> BinauralEstimate:
    audio, sample_rate = read_stereo(path)
    segment = highest_energy_segment(audio)
    delay_us = gcc_phat_delay_us(segment[:, 0], segment[:, 1], sample_rate)
    return BinauralEstimate(delay_us=delay_us, angle_degrees=invert_woodworth(delay_us))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover Woodworth binaural angles from stereo WAV files via "
            "GCC-PHAT delay estimation followed by numerical inversion."
        )
    )
    parser.add_argument("files", type=Path, nargs="+", help="stereo files to analyze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for path in args.files:
        try:
            result = estimate_binaural_angle(path)
            print(
                f"{path}: delay={result.delay_us:+.2f} us, "
                f"angle={result.angle_degrees:+.2f}°"
            )
        except (OSError, ValueError) as error:
            failed = True
            print(f"{path}: error: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
