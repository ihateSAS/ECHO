#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialize.mixture import StemPlacement, mix_stems  # noqa: E402


def read_mono(path: Path, required_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"{path} has {audio.shape[1]} channels; the input must be mono")
    if sample_rate != required_sample_rate:
        raise ValueError(
            f"{path} is {sample_rate} Hz; every stem in one mixture must "
            f"share a sample rate ({required_sample_rate} Hz from the first stem) "
            "-- resample explicitly first rather than mixing rates silently"
        )
    return audio[:, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stereo mixture from several mono stems, each panned to "
            "its own known angle and summed. Prints and optionally writes "
            "the per-stem ground truth alongside the audio."
        )
    )
    parser.add_argument(
        "--stem",
        nargs=2,
        metavar=("PATH", "DEGREES"),
        action="append",
        dest="stems",
        required=True,
        help="a mono WAV file and its azimuth in degrees; repeat for each instrument",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tests/generated_mixture/mixture.wav"),
        help="output WAV path (default: tests/generated_mixture/mixture.wav)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stems:
        raise SystemExit("at least one --stem is required")

    first_path = Path(args.stems[0][0])
    required_sample_rate = sf.info(first_path).samplerate

    placements = [
        StemPlacement(
            mono=read_mono(Path(path_str), required_sample_rate),
            azimuth_deg=float(angle_str),
            label=Path(path_str).stem,
        )
        for path_str, angle_str in args.stems
    ]

    mix, ground_truth = mix_stems(placements)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, mix, required_sample_rate)

    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            [{"label": g.label, "azimuth_deg": g.azimuth_deg} for g in ground_truth],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    peak = float(np.max(np.abs(mix)))
    print(f"Wrote {args.output} ({len(ground_truth)} stems, peak={peak:.3f})")
    print(f"Wrote {manifest_path}")
    for g in ground_truth:
        print(f"  {g.label}: {g.azimuth_deg:+g} degrees")
    if peak > 1.0:
        print(f"warning: mixture peak {peak:.3f} clips; consider lowering input levels")


if __name__ == "__main__":
    main()
