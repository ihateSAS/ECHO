#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialize.reverb import schroeder_reverb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a Schroeder/Freeverb-style stereo reverb with a known RT60 "
            "to a stereo WAV file, for building realistic-but-ground-truthed "
            "training audio."
        )
    )
    parser.add_argument("input", type=Path, help="input stereo WAV file")
    parser.add_argument(
        "--rt60",
        type=float,
        required=True,
        metavar="SECONDS",
        help="target time for the reverb tail to decay 60 dB",
    )
    parser.add_argument(
        "--wet",
        type=float,
        required=True,
        metavar="[0-1]",
        help="wet/dry mix; 0 returns the dry signal unchanged, 1 is fully wet",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.25,
        metavar="[0-1)",
        help="high-frequency damping in the reverb tail (default: 0.25)",
    )
    parser.add_argument(
        "--diffusion",
        type=float,
        default=0.6,
        metavar="[0-1]",
        help=(
            "how much of the tail is independent per-channel noise rather "
            "than a shared, correlated tail; this is what reduces "
            "interchannel coherence (default: 0.6)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the diffusion noise, for reproducibility (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_reverb"),
        help="directory for the output WAV file (default: tests/generated_reverb)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, sample_rate = sf.read(args.input, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise SystemExit(f"{args.input} has {audio.shape[1]} channels; expected stereo")

    wet_audio = schroeder_reverb(
        audio,
        sample_rate,
        rt60_s=args.rt60,
        wet=args.wet,
        damping=args.damping,
        diffusion=args.diffusion,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rt60_label = f"{args.rt60:g}".replace(".", "p")
    wet_label = f"{args.wet:g}".replace(".", "p")
    destination = args.output_dir / (
        f"{args.input.stem}_reverb_rt60_{rt60_label}s_wet_{wet_label}.wav"
    )
    sf.write(destination, wet_audio, sample_rate)
    print(
        f"Wrote {destination} "
        f"(rt60={args.rt60:g}s, wet={args.wet:g}, damping={args.damping:g}, "
        f"diffusion={args.diffusion:g}, seed={args.seed}, "
        f"duration={wet_audio.shape[0] / sample_rate:.2f}s)"
    )


if __name__ == "__main__":
    main()
