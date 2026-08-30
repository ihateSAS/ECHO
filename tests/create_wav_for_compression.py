#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialize.compression import stereo_linked_compressor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply stereo-linked dynamic range compression (shared peak "
            "detector, so the L/R level ratio is preserved) to a stereo "
            "WAV file."
        )
    )
    parser.add_argument("input", type=Path, help="input stereo WAV file")
    parser.add_argument(
        "--threshold", type=float, required=True, metavar="DB", help="threshold in dBFS"
    )
    parser.add_argument(
        "--ratio", type=float, required=True, metavar="N:1", help="compression ratio (>= 1)"
    )
    parser.add_argument(
        "--attack", type=float, default=0.005, metavar="SECONDS", help="attack time (default: 5ms)"
    )
    parser.add_argument(
        "--release", type=float, default=0.10, metavar="SECONDS", help="release time (default: 100ms)"
    )
    parser.add_argument(
        "--makeup", type=float, default=0.0, metavar="DB", help="makeup gain in dB (default: 0)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_compression"),
        help="directory for the output WAV file (default: tests/generated_compression)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, sample_rate = sf.read(args.input, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise SystemExit(f"{args.input} has {audio.shape[1]} channels; expected stereo")

    compressed = stereo_linked_compressor(
        audio,
        sample_rate,
        threshold_db=args.threshold,
        ratio=args.ratio,
        attack_s=args.attack,
        release_s=args.release,
        makeup_gain_db=args.makeup,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_label = f"{args.threshold:g}".replace(".", "p").replace("-", "neg")
    ratio_label = f"{args.ratio:g}".replace(".", "p")
    destination = args.output_dir / (
        f"{args.input.stem}_comp_thresh_{threshold_label}dB_ratio_{ratio_label}.wav"
    )
    sf.write(destination, compressed, sample_rate)
    print(
        f"Wrote {destination} "
        f"(threshold={args.threshold:g}dB, ratio={args.ratio:g}:1, "
        f"attack={args.attack*1000:g}ms, release={args.release*1000:g}ms, "
        f"makeup={args.makeup:g}dB)"
    )


if __name__ == "__main__":
    main()
