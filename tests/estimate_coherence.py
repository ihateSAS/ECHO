#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cues.coherence import (  # noqa: E402
    MAX_BAND_HZ,
    MIN_BAND_HZ,
    CoherenceEstimate,
    estimate_coherence,
)


def estimate_coherence_file(path: Path) -> CoherenceEstimate:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected stereo")
    if float(np.max(np.abs(audio))) <= np.finfo(np.float64).eps:
        raise ValueError(f"{path} contains only silence")
    return estimate_coherence(audio, sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Report median interchannel coherence (gamma^2, {MIN_BAND_HZ:g} Hz-"
            f"{MAX_BAND_HZ / 1000:g} kHz) for stereo WAV files. Close to 1 means a "
            "compact point source; lower values indicate decorrelation such as a "
            "reverb tail, a widened image, or genuinely different content per "
            "channel."
        )
    )
    parser.add_argument("files", type=Path, nargs="+", help="stereo files to analyze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for path in args.files:
        try:
            result = estimate_coherence_file(path)
            print(
                f"{path}: coherence={result.value:.4f}, "
                f"bins={result.active_bins}, "
                f"segments={result.segments}x{result.segment_length}"
            )
        except (OSError, ValueError) as error:
            failed = True
            print(f"{path}: error: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
