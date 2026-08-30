#!/usr/bin/env python3


from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from create_wav_for_pitch import read_f0_annotation  # noqa: E402
from cues.pitch import (  # noqa: E402
    F0_TOLERANCE_PERCENT,
    INSTRUMENT_RANGE_HZ,
    PitchTrack,
    compare_to_reference,
    instrument_range_hz,
    track_pitch,
)


URMP_STEM_PATTERN = re.compile(r"^AuSep_\d+_([a-z]+)_", re.IGNORECASE)


def instrument_code_from_name(path: Path) -> str | None:
    match = URMP_STEM_PATTERN.match(path.name)
    if match is None:
        return None
    code = match.group(1).lower()
    return code if code in INSTRUMENT_RANGE_HZ else None


def default_reference_path(audio_path: Path) -> Path | None:
    for candidate in (
        audio_path.with_name(f"F0s_{audio_path.stem}.txt"),
        audio_path.with_name(audio_path.name.replace("AuSep_", "F0s_")).with_suffix(
            ".txt"
        ),
    ):
        if candidate.is_file():
            return candidate
    return None


def read_mono_any_rate(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(
            f"{path} has {audio.shape[1]} channels; pitch tracking wants an "
            "isolated mono stem"
        )
    return audio[:, 0], sample_rate


def track_file(path: Path, instrument_code: str | None) -> tuple[PitchTrack, str]:
    code = instrument_code or instrument_code_from_name(path)
    if code is None:
        raise ValueError(
            f"cannot tell which instrument {path.name} is; pass --instrument "
            f"(one of: {', '.join(sorted(INSTRUMENT_RANGE_HZ))})"
        )
    fmin_hz, fmax_hz = instrument_range_hz(code)
    mono, sample_rate = read_mono_any_rate(path)
    return track_pitch(mono, sample_rate, fmin_hz, fmax_hz), code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track fundamental frequency with pYIN and, when a URMP-format "
            "F0s_*.txt annotation is available, report how close the track "
            "lands to it."
        )
    )
    parser.add_argument("files", type=Path, nargs="+", help="mono stems to track")
    parser.add_argument(
        "-i",
        "--instrument",
        default=None,
        metavar="CODE",
        help=(
            "URMP instrument code fixing the search range; inferred from an "
            "AuSep_* filename when omitted"
        ),
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        default=None,
        help="F0s_*.txt annotation to score against (default: look beside the audio)",
    )
    parser.add_argument(
        "-t",
        "--tolerance-percent",
        type=float,
        default=F0_TOLERANCE_PERCENT,
        help=f"f0 tolerance in percent (default: {F0_TOLERANCE_PERCENT:g})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for path in args.files:
        try:
            track, code = track_file(path, args.instrument)
            voiced = int(np.count_nonzero(track.voiced))
            median_hz = (
                float(np.median(track.f0_hz[track.voiced])) if voiced else float("nan")
            )
            print(
                f"{path}: instrument={code}, frames={track.f0_hz.size}, "
                f"voiced={voiced}, median f0={median_hz:.2f} Hz, "
                f"range={track.fmin_hz:g}-{track.fmax_hz:g} Hz"
            )

            reference_path = args.reference or default_reference_path(path)
            if reference_path is None:
                print(f"{path}: no F0s_*.txt annotation found, not scored")
                continue

            times, f0 = read_f0_annotation(reference_path)
            result = compare_to_reference(track, times, f0, args.tolerance_percent)
            print(
                f"{path}: vs {reference_path.name}: "
                f"median error={result.median_abs_percent_error:.2f}%, "
                f"mean={result.mean_abs_percent_error:.2f}%, "
                f"within {result.tolerance_percent:g}%="
                f"{100.0 * result.within_tolerance:.1f}% of "
                f"{result.frames_compared} frames, "
                f"voicing agreement={100.0 * result.voicing_agreement:.1f}%"
            )
        except (OSError, ValueError) as error:
            failed = True
            print(f"{path}: error: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
