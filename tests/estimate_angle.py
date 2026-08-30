#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

SPEAKER_ANGLE_DEGREES = 30.0
ACTIVE_LEVEL_DB = -60.0


def channel_spectrum(
    channel: np.ndarray, sample_rate: int, nperseg: int
) -> np.ndarray:
    _, _, spectrum = stft(
        channel,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        boundary=None,
        padded=False,
    )
    return np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=1))


def estimate_angle(path: Path) -> tuple[float, float, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected stereo")
    if audio.shape[0] < 16:
        raise ValueError(f"{path} is too short to analyze")

    peak = float(np.max(np.abs(audio)))
    if peak <= np.finfo(np.float64).eps:
        raise ValueError(f"{path} contains only silence")

    nperseg = min(4096, audio.shape[0])
    left = channel_spectrum(audio[:, 0], sample_rate, nperseg)
    right = channel_spectrum(audio[:, 1], sample_rate, nperseg)
    combined = np.hypot(left, right)


    threshold = np.max(combined) * 10.0 ** (ACTIVE_LEVEL_DB / 20.0)
    floor = np.finfo(np.float64).tiny
    active = (combined >= threshold) & (left > floor) & (right > floor)
    if not np.any(active):
        raise ValueError(f"{path} has no usable frequency bins")

    level_differences = 20.0 * np.log10(left[active] / right[active])
    median_db = float(np.median(level_differences))

    ratio = 10.0 ** (median_db / 20.0)
    difference = (ratio - 1.0) / (ratio + 1.0)
    angle = math.degrees(
        math.atan(difference * math.tan(math.radians(SPEAKER_ANGLE_DEGREES)))
    )
    return angle, median_db, int(np.count_nonzero(active))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover tangent-law panning angles from stereo WAV files. "
            "Positive results are left and negative results are right."
        )
    )
    parser.add_argument("files", type=Path, nargs="+", help="stereo files to analyze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for path in args.files:
        try:
            angle, median_db, bins = estimate_angle(path)
            print(
                f"{path}: angle={angle:+.2f}°, "
                f"median L/R={median_db:+.3f} dB, bins={bins}"
            )
        except (OSError, ValueError) as error:
            failed = True
            print(f"{path}: error: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
