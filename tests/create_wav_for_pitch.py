#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48_000
ANNOTATION_HOP_SECONDS = 0.01
DEFAULT_HARMONICS = 8
DEFAULT_NOTE_SECONDS = 1.0
DEFAULT_GAP_SECONDS = 0.25


EDGE_GUARD_SECONDS = 0.05
FADE_SECONDS = 0.01

DEFAULT_FREQUENCIES_HZ = (65.41, 98.00, 146.83, 220.00)


def harmonic_tone(
    f0_hz: float,
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    harmonics: int = DEFAULT_HARMONICS,
) -> np.ndarray:
    if f0_hz <= 0.0:
        raise ValueError("f0_hz must be positive")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    if harmonics < 1:
        raise ValueError("harmonics must be at least one")

    times = np.arange(round(duration_seconds * sample_rate)) / sample_rate
    tone = np.zeros_like(times)
    for partial in range(1, harmonics + 1):
        frequency = partial * f0_hz
        if frequency >= sample_rate / 2.0:
            break
        tone += np.sin(2.0 * np.pi * frequency * times) / partial
    return tone * (0.5 / max(float(np.max(np.abs(tone))), 1e-12)) * _fade(times.size, sample_rate)


def _fade(num_samples: int, sample_rate: int) -> np.ndarray:
    envelope = np.ones(num_samples)
    fade_samples = min(round(FADE_SECONDS * sample_rate), num_samples // 2)
    if fade_samples < 2:
        return envelope
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    envelope[:fade_samples] = ramp
    envelope[-fade_samples:] = ramp[::-1]
    return envelope


def note_sequence(
    frequencies_hz: tuple[float, ...] | list[float],
    note_seconds: float = DEFAULT_NOTE_SECONDS,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    harmonics: int = DEFAULT_HARMONICS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(frequencies_hz):
        raise ValueError("provide at least one frequency")
    if note_seconds <= 0.0 or gap_seconds < 0.0:
        raise ValueError("note_seconds must be positive and gap_seconds non-negative")

    note_samples = round(note_seconds * sample_rate)
    gap_samples = round(gap_seconds * sample_rate)
    pieces: list[np.ndarray] = []
    spans: list[tuple[float, float, float]] = []
    for f0_hz in frequencies_hz:
        start = sum(piece.size for piece in pieces) / sample_rate
        pieces.append(harmonic_tone(f0_hz, note_samples / sample_rate, sample_rate, harmonics))
        spans.append((start, start + note_samples / sample_rate, f0_hz))
        if gap_samples:
            pieces.append(np.zeros(gap_samples))

    audio = np.concatenate(pieces)
    times = np.arange(math.floor(audio.size / sample_rate / ANNOTATION_HOP_SECONDS))
    times = times * ANNOTATION_HOP_SECONDS
    f0 = np.zeros_like(times)
    for start, end, f0_hz in spans:
        voiced = (times >= start + EDGE_GUARD_SECONDS) & (
            times <= end - EDGE_GUARD_SECONDS
        )
        f0[voiced] = f0_hz
    return audio, times, f0


def glide(
    start_hz: float,
    end_hz: float,
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    harmonics: int = DEFAULT_HARMONICS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if start_hz <= 0.0 or end_hz <= 0.0:
        raise ValueError("glide frequencies must be positive")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")

    times = np.arange(round(duration_seconds * sample_rate)) / sample_rate
    ratio = end_hz / start_hz
    instantaneous = start_hz * ratio ** (times / duration_seconds)
    phase = np.cumsum(instantaneous) * (2.0 * np.pi / sample_rate)

    audio = np.zeros_like(times)
    for partial in range(1, harmonics + 1):
        if partial * end_hz >= sample_rate / 2.0 or partial * start_hz >= sample_rate / 2.0:
            break
        audio += np.sin(partial * phase) / partial
    audio *= (0.5 / max(float(np.max(np.abs(audio))), 1e-12)) * _fade(times.size, sample_rate)

    annotation_times = (
        np.arange(math.floor(duration_seconds / ANNOTATION_HOP_SECONDS))
        * ANNOTATION_HOP_SECONDS
    )
    annotation_f0 = start_hz * ratio ** (annotation_times / duration_seconds)
    guard = (annotation_times >= EDGE_GUARD_SECONDS) & (
        annotation_times <= duration_seconds - EDGE_GUARD_SECONDS
    )
    return audio, annotation_times, np.where(guard, annotation_f0, 0.0)


def write_f0_annotation(
    path: Path, times_seconds: np.ndarray, f0_hz: np.ndarray
) -> None:
    if times_seconds.shape != f0_hz.shape:
        raise ValueError("times and f0 arrays must have the same shape")
    lines = (f"{t:.3f}\t{f:.3f}" for t, f in zip(times_seconds, f0_hz))
    path.write_text("\n".join(lines) + "\n")


def read_f0_annotation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{path} is not a two-column time/f0 annotation")
    return values[:, 0].copy(), values[:, 1].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render mono audio at known fundamental frequencies, with a "
            "matching URMP-format F0 annotation next to it."
        )
    )
    parser.add_argument(
        "-f",
        "--frequency",
        type=float,
        action="append",
        dest="frequencies",
        default=[],
        metavar="HZ",
        help=(
            "held-note frequency; repeat for a sequence (default: "
            + " ".join(f"{f:g}" for f in DEFAULT_FREQUENCIES_HZ)
            + ")"
        ),
    )
    parser.add_argument(
        "--glide",
        type=float,
        nargs=2,
        default=None,
        metavar=("START_HZ", "END_HZ"),
        help="also render a continuous glissando between these frequencies",
    )
    parser.add_argument(
        "--note-seconds",
        type=float,
        default=DEFAULT_NOTE_SECONDS,
        help=f"length of each held note (default: {DEFAULT_NOTE_SECONDS:g})",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=DEFAULT_GAP_SECONDS,
        help=f"silence between notes (default: {DEFAULT_GAP_SECONDS:g})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests/generated_pitch"),
        help="directory for generated files (default: tests/generated_pitch)",
    )
    return parser.parse_args()


def _write(output_dir: Path, name: str, audio, times, f0) -> None:
    audio_path = output_dir / f"{name}.wav"
    annotation_path = output_dir / f"F0s_{name}.txt"
    sf.write(audio_path, audio, SAMPLE_RATE, subtype="PCM_24")
    write_f0_annotation(annotation_path, times, f0)
    voiced = int(np.count_nonzero(f0 > 0.0))
    print(
        f"Wrote {audio_path} and {annotation_path} "
        f"({audio.size / SAMPLE_RATE:.2f} s, {voiced}/{f0.size} voiced frames)"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frequencies = args.frequencies or list(DEFAULT_FREQUENCIES_HZ)

    audio, times, f0 = note_sequence(
        frequencies, note_seconds=args.note_seconds, gap_seconds=args.gap_seconds
    )
    name = "pitch_notes_" + "_".join(f"{round(f)}" for f in frequencies) + "hz"
    _write(args.output_dir, name, audio, times, f0)

    if args.glide is not None:
        start_hz, end_hz = args.glide
        total = args.note_seconds * max(len(frequencies), 1)
        audio, times, f0 = glide(start_hz, end_hz, total)
        name = f"pitch_glide_{round(start_hz)}_to_{round(end_hz)}hz"
        _write(args.output_dir, name, audio, times, f0)


if __name__ == "__main__":
    main()
