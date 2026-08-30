#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.real_stereo import (  # noqa: E402
    AZIMUTH_TOLERANCE_DEG,
    CONTROL_CHAINS,
    DEFAULT_CONTROL_ANGLES_DEG,
    DEFAULT_WINDOW_SECONDS,
    TrackSurvey,
    survey_track,
)

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")


def find_audio(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def parse_angles(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def survey_to_dict(survey: TrackSurvey) -> dict[str, object]:
    ild_mean, ild_std, ild_range = survey.spread("ild_db")
    azimuth_mean, azimuth_std, azimuth_range = survey.spread("estimated_azimuth_deg")
    coherence_mean, coherence_std, coherence_range = survey.spread("coherence")
    itd_mean, itd_std, itd_range = survey.spread("itd_us")
    flatness_mean, flatness_std, _ = survey.spread("ild_flatness_db")
    return {
        "name": survey.name,
        "sample_rate": survey.sample_rate,
        "duration_seconds": round(survey.duration_seconds, 2),
        "channels": survey.channels,
        "side_to_total_db": round(survey.side_to_total_db, 2),
        "windows_measured": len(survey.windows),
        "summary": {
            "ild_db": {
                "mean": ild_mean, "std": ild_std, "peak_to_peak": ild_range
            },
            "azimuth_deg": {
                "mean": azimuth_mean, "std": azimuth_std, "peak_to_peak": azimuth_range
            },
            "coherence": {
                "mean": coherence_mean,
                "std": coherence_std,
                "peak_to_peak": coherence_range,
            },
            "itd_us": {
                "mean": itd_mean, "std": itd_std, "peak_to_peak": itd_range
            },
            "ild_flatness_db": {"mean": flatness_mean, "std": flatness_std},
            "regimes": survey.regimes,
            "mix_f0_hz": survey.mix_f0_hz,
            "mix_voiced_fraction": survey.mix_voiced_fraction,
            "worst_control_error_deg": survey.worst_control_error_deg,
            "worst_control_error_deg_by_chain": survey.worst_control_error_by_chain(),
        },
        "windows": [
            {
                "start_seconds": window.start_seconds,
                "rms_dbfs": window.rms_dbfs,
                "evidence": asdict(window.evidence),
            }
            for window in survey.windows
        ],
        "controls": [
            {**asdict(control), "azimuth_error_deg": control.azimuth_error_deg}
            for control in survey.controls
        ],
    }


def print_track(survey: TrackSurvey) -> None:
    ild_mean, ild_std, _ = survey.spread("ild_db")
    azimuth_mean, azimuth_std, azimuth_range = survey.spread("estimated_azimuth_deg")
    coherence_mean, coherence_std, _ = survey.spread("coherence")
    itd_mean, itd_std, _ = survey.spread("itd_us")
    regimes = ", ".join(f"{label} x{count}" for label, count in survey.regimes.items())
    by_chain = survey.worst_control_error_by_chain()
    print(
        f"\n{survey.name}  ({survey.duration_seconds:.0f} s, "
        f"{survey.sample_rate} Hz, side/total {survey.side_to_total_db:+.1f} dB, "
        f"{len(survey.windows)} windows)"
    )
    print(f"  ILD          {ild_mean:+7.2f} dB  +- {ild_std:.2f}")
    print(
        f"  azimuth      {azimuth_mean:+7.2f} deg +- {azimuth_std:.2f} "
        f"(spread {azimuth_range:.2f})"
    )
    print(f"  coherence    {coherence_mean:7.3f}     +- {coherence_std:.3f}")
    print(f"  ITD          {itd_mean:+7.1f} us  +- {itd_std:.1f}")
    print(f"  regime       {regimes}")
    print(
        f"  mix f0       {survey.mix_f0_hz:.1f} Hz over "
        f"{survey.mix_voiced_fraction * 100:.0f}% voiced frames"
    )
    if by_chain:
        print(f"  control      worst planted-angle error, tolerance {AZIMUTH_TOLERANCE_DEG:g} deg")
        for name, error in by_chain.items():
            verdict = "ok" if error <= AZIMUTH_TOLERANCE_DEG else "OVER TOLERANCE"
            print(f"    {name:<12} {error:6.2f} deg  {verdict}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",
        default=Path("data/real_stereo"),
        help="stereo file, or a directory of them (default: data/real_stereo)",
    )
    parser.add_argument(
        "-w",
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"analysis window length (default: {DEFAULT_WINDOW_SECONDS:g})",
    )
    parser.add_argument(
        "-a",
        "--control-angles",
        default=",".join(f"{angle:g}" for angle in DEFAULT_CONTROL_ANGLES_DEG),
        help="comma-separated planted angles for the control",
    )
    parser.add_argument(
        "--control-windows",
        type=int,
        default=1,
        help="how many windows per track also get the planted-angle control",
    )
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    parser.add_argument("--limit", type=int, default=None, help="stop after N files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = find_audio(args.audio)
    if not paths:
        raise SystemExit(f"no audio found under {args.audio}")
    if args.limit is not None:
        paths = paths[: args.limit]

    surveys: list[TrackSurvey] = []
    for path in paths:
        try:
            survey = survey_track(
                path,
                window_seconds=args.window_seconds,
                control_angles_deg=parse_angles(args.control_angles),
                control_windows=args.control_windows,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"\n{path.name}  SKIPPED: {exc}")
            continue
        surveys.append(survey)
        print_track(survey)

    if not surveys:
        raise SystemExit("nothing could be surveyed")

    azimuth_spreads = [survey.spread("estimated_azimuth_deg")[2] for survey in surveys]
    coherences = [survey.spread("coherence")[0] for survey in surveys]
    print(f"\n{len(surveys)} tracks.")
    print("  worst planted-angle error over every track and angle, by chain:")
    for name in CONTROL_CHAINS:
        errors = [
            survey.worst_control_error_by_chain().get(name)
            for survey in surveys
            if name in survey.worst_control_error_by_chain()
        ]
        finite = [error for error in errors if error is not None and math.isfinite(error)]
        if not finite:
            continue
        verdict = "ok" if max(finite) <= AZIMUTH_TOLERANCE_DEG else "OVER TOLERANCE"
        print(f"    {name:<12} {max(finite):6.2f} deg  {verdict}")
    print(
        f"  widest within-track azimuth spread: {max(azimuth_spreads):.2f} deg. "
        f"Mix coherence {min(coherences):.3f} to {max(coherences):.3f}."
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "window_seconds": args.window_seconds,
                    "control_angles_deg": list(parse_angles(args.control_angles)),
                    "azimuth_tolerance_deg": AZIMUTH_TOLERANCE_DEG,
                    "tracks": [survey_to_dict(survey) for survey in surveys],
                },
                indent=1,
            )
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
