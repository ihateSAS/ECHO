#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.config import BenchmarkConfig, load_config  # noqa: E402
from benchmark.provenance import build_id  # noqa: E402
from eval.generalization import distinct_answer_angles  # noqa: E402


def widest_gap_deg(angles: tuple[float, ...]) -> float:
    inside = sorted(angles)
    if len(inside) < 2:
        return float("inf")
    return max(high - low for low, high in zip(inside, inside[1:])) / 2.0


def report(path: Path, config: BenchmarkConfig) -> list[str]:
    answers = distinct_answer_angles(config)
    recipes = 1 + len(config.processing_recipes)
    per_stem = len(config.angles_degrees) * recipes
    gap = widest_gap_deg(answers)
    tolerance = config.azimuth_tolerance_degrees

    print(f"{path.name}")
    print(f"  build id                 {build_id(config)}")
    print(f"  rendering                {config.rendering} (+/-{config.maximum_abs_angle_deg:g} deg)")
    print(f"  angles x recipes         {len(config.angles_degrees)} x {recipes} = {per_stem} items per stem")
    print(f"  distinct questions       {len(answers)}")
    print(f"  widest gap between them  {gap:.2f} deg")
    print(f"  azimuth tolerance        {tolerance:g} deg")

    warnings: list[str] = []
    if gap <= tolerance:
        warnings.append(
            f"the widest gap between answers ({gap:.2f} deg) is within the "
            f"{tolerance:g} deg tolerance, so a model that answers with the "
            "nearest angle it saw in training scores 100% within tolerance "
            "on every item. Report the share of answers landing on a trained "
            "angle instead, or sample angles continuously "
            "(see eval/generalization.py)."
        )
    if len(answers) < 32:
        warnings.append(
            f"only {len(answers)} distinct questions. Every stem at the same "
            "angle presents the identical cue, so item count does not add "
            "difficulty -- a lookup table of this size solves the benchmark."
        )
    for warning in warnings:
        print(f"  WARNING: {warning}")
    if not warnings:
        print("  no warnings")
    print()
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any config raises a warning (for CI)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    for path in args.configs:
        total += len(report(path, load_config(path)))
    if total and args.strict:
        raise SystemExit(f"{total} warning(s)")


if __name__ == "__main__":
    main()
