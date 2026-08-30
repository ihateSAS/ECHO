#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.provenance import check_build_ids  # noqa: E402
from eval.score import (  # noqa: E402
    AZIMUTH_TOLERANCE_DEG,
    ScoreReport,
    constant_baseline,
    front_end_baseline,
    random_baseline,
    read_answers,
    read_build_id,
    read_labels,
    refusal_baseline,
    score_answers,
)

DEFAULT_LABELS = Path(
    "artifacts/stereomusicqa_v0.1/test_private_labels.jsonl"
)
DEFAULT_ANGLES = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def parse_angles(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help=f"private label manifest (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--answers", type=Path, default=None, help="model answers, one JSON per line"
    )
    parser.add_argument(
        "--name", default="model", help="what to call the answers in the table"
    )
    parser.add_argument(
        "--baselines", action="store_true", help="also score the reference baselines"
    )
    parser.add_argument(
        "--angles",
        default=",".join(f"{angle:g}" for angle in DEFAULT_ANGLES),
        help="angle grid the random baseline draws from",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    parser.add_argument(
        "--allow-build-mismatch",
        action="store_true",
        help=(
            "score answers against labels from a different benchmark build. "
            "Off by default because every build numbers its held-out items "
            "from the same sequence, so a mismatched pair silently compares "
            "unrelated items and still prints a number."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.labels.is_file():
        raise SystemExit(
            f"no labels at {args.labels}. Build the benchmark first:\n"
            "  python scripts/build_benchmark.py "
            "--config benchmark/configs/stereomusicqa_v0.1.json"
        )
    labels = read_labels(args.labels)
    if not labels:
        raise SystemExit(f"{args.labels} has no items")

    reports: list[ScoreReport] = []
    if args.answers is not None:
        if not args.allow_build_mismatch:
            check_build_ids(
                read_build_id(args.labels),
                read_build_id(args.answers),
                labels_name=str(args.labels),
                answers_name=str(args.answers),
            )
        reports.append(
            score_answers(labels, read_answers(args.answers), name=args.name)
        )
    if args.baselines or args.answers is None:
        reports.extend(
            [
                refusal_baseline(labels),
                constant_baseline(labels, 0.0),
                random_baseline(labels, parse_angles(args.angles), seed=args.seed),
                front_end_baseline(labels),
            ]
        )

    print(f"{len(labels)} items from {args.labels}\n")
    for report in reports:
        print(report.summary())

    print(
        f"\nwithin {AZIMUTH_TOLERANCE_DEG:g} deg is over every item "
        "(an unanswered item is a wrong one); MAE is over answered items only, "
        "which is why coverage is printed beside it. 'inference' is the "
        "verifier's faithfulness rate and is deliberately not folded into "
        "the accuracy."
    )

    edges = reports[-1].by_angle()
    print("\nwithin-tolerance by planted angle (last row above):")
    print("  " + "  ".join(f"{angle:+.0f}" for angle in edges))
    print("  " + "  ".join(f"{rate:.0%}".rjust(3) for rate in edges.values()))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "labels": str(args.labels),
                    "items": len(labels),
                    "reports": [report.to_dict() for report in reports],
                },
                indent=1,
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
