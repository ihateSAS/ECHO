#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mixture_score import (  # noqa: E402
    MixtureScoreReport,
    constant_count_baseline,
    front_end_baseline,
    read_mixture_labels,
    refusal_baseline,
    score_mixture_answers,
)
from eval.score import read_answers  # noqa: E402

DEFAULT_LABELS = Path("artifacts/stereomusicqa_v0.2/mixture_test_private_labels.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--answers", type=Path, default=None, help="model answers, one JSON per line"
    )
    parser.add_argument("--name", default="model")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.labels.is_file():
        raise SystemExit(
            f"no labels at {args.labels}. Build the mixture items first:\n"
            "  python scripts/build_mixture_benchmark.py "
            "--config benchmark/configs/stereomusicqa_v0.2.json"
        )
    labels = read_mixture_labels(args.labels)
    if not labels:
        raise SystemExit(f"{args.labels} has no items")

    reports: list[MixtureScoreReport] = []
    if args.answers is not None:
        reports.append(
            score_mixture_answers(labels, read_answers(args.answers), name=args.name)
        )
    if args.baselines or args.answers is None:
        reports.extend(
            [
                refusal_baseline(labels),
                constant_count_baseline(labels, count=2),
                constant_count_baseline(labels, count=3),
                front_end_baseline(labels),
            ]
        )

    print(f"{len(labels)} items from {args.labels}\n")
    for report in reports:
        print(report.summary())
    print(
        "\nexact match is over every item (an unanswered item is a wrong one); "
        "precision is over answered items only, which is why coverage is printed "
        "beside it."
    )

    if args.json is not None:
        import json

        args.json.write_text(
            json.dumps(
                {
                    "labels": str(args.labels),
                    "items": len(labels),
                    "reports": [report.to_dict() for report in reports],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
