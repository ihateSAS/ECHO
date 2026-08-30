#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.config import load_config
from benchmark.mixture_build import build_mixture_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build StereoMusicQA mixture items")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark/configs/stereomusicqa_v0.2.json"),
    )
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    summary = build_mixture_benchmark(config)
    print(
        f"Built {summary.accepted_items} mixture items from "
        f"{summary.eligible_pieces} pieces ({summary.skipped_pieces} skipped, "
        f"couldn't be angle-separated); rejected {summary.rejected_items}. "
        f"Output: {summary.output_root}"
    )


if __name__ == "__main__":
    main()
