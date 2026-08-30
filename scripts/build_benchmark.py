#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.build import build_benchmark
from benchmark.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build StereoMusicQA")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmark/configs/stereomusicqa_v0.1.json"),
    )
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    summary = build_benchmark(config)
    print(
        f"Built {summary.accepted_items} items from {summary.inventory_stems} stems; "
        f"rejected {summary.rejected_items}. Output: {summary.output_root}"
    )


if __name__ == "__main__":
    main()
