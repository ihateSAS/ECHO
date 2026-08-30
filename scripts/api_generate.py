#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf

from benchmark.inventory import INSTRUMENT_NAMES
from benchmark.measure import measure_render
from model.providers import get_api_provider
from model.qwen_backbone import describe_cues

ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "stereomusicqa_v0.2"
TEST_PUBLIC = ARTIFACTS_ROOT / "test_public.jsonl"

INSTRUMENT_CODE_FOR_NAME = {name: code for code, name in INSTRUMENT_NAMES.items()}


def _read_items(limit: int | None) -> list[dict]:
    items = [
        json.loads(line)
        for line in TEST_PUBLIC.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        import random

        shuffled = items[:]
        random.Random(0).shuffle(shuffled)
        items = shuffled[:limit]
    return items


def _already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["item_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, help="e.g. gpt-4o-audio, gemini")
    parser.add_argument("--audio-only", action="store_true", help="perception condition")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--answers-path", type=Path, default=None)
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="seconds between calls (rate limits)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not TEST_PUBLIC.is_file():
        raise SystemExit(f"no {TEST_PUBLIC} -- build the benchmark first")

    condition = "audio_only" if args.audio_only else "cues"
    answers_path = args.answers_path or (
        REPO_ROOT
        / "answers"
        / f"{args.provider}_{condition}.jsonl".replace("/", "_")
    )
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    provider = get_api_provider(args.provider)
    items = _read_items(args.limit)
    done = _already_done(answers_path)
    print(
        f"provider={provider.name} condition={condition} "
        f"items={len(items)} already_done={len(done)} -> {answers_path}"
    )

    written = 0
    with answers_path.open("a", encoding="utf-8") as out:
        for index, item in enumerate(items):
            if item["item_id"] in done:
                continue
            stereo, sample_rate = sf.read(
                ARTIFACTS_ROOT / item["audio_path"], dtype="float64", always_2d=True
            )

            started = time.time()
            reply = None
            last_error: Exception | None = None
            for attempt in range(4):
                try:
                    if args.audio_only:
                        reply = provider.answer_audio_only(
                            stereo, sample_rate, item["question"]
                        )
                    else:
                        instrument = item["target_instrument"]
                        evidence = measure_render(
                            stereo, sample_rate, INSTRUMENT_CODE_FOR_NAME.get(instrument)
                        )
                        f0 = evidence.f0_hz if evidence.f0_hz == evidence.f0_hz else None
                        cue_text = describe_cues(evidence, instrument=instrument, f0_hz=f0)
                        reply = provider.answer_with_cues(
                            stereo, sample_rate, cue_text, item["question"]
                        )
                    break
                except Exception as error:  # noqa: BLE001 -- provider errors are unpredictable
                    last_error = error
                    if attempt < 3:
                        backoff = 2.0 * (attempt + 1)
                        print(f"  retry {attempt + 1}/3 after {type(error).__name__}: {backoff}s")
                        time.sleep(backoff)

            if reply is None:
                print(f"[{index + 1}/{len(items)}] {item['item_id']}: SKIPPED after retries -- {last_error}")
                continue

            out.write(
                json.dumps(
                    {"item_id": item["item_id"], "answer": reply}
                    | ({"build_id": item["build_id"]} if item.get("build_id") else {})
                )
                + "\n"
            )
            out.flush()
            written += 1
            print(
                f"[{index + 1}/{len(items)}] {item['item_id']} "
                f"({time.time() - started:.1f}s): {reply[:70]!r}"
            )
            if args.sleep:
                time.sleep(args.sleep)

    print(f"wrote {written} new answers to {answers_path}")
    print(
        "score with:\n"
        f"  python scripts/score_benchmark.py --labels "
        f"artifacts/stereomusicqa_v0.2/test_private_labels.jsonl "
        f"--answers {answers_path} --baselines"
    )


if __name__ == "__main__":
    main()
