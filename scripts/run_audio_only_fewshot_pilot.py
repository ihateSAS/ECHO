from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.providers import FewShotExample, get_api_provider  # noqa: E402

ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "stereomusicqa_v0.2"
TEST_PUBLIC = ARTIFACTS_ROOT / "test_public.jsonl"
TRAIN = ARTIFACTS_ROOT / "train.jsonl"

FEWSHOT_ITEM_IDS = (
    "AuSep_3_va_44_K515_amp_p20_9194ms_absolute",
    "AuSep_1_fl_37_Rondeau_amp_neg20_40799ms_absolute",
    "AuSep_1_vn_26_King_amp_p0_34533ms_absolute",
)

STEREO_PREAMBLE_QUESTION = "Where is the instrument positioned in the stereo image?"


def _load_record(records_by_id: dict, item_id: str) -> dict:
    return records_by_id[item_id]


def load_fewshot_examples() -> list[FewShotExample]:
    records = {}
    with TRAIN.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["item_id"]] = record

    examples = []
    for item_id in FEWSHOT_ITEM_IDS:
        record = records[item_id]
        stereo, sample_rate = sf.read(
            ARTIFACTS_ROOT / record["audio_path"], dtype="float64", always_2d=True
        )
        examples.append(
            FewShotExample(
                stereo=stereo,
                sample_rate=sample_rate,
                side=record["question"]["answer_side"],
            )
        )
    return examples


def pilot_item_ids(n: int, seed: int = 0) -> list[str]:
    items = [
        json.loads(line)["item_id"]
        for line in TEST_PUBLIC.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--answers-path", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    answers_path = args.answers_path or (
        REPO_ROOT / "answers" / f"{args.provider}_audio_only_fewshot_pilot.jsonl"
    )
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    provider = get_api_provider(args.provider)
    examples = load_fewshot_examples()
    ids = pilot_item_ids(args.n, args.seed)

    done = set()
    if answers_path.exists():
        done = {
            json.loads(line)["item_id"]
            for line in answers_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    test_records = {}
    with TEST_PUBLIC.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["item_id"] in ids:
                test_records[record["item_id"]] = record

    print(
        f"provider={provider.name} pilot_n={len(ids)} fewshot_k={len(examples)} "
        f"already_done={len(done)} -> {answers_path}"
    )

    written = 0
    with answers_path.open("a", encoding="utf-8") as out:
        for index, item_id in enumerate(ids):
            if item_id in done:
                continue
            record = test_records[item_id]
            stereo, sample_rate = sf.read(
                ARTIFACTS_ROOT / record["audio_path"], dtype="float64", always_2d=True
            )

            started = time.time()
            reply = None
            last_error = None
            for attempt in range(4):
                try:
                    reply = provider.answer_audio_only_fewshot(
                        examples, stereo, sample_rate, record["question"]
                    )
                    break
                except Exception as error:  # noqa: BLE001
                    last_error = error
                    if attempt < 3:
                        backoff = 2.0 * (attempt + 1)
                        print(f"  retry {attempt + 1}/3 after {type(error).__name__}: {backoff}s")
                        time.sleep(backoff)

            if reply is None:
                print(f"[{index + 1}/{len(ids)}] {item_id}: SKIPPED after retries -- {last_error}")
                continue

            out.write(json.dumps({"item_id": item_id, "answer": reply}) + "\n")
            out.flush()
            written += 1
            print(f"[{index + 1}/{len(ids)}] {item_id} ({time.time() - started:.1f}s): {reply[:70]!r}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"wrote {written} new answers to {answers_path}")


if __name__ == "__main__":
    main()
