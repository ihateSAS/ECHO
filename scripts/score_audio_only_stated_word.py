from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_SIDE_RE = re.compile(r"\b(left|right|cent(?:er|re))\b", re.IGNORECASE)


def stated_side(text: str) -> str | None:
    match = _SIDE_RE.search(text)
    if match is None:
        return None
    word = match.group(1).lower()
    return "center" if word.startswith("cent") else word


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args()

    true_side: dict[str, str] = {}
    with args.labels.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            true_side[record["item_id"]] = record["answer_side"]

    answers: dict[str, str] = {}
    with args.answers.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            answers[record["item_id"]] = record["answer"]

    confusion: Counter[tuple[str, str | None]] = Counter()
    stated_dist: Counter[str | None] = Counter()
    correct = 0
    total = 0

    for item_id, side in true_side.items():
        if item_id not in answers:
            continue
        total += 1
        said = stated_side(answers[item_id])
        stated_dist[said] += 1
        confusion[(side, said)] += 1
        if said == side:
            correct += 1

    print(f"{total} items scored from {args.answers}")
    print(f"side accuracy (stated word): {correct / total:.1%}")
    print()
    print("stated-side distribution:", dict(stated_dist))
    print()
    print("confusion (true side -> stated side):")
    for true in ("left", "right", "center"):
        n_true = sum(v for (t, _), v in confusion.items() if t == true)
        if n_true == 0:
            continue
        row = {said: v for (t, said), v in confusion.items() if t == true}
        n_correct = row.get(true, 0)
        print(f"  true={true:6s} n={n_true:4d} correct={n_correct / n_true:5.1%}  {row}")


if __name__ == "__main__":
    main()
