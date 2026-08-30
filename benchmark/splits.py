from __future__ import annotations

import random
from pathlib import Path
from typing import cast

from benchmark.schemas import Split, StemRecord


def assign_piece_splits(
    inventory: list[StemRecord],
    seed: int,
    train_fraction: float = 0.70,
    dev_fraction: float = 0.15,
) -> dict[str, Split]:
    if train_fraction < 0 or dev_fraction < 0 or train_fraction + dev_fraction > 1:
        raise ValueError("invalid split fractions")
    piece_ids = sorted({record.piece_id for record in inventory})
    if not piece_ids:
        raise ValueError("cannot split an empty inventory")

    rng = random.Random(seed)
    rng.shuffle(piece_ids)
    train_end = round(len(piece_ids) * train_fraction)
    dev_end = train_end + round(len(piece_ids) * dev_fraction)
    assignments: dict[str, Split] = {}
    for index, piece_id in enumerate(piece_ids):
        label = "train" if index < train_end else "dev" if index < dev_end else "test"
        assignments[piece_id] = cast(Split, label)
    return assignments


def validate_split_assignments(
    inventory: list[StemRecord], assignments: dict[str, Split]
) -> None:
    expected = {record.piece_id for record in inventory}
    if set(assignments) != expected:
        raise ValueError("split assignments do not match inventory piece IDs")


def write_split_files(assignments: dict[str, Split], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        piece_ids = sorted(key for key, value in assignments.items() if value == split)
        (output_dir / f"{split}.txt").write_text(
            "".join(f"{piece_id}\n" for piece_id in piece_ids), encoding="utf-8"
        )
