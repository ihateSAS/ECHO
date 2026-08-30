from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.config import BenchmarkConfig
from benchmark.inventory import build_inventory
from benchmark.measure import measure_render
from benchmark.provenance import build_id
from benchmark.questions import make_absolute_localization_question
from benchmark.render import make_render_plans, render_audio, write_render
from benchmark.schemas import BenchmarkItem, to_json_dict
from benchmark.splits import (
    assign_piece_splits,
    validate_split_assignments,
    write_split_files,
)
from benchmark.validate import validate_render


@dataclass(frozen=True)
class BuildSummary:
    build_id: str
    inventory_stems: int
    render_plans: int
    accepted_items: int
    rejected_items: int
    output_root: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            payload = record if isinstance(record, dict) else to_json_dict(record)
            json.dump(payload, file, ensure_ascii=False, sort_keys=True)
            file.write("\n")


def _public_test_item(item: BenchmarkItem, build: str) -> dict[str, object]:
    return {
        "build_id": build,
        "schema_version": item.schema_version,
        "item_id": item.item_id,
        "audio_path": item.audio_path,
        "audio_sha256": item.audio_sha256,
        "split": item.split,
        "question_type": item.question.question_type,
        "question": item.question.question,
        "target_instrument": item.question.target_instrument,
    }


def _private_test_label(item: BenchmarkItem, build: str) -> dict[str, object]:
    return {
        "build_id": build,
        "item_id": item.item_id,
        "answer_side": item.question.answer_side,
        "answer_azimuth_deg": item.question.answer_azimuth_deg,
        "render": to_json_dict(item.render),
        "measured_evidence": to_json_dict(item.measured_evidence),
    }


def build_benchmark(config: BenchmarkConfig) -> BuildSummary:
    build = build_id(config)
    inventory = build_inventory(config.source_root)
    assignments = assign_piece_splits(inventory, config.seed)
    validate_split_assignments(inventory, assignments)
    plans = make_render_plans(inventory, assignments, config)
    if not plans:
        raise ValueError("no stems are long enough for the configured clip duration")

    config.output_root.mkdir(parents=True, exist_ok=True)
    write_split_files(assignments, config.output_root / "splits")
    accepted: list[BenchmarkItem] = []
    rejected: list[dict[str, object]] = []

    for plan in plans:
        stereo = render_audio(plan)
        evidence = measure_render(stereo, plan.sample_rate, plan.instrument_code)
        failures = validate_render(stereo, plan, evidence, config)
        if failures:
            rejected.append(
                {
                    "render": to_json_dict(plan),
                    "measured_evidence": to_json_dict(evidence),
                    "failures": failures,
                }
            )
            continue

        relative_path = Path("audio") / plan.split / f"{plan.published_id}.wav"
        absolute_path = config.output_root / relative_path
        write_render(stereo, plan.sample_rate, absolute_path)
        question = make_absolute_localization_question(plan)
        accepted.append(
            BenchmarkItem(
                schema_version=config.version,
                item_id=question.question_id,
                audio_path=str(relative_path),
                audio_sha256=sha256_file(absolute_path),
                split=plan.split,
                question=question,
                render=plan,
                measured_evidence=evidence,
            )
        )

    for split in ("train", "dev"):
        write_jsonl(
            config.output_root / f"{split}.jsonl",
            [
                {"build_id": build, **to_json_dict(item)}
                for item in accepted
                if item.split == split
            ],
        )
    test_items = [item for item in accepted if item.split == "test"]
    write_jsonl(
        config.output_root / "test_public.jsonl",
        [_public_test_item(item, build) for item in test_items],
    )
    write_jsonl(
        config.output_root / "test_private_labels.jsonl",
        [_private_test_label(item, build) for item in test_items],
    )
    write_jsonl(config.output_root / "rejected_items.jsonl", rejected)

    summary = BuildSummary(
        build_id=build,
        inventory_stems=len(inventory),
        render_plans=len(plans),
        accepted_items=len(accepted),
        rejected_items=len(rejected),
        output_root=config.output_root,
    )
    (config.output_root / "build_summary.json").write_text(
        json.dumps(to_json_dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
