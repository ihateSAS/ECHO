from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.build import sha256_file, write_jsonl
from benchmark.config import BenchmarkConfig
from benchmark.inventory import build_inventory
from benchmark.mixture_measure import measure_mixture
from benchmark.mixture_questions import make_mixture_localization_question
from benchmark.mixture_render import make_mixture_render_plans, render_mixture_audio
from benchmark.mixture_validate import validate_mixture_render
from benchmark.render import write_render
from benchmark.schemas import MixtureBenchmarkItem, to_json_dict
from benchmark.splits import assign_piece_splits, validate_split_assignments


@dataclass(frozen=True)
class MixtureBuildSummary:
    inventory_stems: int
    eligible_pieces: int
    skipped_pieces: int
    render_plans: int
    accepted_items: int
    rejected_items: int
    output_root: Path


def _public_test_item(item: MixtureBenchmarkItem) -> dict[str, object]:
    return {
        "schema_version": item.schema_version,
        "item_id": item.item_id,
        "audio_path": item.audio_path,
        "audio_sha256": item.audio_sha256,
        "split": item.split,
        "question_type": item.question.question_type,
        "question": item.question.question,
    }


def _private_test_label(item: MixtureBenchmarkItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "answer_count": item.question.answer_count,
        "answer_azimuths_deg": item.question.answer_azimuths_deg,
        "answer_sides": item.question.answer_sides,
        "render": to_json_dict(item.render),
        "measured_evidence": to_json_dict(item.measured_evidence),
    }


def build_mixture_benchmark(config: BenchmarkConfig) -> MixtureBuildSummary:
    inventory = build_inventory(config.source_root)
    assignments = assign_piece_splits(inventory, config.seed)
    validate_split_assignments(inventory, assignments)

    plans, skipped = make_mixture_render_plans(inventory, assignments, config)
    if not plans:
        raise ValueError("no pieces produced a placeable mixture render plan")

    config.output_root.mkdir(parents=True, exist_ok=True)
    accepted: list[MixtureBenchmarkItem] = []
    rejected: list[dict[str, object]] = []

    for plan in plans:
        stereo = render_mixture_audio(plan, inventory)
        evidence = measure_mixture(stereo, plan.sample_rate)
        failures = validate_mixture_render(stereo, plan, evidence, config)
        if failures:
            rejected.append(
                {
                    "render": to_json_dict(plan),
                    "measured_evidence": to_json_dict(evidence),
                    "failures": failures,
                }
            )
            continue

        relative_path = Path("mixture_audio") / plan.split / f"{plan.published_id}.wav"
        absolute_path = config.output_root / relative_path
        write_render(stereo, plan.sample_rate, absolute_path)
        question = make_mixture_localization_question(plan)
        accepted.append(
            MixtureBenchmarkItem(
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
            config.output_root / f"mixture_{split}.jsonl",
            [item for item in accepted if item.split == split],
        )
    test_items = [item for item in accepted if item.split == "test"]
    write_jsonl(
        config.output_root / "mixture_test_public.jsonl",
        [_public_test_item(item) for item in test_items],
    )
    write_jsonl(
        config.output_root / "mixture_test_private_labels.jsonl",
        [_private_test_label(item) for item in test_items],
    )
    write_jsonl(config.output_root / "mixture_rejected_items.jsonl", rejected)

    summary = MixtureBuildSummary(
        inventory_stems=len(inventory),
        eligible_pieces=len({plan.piece_id for plan in plans}),
        skipped_pieces=len(skipped),
        render_plans=len(plans),
        accepted_items=len(accepted),
        rejected_items=len(rejected),
        output_root=config.output_root,
    )
    (config.output_root / "mixture_build_summary.json").write_text(
        json.dumps(to_json_dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
