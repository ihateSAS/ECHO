#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.questions import side_for_angle
from eval.score import signed_azimuth
from scripts.score_audio_only_stated_word import stated_side


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = REPO_ROOT / "artifacts/stereomusicqa_v0.2/test_private_labels.jsonl"


@dataclass(frozen=True)
class ModelAnswers:
    name: str
    cue_text: Path
    audio_only: Path


MODELS = (
    ModelAnswers(
        "Qwen2-Audio-7B-Instruct",
        REPO_ROOT
        / "experiments/2026-08-10-qwen2audio-corrected-question-full/answers.jsonl",
        REPO_ROOT
        / "experiments/2026-08-10-audio-only-stereo-perception/answers.jsonl",
    ),
    ModelAnswers(
        "Gemini Flash-Lite",
        REPO_ROOT
        / "experiments/2026-08-11-multimodel-comparison/answers_cues_gemini.jsonl",
        REPO_ROOT
        / "experiments/2026-08-11-multimodel-comparison/answers_audio_only_gemini.jsonl",
    ),
    ModelAnswers(
        "GPT-audio",
        REPO_ROOT
        / "experiments/2026-08-11-multimodel-comparison/answers_cues_gptaudio.jsonl",
        REPO_ROOT
        / "experiments/2026-08-11-multimodel-comparison/answers_audio_only_gptaudio.jsonl",
    ),
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _answers(path: Path) -> dict[str, str]:
    return {row["item_id"]: row["answer"] for row in _jsonl(path)}


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


SIDES = ("left", "right", "center")


def _balanced_accuracy(true: list[str], predicted: list[str | None]) -> float:
    recalls = []
    for side in SIDES:
        relevant = [index for index, value in enumerate(true) if value == side]
        recalls.append(np.mean([predicted[index] == side for index in relevant]))
    return float(np.mean(recalls))


def _macro_f1(true: list[str], predicted: list[str | None]) -> float:
    scores = []
    for side in SIDES:
        true_positive = sum(t == side and p == side for t, p in zip(true, predicted))
        false_positive = sum(t != side and p == side for t, p in zip(true, predicted))
        false_negative = sum(t == side and p != side for t, p in zip(true, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def evaluate(
    model: ModelAnswers,
    *,
    seed: int,
    samples: int,
    cluster_by: str,
) -> dict[str, object]:
    cue_answers = _answers(model.cue_text)
    audio_answers = _answers(model.audio_only)
    rows: list[dict[str, object]] = []

    for label in _jsonl(LABELS):
        item_id = label["item_id"]
        true_side = label["answer_side"]

        cue_angle = signed_azimuth(cue_answers.get(item_id, ""))
        cue_side = None if cue_angle is None else side_for_angle(cue_angle)
        audio_side = stated_side(audio_answers.get(item_id, ""))
        rows.append({
            "stem": label["render"]["stem_id"],
            "piece": label["render"]["piece_id"],
            "true_side": true_side,
            "cue_side": cue_side,
            "audio_side": audio_side,
            "cue_correct": cue_side == true_side,
            "audio_correct": audio_side == true_side,
        })

    clusters = sorted({str(row[cluster_by]) for row in rows})
    by_cluster = {
        cluster: [row for row in rows if row[cluster_by] == cluster]
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    boot = np.empty((samples, 3), dtype=np.float64)

    for index in range(samples):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_rows = [
            row for cluster in sampled_clusters for row in by_cluster[cluster]
        ]
        cue_rate = np.mean([bool(row["cue_correct"]) for row in sampled_rows])
        audio_rate = np.mean([bool(row["audio_correct"]) for row in sampled_rows])
        boot[index] = cue_rate, audio_rate, cue_rate - audio_rate

    true = [str(row["true_side"]) for row in rows]
    cue_predicted = [row["cue_side"] for row in rows]
    audio_predicted = [row["audio_side"] for row in rows]
    cue_rate = float(np.mean([bool(row["cue_correct"]) for row in rows]))
    audio_rate = float(np.mean([bool(row["audio_correct"]) for row in rows]))
    return {
        "model": model.name,
        "items": len(rows),
        "cluster_by": cluster_by,
        "clusters": len(clusters),
        "cue_text_side_accuracy": cue_rate,
        "cue_text_balanced_accuracy": _balanced_accuracy(true, cue_predicted),
        "cue_text_macro_f1": _macro_f1(true, cue_predicted),
        "cue_text_ci95": _percentile_interval(boot[:, 0]),
        "audio_only_side_accuracy": audio_rate,
        "audio_only_balanced_accuracy": _balanced_accuracy(true, audio_predicted),
        "audio_only_macro_f1": _macro_f1(true, audio_predicted),
        "audio_only_ci95": _percentile_interval(boot[:, 1]),
        "paired_gap": cue_rate - audio_rate,
        "paired_gap_ci95": _percentile_interval(boot[:, 2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20_260_814)
    parser.add_argument("--samples", type=int, default=20_000)
    args = parser.parse_args()
    result = {
        "method": "paired percentile cluster bootstrap",
        "seed": args.seed,
        "bootstrap_samples": args.samples,
        "stem_cluster_analysis": [
            evaluate(model, seed=args.seed, samples=args.samples, cluster_by="stem")
            for model in MODELS
        ],
        "piece_cluster_sensitivity": [
            evaluate(model, seed=args.seed, samples=args.samples, cluster_by="piece")
            for model in MODELS
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
