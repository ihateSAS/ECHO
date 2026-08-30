from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from benchmark.config import BenchmarkConfig


BUILD_ID_LENGTH = 12


def fingerprint_fields(config: BenchmarkConfig) -> dict[str, object]:
    return {
        "benchmark_name": config.benchmark_name,
        "version": config.version,
        "seed": config.seed,
        "sample_rate": config.sample_rate,
        "clip_duration_seconds": config.clip_duration_seconds,
        "angles_degrees": list(config.angles_degrees),
        "azimuth_tolerance_degrees": config.azimuth_tolerance_degrees,


        "rendering": config.rendering,
        "processing_recipes": [
            [{"kind": step.kind, "params": dict(sorted(step.params.items()))} for step in recipe]
            for recipe in config.processing_recipes
        ],
    }


def build_id(config: BenchmarkConfig) -> str:
    payload = json.dumps(fingerprint_fields(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:BUILD_ID_LENGTH]


class BuildMismatch(RuntimeError):
    pass


def check_build_ids(
    labels_build_id: str | None,
    answers_build_id: str | None,
    labels_name: str = "labels",
    answers_name: str = "answers",
) -> None:
    if labels_build_id is None or answers_build_id is None:
        return
    if labels_build_id != answers_build_id:
        raise BuildMismatch(
            f"{answers_name} were generated against benchmark build "
            f"{answers_build_id!r}, but {labels_name} are from build "
            f"{labels_build_id!r}. These are different benchmarks that share "
            "an item-id space, so scoring them together would silently "
            "compare unrelated items. Regenerate the answers against the "
            "labels you want to score, or pass --allow-build-mismatch if you "
            "genuinely mean to do this."
        )
