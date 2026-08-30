from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchmark.schemas import ProcessingStep

Rendering = Literal["amplitude", "binaural"]


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_name: str
    version: str
    seed: int
    sample_rate: int
    clip_duration_seconds: float
    angles_degrees: tuple[float, ...]
    azimuth_tolerance_degrees: float
    minimum_rms_db: float
    maximum_peak: float
    source_root: Path
    output_root: Path


    processing_recipes: tuple[tuple[ProcessingStep, ...], ...] = ()


    retry_quiet_excerpts: bool = False
    max_excerpt_retries: int = 3


    rendering: Rendering = "amplitude"

    @property
    def maximum_abs_angle_deg(self) -> float:
        return 30.0 if self.rendering == "amplitude" else 90.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.clip_duration_seconds <= 0:
            raise ValueError("clip_duration_seconds must be positive")
        if self.rendering not in ("amplitude", "binaural"):
            raise ValueError(
                f"rendering must be 'amplitude' or 'binaural', got {self.rendering!r}"
            )
        if not self.angles_degrees:
            raise ValueError("angles_degrees must not be empty")
        limit = self.maximum_abs_angle_deg
        if any(abs(angle) >= limit for angle in self.angles_degrees):
            raise ValueError(
                f"{self.rendering} angles must be strictly inside "
                f"+/-{limit:g} degrees"
            )
        if self.rendering == "binaural" and self.processing_recipes:
            raise ValueError(
                "processing_recipes are not supported with binaural rendering: "
                "their effect on Woodworth ITD and head shadow has not been "
                "derived, so the resulting items would carry unverified "
                "ground truth (see benchmark/process.py)"
            )
        if self.azimuth_tolerance_degrees <= 0:
            raise ValueError("azimuth_tolerance_degrees must be positive")
        if not 0.0 < self.maximum_peak <= 1.0:
            raise ValueError("maximum_peak must be in (0, 1]")
        if self.max_excerpt_retries < 1:
            raise ValueError("max_excerpt_retries must be at least 1")


def _parse_processing_recipes(raw: list) -> tuple[tuple[ProcessingStep, ...], ...]:
    return tuple(
        tuple(
            ProcessingStep(
                kind=str(step["kind"]),
                params={str(k): float(v) for k, v in step["params"].items()},
            )
            for step in recipe
        )
        for recipe in raw
    )


def load_config(path: Path) -> BenchmarkConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    base = path.parent

    def resolved(raw_path: str) -> Path:
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    return BenchmarkConfig(
        benchmark_name=str(raw["benchmark_name"]),
        version=str(raw["version"]),
        seed=int(raw["seed"]),
        sample_rate=int(raw["sample_rate"]),
        clip_duration_seconds=float(raw["clip_duration_seconds"]),
        angles_degrees=tuple(float(value) for value in raw["angles_degrees"]),
        azimuth_tolerance_degrees=float(raw["azimuth_tolerance_degrees"]),
        minimum_rms_db=float(raw["minimum_rms_db"]),
        maximum_peak=float(raw["maximum_peak"]),
        source_root=resolved(str(raw["source_root"])),
        output_root=resolved(str(raw["output_root"])),
        processing_recipes=_parse_processing_recipes(raw.get("processing_recipes", [])),
        retry_quiet_excerpts=bool(raw.get("retry_quiet_excerpts", False)),
        max_excerpt_retries=int(raw.get("max_excerpt_retries", 3)),
        rendering=raw.get("rendering", "amplitude"),
    )
