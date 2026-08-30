from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from benchmark.config import BenchmarkConfig
from benchmark.process import apply_processing_chain, processing_label
from benchmark.schemas import RenderPlan, Split, StemRecord
from benchmark.validate import rms_db
from spatialize.amplitude import render_amplitude_pan
from spatialize.binaural import binaural_render


def stable_seed(value: str, global_seed: int) -> int:
    digest = hashlib.sha256(f"{global_seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _excerpt_rms_db(stem: StemRecord, start_seconds: float, duration_seconds: float) -> float:
    frames = round(duration_seconds * stem.sample_rate)
    start = round(start_seconds * stem.sample_rate)
    mono, _ = sf.read(
        stem.path, start=start, frames=frames, dtype="float64", always_2d=True
    )
    return rms_db(mono[:, 0])


def _pick_start_seconds(stem: StemRecord, duration_seconds: float, config: BenchmarkConfig) -> float:
    maximum_start = duration_seconds - config.clip_duration_seconds
    seed = stable_seed(stem.stem_id, config.seed)
    rng = random.Random(seed)
    start_seconds = rng.uniform(0.0, maximum_start) if maximum_start > 0 else 0.0

    if not config.retry_quiet_excerpts or maximum_start <= 0.0:
        return start_seconds

    attempts = 1
    while (
        attempts < config.max_excerpt_retries
        and _excerpt_rms_db(stem, start_seconds, config.clip_duration_seconds)
        < config.minimum_rms_db
    ):
        start_seconds = rng.uniform(0.0, maximum_start)
        attempts += 1
    return start_seconds


def make_render_plans(
    inventory: list[StemRecord],
    assignments: dict[str, Split],
    config: BenchmarkConfig,
) -> list[RenderPlan]:
    plans: list[RenderPlan] = []
    for stem in inventory:
        duration_seconds = stem.frames / stem.sample_rate
        if duration_seconds < config.clip_duration_seconds:
            continue
        seed = stable_seed(stem.stem_id, config.seed)
        start_seconds = _pick_start_seconds(stem, duration_seconds, config)

        for angle in config.angles_degrees:
            angle_label = f"{angle:+g}".replace("+", "p").replace("-", "neg")


            regime_tag = "amp" if config.rendering == "amplitude" else "bin"
            for recipe in ((), *config.processing_recipes):
                base_id = (
                    f"{stem.stem_id}_{regime_tag}_{angle_label}_"
                    f"{round(start_seconds * 1000)}ms"
                )
                render_id = base_id if not recipe else f"{base_id}_{processing_label(recipe)}"
                plans.append(
                    RenderPlan(
                        render_id=render_id,
                        stem_id=stem.stem_id,
                        piece_id=stem.piece_id,
                        split=assignments[stem.piece_id],
                        instrument=stem.instrument_name,
                        instrument_code=stem.instrument_code,
                        source_path=stem.path,
                        start_seconds=start_seconds,
                        duration_seconds=config.clip_duration_seconds,
                        sample_rate=config.sample_rate,
                        regime=config.rendering,
                        azimuth_deg=angle,
                        seed=seed,
                        processing=recipe,
                    )
                )
    return assign_public_test_ids(plans, config.seed)


def assign_public_test_ids(
    plans: list[RenderPlan], global_seed: int
) -> list[RenderPlan]:
    test_plans = [plan for plan in plans if plan.split == "test"]
    order = list(range(len(test_plans)))
    random.Random(stable_seed("public-test-ids", global_seed)).shuffle(order)
    opaque = {
        plan.render_id: f"smq_test_{index:05d}"
        for plan, index in zip(test_plans, order)
    }
    return [
        replace(plan, public_id=opaque[plan.render_id])
        if plan.split == "test"
        else plan
        for plan in plans
    ]


def _load_excerpt(plan: RenderPlan) -> np.ndarray:
    info = sf.info(plan.source_path)
    source_start = round(plan.start_seconds * info.samplerate)
    source_frames = round(plan.duration_seconds * info.samplerate)
    mono, source_rate = sf.read(
        plan.source_path,
        start=source_start,
        frames=source_frames,
        dtype="float64",
        always_2d=True,
    )
    if mono.shape[1] != 1:
        raise ValueError(f"{plan.source_path} is not mono")
    signal = mono[:, 0]
    if source_rate != plan.sample_rate:
        gcd = int(np.gcd(source_rate, plan.sample_rate))
        signal = resample_poly(signal, plan.sample_rate // gcd, source_rate // gcd)

    required = round(plan.duration_seconds * plan.sample_rate)
    if signal.size < required:
        signal = np.pad(signal, (0, required - signal.size))
    return np.asarray(signal[:required], dtype=np.float64)


def render_audio(plan: RenderPlan) -> np.ndarray:
    excerpt = _load_excerpt(plan)
    if plan.regime == "binaural":
        return binaural_render(excerpt, plan.sample_rate, plan.azimuth_deg)
    stereo = render_amplitude_pan(excerpt, plan.azimuth_deg)
    if plan.processing:
        stereo = apply_processing_chain(stereo, plan.sample_rate, plan.processing)
    return stereo


def write_render(stereo: np.ndarray, sample_rate: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, stereo, sample_rate, subtype="PCM_24")
