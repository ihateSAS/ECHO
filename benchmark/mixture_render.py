from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from benchmark.config import BenchmarkConfig
from benchmark.render import stable_seed
from benchmark.schemas import MixtureRenderPlan, MixtureStemPlan, Split, StemRecord
from spatialize.mixture import StemPlacement, mix_stems


MIN_ANGLE_SEPARATION_DEG = 10.0


MAX_STEMS_PER_MIXTURE = 5


def _group_by_piece(inventory: list[StemRecord]) -> dict[str, list[StemRecord]]:
    groups: dict[str, list[StemRecord]] = {}
    for stem in inventory:
        groups.setdefault(stem.piece_id, []).append(stem)
    for stems in groups.values():
        stems.sort(key=lambda stem: (stem.instrument_code, stem.stem_id))
    return groups


def _pick_separated_angles(
    piece_id: str,
    count: int,
    angles_degrees: tuple[float, ...],
    global_seed: int,
    min_separation_deg: float,
) -> list[float] | None:
    rng = random.Random(stable_seed(f"mixture-angles-{piece_id}", global_seed))
    candidates = list(angles_degrees)
    for _ in range(2_000):
        rng.shuffle(candidates)
        chosen: list[float] = []
        for angle in candidates:
            if all(abs(angle - other) >= min_separation_deg for other in chosen):
                chosen.append(angle)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def make_mixture_render_plans(
    inventory: list[StemRecord],
    assignments: dict[str, Split],
    config: BenchmarkConfig,
) -> tuple[list[MixtureRenderPlan], list[str]]:
    plans: list[MixtureRenderPlan] = []
    skipped: list[str] = []

    for piece_id, stems in sorted(_group_by_piece(inventory).items()):
        if len(stems) < 2:
            continue
        if len(stems) > MAX_STEMS_PER_MIXTURE:
            skipped.append(piece_id)
            continue

        angles = _pick_separated_angles(
            piece_id, len(stems), config.angles_degrees, config.seed, MIN_ANGLE_SEPARATION_DEG
        )
        if angles is None:
            skipped.append(piece_id)
            continue

        shortest_duration = min(stem.frames / stem.sample_rate for stem in stems)
        if shortest_duration < config.clip_duration_seconds:
            skipped.append(piece_id)
            continue

        seed = stable_seed(piece_id, config.seed)
        rng = random.Random(seed)
        maximum_start = shortest_duration - config.clip_duration_seconds
        start_seconds = rng.uniform(0.0, maximum_start) if maximum_start > 0 else 0.0

        stem_plans = tuple(
            MixtureStemPlan(
                stem_id=stem.stem_id,
                label=stem.instrument_name,
                instrument_code=stem.instrument_code,
                azimuth_deg=angle,
            )
            for stem, angle in zip(stems, angles)
        )
        plans.append(
            MixtureRenderPlan(
                render_id=f"mixture_{piece_id}_{round(start_seconds * 1000)}ms",
                piece_id=piece_id,
                split=assignments[piece_id],
                stems=stem_plans,
                start_seconds=start_seconds,
                duration_seconds=config.clip_duration_seconds,
                sample_rate=config.sample_rate,
                seed=seed,
            )
        )

    return assign_public_test_ids(plans, config.seed), skipped


def assign_public_test_ids(
    plans: list[MixtureRenderPlan], global_seed: int
) -> list[MixtureRenderPlan]:
    test_plans = [plan for plan in plans if plan.split == "test"]
    order = list(range(len(test_plans)))
    random.Random(stable_seed("mixture-public-test-ids", global_seed)).shuffle(order)
    opaque = {
        plan.render_id: f"smq_mixture_test_{index:05d}"
        for plan, index in zip(test_plans, order)
    }
    return [
        replace(plan, public_id=opaque[plan.render_id]) if plan.split == "test" else plan
        for plan in plans
    ]


def _stem_path(inventory: list[StemRecord], stem_id: str) -> Path:
    for stem in inventory:
        if stem.stem_id == stem_id:
            return stem.path
    raise KeyError(f"stem_id {stem_id!r} not found in inventory")


def _load_excerpt(
    path: Path, start_seconds: float, duration_seconds: float, sample_rate: int
) -> np.ndarray:
    info = sf.info(path)
    source_start = round(start_seconds * info.samplerate)
    source_frames = round(duration_seconds * info.samplerate)
    mono, source_rate = sf.read(
        path, start=source_start, frames=source_frames, dtype="float64", always_2d=True
    )
    if mono.shape[1] != 1:
        raise ValueError(f"{path} is not mono")
    signal = mono[:, 0]
    if source_rate != sample_rate:
        gcd = int(np.gcd(source_rate, sample_rate))
        signal = resample_poly(signal, sample_rate // gcd, source_rate // gcd)

    required = round(duration_seconds * sample_rate)
    if signal.size < required:
        signal = np.pad(signal, (0, required - signal.size))
    return np.asarray(signal[:required], dtype=np.float64)


def render_mixture_audio(
    plan: MixtureRenderPlan, inventory: list[StemRecord]
) -> np.ndarray:
    placements = [
        StemPlacement(
            mono=_load_excerpt(
                _stem_path(inventory, stem.stem_id),
                plan.start_seconds,
                plan.duration_seconds,
                plan.sample_rate,
            ),
            azimuth_deg=stem.azimuth_deg,
            label=stem.label,
        )
        for stem in plan.stems
    ]
    stereo, _ = mix_stems(placements)
    return stereo
