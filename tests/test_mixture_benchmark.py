from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.inventory import build_inventory  # noqa: E402
from benchmark.mixture_build import build_mixture_benchmark  # noqa: E402
from benchmark.mixture_measure import measure_mixture  # noqa: E402
from benchmark.mixture_questions import make_mixture_localization_question  # noqa: E402
from benchmark.mixture_render import (  # noqa: E402
    MAX_STEMS_PER_MIXTURE,
    MIN_ANGLE_SEPARATION_DEG,
    _pick_separated_angles,
    make_mixture_render_plans,
    render_mixture_audio,
)
from benchmark.mixture_validate import validate_mixture_render  # noqa: E402
from benchmark.schemas import MixtureRenderPlan, MixtureStemPlan  # noqa: E402
from benchmark.splits import assign_piece_splits  # noqa: E402

SAMPLE_RATE = 48_000
ANGLES = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)

URMP_ROOT = Path(__file__).resolve().parent.parent / "data" / "URMP"
needs_urmp = pytest.mark.skipif(
    not URMP_ROOT.exists(), reason=f"URMP dataset not present under {URMP_ROOT}"
)


def _config(source_root: Path, output_root: Path, **overrides) -> BenchmarkConfig:
    defaults = dict(
        benchmark_name="StereoMusicQA",
        version="0.2.0-test",
        seed=2026,
        sample_rate=SAMPLE_RATE,
        clip_duration_seconds=0.5,
        angles_degrees=ANGLES,
        azimuth_tolerance_degrees=5.0,
        minimum_rms_db=-45.0,
        maximum_peak=0.99,
        source_root=source_root,
        output_root=output_root,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_pick_separated_angles_respects_the_minimum_separation():
    chosen = _pick_separated_angles("piece", 5, ANGLES, global_seed=2026, min_separation_deg=10.0)
    assert chosen is not None
    assert len(chosen) == 5
    for i, angle in enumerate(chosen):
        for other in chosen[i + 1 :]:
            assert abs(angle - other) >= 10.0


def test_pick_separated_angles_is_deterministic():
    first = _pick_separated_angles("piece-x", 4, ANGLES, global_seed=7, min_separation_deg=10.0)
    second = _pick_separated_angles("piece-x", 4, ANGLES, global_seed=7, min_separation_deg=10.0)
    assert first == second


def test_pick_separated_angles_returns_none_when_infeasible():
    assert (
        _pick_separated_angles("piece", 7, ANGLES, global_seed=2026, min_separation_deg=10.0)
        is None
    )


def test_max_stems_per_mixture_is_actually_placeable():
    assert (
        _pick_separated_angles(
            "any", MAX_STEMS_PER_MIXTURE, ANGLES, global_seed=1, min_separation_deg=MIN_ANGLE_SEPARATION_DEG
        )
        is not None
    )


def test_question_reports_sorted_left_to_right_answers():
    plan = MixtureRenderPlan(
        render_id="mixture_test_0ms",
        piece_id="test_piece",
        split="test",
        stems=(
            MixtureStemPlan("s1", "violin", "vn", azimuth_deg=15.0),
            MixtureStemPlan("s2", "cello", "vc", azimuth_deg=-20.0),
            MixtureStemPlan("s3", "flute", "fl", azimuth_deg=0.0),
        ),
        start_seconds=0.0,
        duration_seconds=0.5,
        sample_rate=SAMPLE_RATE,
        seed=1,
    )
    question = make_mixture_localization_question(plan)
    assert question.answer_count == 3


    assert question.answer_azimuths_deg == (15.0, 0.0, -20.0)
    assert question.answer_sides == ("left", "center", "right")


    for name in ("violin", "cello", "flute"):
        assert name not in question.question


@needs_urmp
def test_real_urmp_duet_recovers_planted_angles_through_the_full_pipeline(tmp_path: Path):
    inventory = build_inventory(URMP_ROOT)
    assignments = assign_piece_splits(inventory, seed=2026)
    config = _config(URMP_ROOT, tmp_path / "out", clip_duration_seconds=4.0)

    plans, skipped = make_mixture_render_plans(inventory, assignments, config)
    jupiter = next((plan for plan in plans if plan.piece_id == "01_Jupiter"), None)
    assert jupiter is not None, "01_Jupiter (violin/cello duet) should produce a plan"
    assert len(jupiter.stems) == 2

    stereo = render_mixture_audio(jupiter, inventory)
    evidence = measure_mixture(stereo, jupiter.sample_rate)
    failures = validate_mixture_render(stereo, jupiter, evidence, config)

    assert failures == []
    assert evidence.source_count == 2
    planted = sorted(stem.azimuth_deg for stem in jupiter.stems)
    measured = sorted(evidence.source_azimuths_deg)
    for planted_angle, measured_angle in zip(planted, measured):
        assert abs(planted_angle - measured_angle) <= config.azimuth_tolerance_degrees


def _write_mini_multi_instrument_urmp(root: Path) -> None:
    root.mkdir(parents=True)
    samples = round(2.0 * SAMPLE_RATE)
    codes = ["vn", "vc", "va", "fl", "ob", "cl", "bn"]

    def _tone(fundamental_hz: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.arange(samples) / SAMPLE_RATE
        signal = np.zeros_like(t)
        for harmonic in range(1, 5):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal += np.sin(2 * np.pi * fundamental_hz * harmonic * t + phase) / harmonic
        return 0.2 * signal / np.max(np.abs(signal))

    piece_sizes = {"01_Solo": 1, "02_Duet": 2, "03_Trio": 3, "04_Sextet": 6}
    for piece_number, (name, size) in enumerate(piece_sizes.items(), start=1):
        for index, code in enumerate(codes[:size], start=1):
            fundamental = 220.0 * (2 ** (index / 12.0))
            sf.write(
                root / f"AuSep_{index}_{code}_{piece_number:02d}_{name}.wav",
                _tone(fundamental, seed=piece_number * 10 + index),
                SAMPLE_RATE,
                subtype="PCM_24",
            )


def test_build_mixture_benchmark_end_to_end(tmp_path: Path):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_multi_instrument_urmp(source)
    config = _config(source, output, clip_duration_seconds=0.5)

    summary = build_mixture_benchmark(config)


    assert summary.skipped_pieces == 1
    assert summary.eligible_pieces == 2
    assert summary.accepted_items + summary.rejected_items == summary.render_plans

    rejected = _read_jsonl(output / "mixture_rejected_items.jsonl")
    assert len(rejected) == summary.rejected_items

    for split in ("train", "dev"):
        for item in _read_jsonl(output / f"mixture_{split}.jsonl"):
            assert (output / item["audio_path"]).exists()
            assert item["question"]["answer_count"] >= 2

    private = _read_jsonl(output / "mixture_test_private_labels.jsonl")
    public = _read_jsonl(output / "mixture_test_public.jsonl")
    assert len(private) == len(public)
    assert all("answer_azimuths_deg" not in item for item in public)


    inventory = build_inventory(source)
    reference_assignments = assign_piece_splits(inventory, seed=config.seed)
    for item in _read_jsonl(output / "mixture_train.jsonl") + _read_jsonl(
        output / "mixture_dev.jsonl"
    ):
        assert item["split"] == reference_assignments[item["render"]["piece_id"]]


def test_build_mixture_benchmark_is_deterministic(tmp_path: Path):
    source = tmp_path / "URMP"
    _write_mini_multi_instrument_urmp(source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_mixture_benchmark(_config(source, first, clip_duration_seconds=0.5))
    build_mixture_benchmark(_config(source, second, clip_duration_seconds=0.5))

    for filename in (
        "mixture_train.jsonl",
        "mixture_dev.jsonl",
        "mixture_test_public.jsonl",
        "mixture_test_private_labels.jsonl",
        "mixture_rejected_items.jsonl",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
