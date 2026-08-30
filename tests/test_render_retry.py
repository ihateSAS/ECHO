from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.build import build_benchmark  # noqa: E402
from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.render import _excerpt_rms_db, _pick_start_seconds, stable_seed  # noqa: E402
from benchmark.schemas import StemRecord  # noqa: E402

SAMPLE_RATE = 48_000


def _config(source: Path, output: Path, **overrides) -> BenchmarkConfig:
    defaults = dict(
        benchmark_name="StereoMusicQA",
        version="0.2.0-test",
        seed=2026,
        sample_rate=SAMPLE_RATE,
        clip_duration_seconds=0.2,
        angles_degrees=(-15.0, 0.0, 15.0),
        azimuth_tolerance_degrees=5.0,
        minimum_rms_db=-45.0,
        maximum_peak=0.99,
        source_root=source,
        output_root=output,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _write_mostly_silent_stem(path: Path, loud_start_s: float, loud_len_s: float = 1.0) -> StemRecord:
    total_s = 10.0
    samples = round(total_s * SAMPLE_RATE)
    audio = 1e-6 * np.random.default_rng(0).standard_normal(samples)

    loud_start = round(loud_start_s * SAMPLE_RATE)
    loud_len = round(loud_len_s * SAMPLE_RATE)
    audio[loud_start : loud_start + loud_len] = 0.3 * np.sin(
        2 * np.pi * 440.0 * np.arange(loud_len) / SAMPLE_RATE
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_24")
    return StemRecord(
        stem_id=path.stem,
        piece_id="01_Test",
        instrument_code="vc",
        instrument_name="cello",
        path=path,
        sample_rate=SAMPLE_RATE,
        channels=1,
        frames=samples,
    )


def test_retry_disabled_reproduces_the_single_draw_exactly(tmp_path: Path):
    stem = _write_mostly_silent_stem(tmp_path / "AuSep_1_vc_01_Test.wav", loud_start_s=5.0)
    off = _config(tmp_path, tmp_path / "out", retry_quiet_excerpts=False)
    on = _config(tmp_path, tmp_path / "out", retry_quiet_excerpts=True)


    seed = stable_seed(stem.stem_id, off.seed)
    import random

    expected_first_draw = random.Random(seed).uniform(
        0.0, (stem.frames / stem.sample_rate) - off.clip_duration_seconds
    )
    assert _pick_start_seconds(
        stem, stem.frames / stem.sample_rate, off
    ) == pytest.approx(expected_first_draw)


    assert (
        _excerpt_rms_db(stem, expected_first_draw, off.clip_duration_seconds)
        < off.minimum_rms_db
    )


def test_retry_enabled_finds_a_loud_excerpt_when_the_first_draw_is_quiet(tmp_path: Path):
    stem = _write_mostly_silent_stem(tmp_path / "AuSep_1_vc_01_Test.wav", loud_start_s=5.0)
    config = _config(
        tmp_path, tmp_path / "out", retry_quiet_excerpts=True, max_excerpt_retries=50
    )
    picked = _pick_start_seconds(stem, stem.frames / stem.sample_rate, config)
    assert _excerpt_rms_db(stem, picked, config.clip_duration_seconds) >= config.minimum_rms_db


def test_retry_is_deterministic(tmp_path: Path):
    stem = _write_mostly_silent_stem(tmp_path / "AuSep_1_vc_01_Test.wav", loud_start_s=5.0)
    config = _config(
        tmp_path, tmp_path / "out", retry_quiet_excerpts=True, max_excerpt_retries=50
    )
    first = _pick_start_seconds(stem, stem.frames / stem.sample_rate, config)
    second = _pick_start_seconds(stem, stem.frames / stem.sample_rate, config)
    assert first == second


def test_retry_gives_up_after_max_attempts_on_a_stem_thats_quiet_everywhere(tmp_path: Path):
    path = tmp_path / "AuSep_1_vc_01_Silent.wav"
    samples = round(3.0 * SAMPLE_RATE)
    sf.write(path, 1e-6 * np.random.default_rng(1).standard_normal(samples), SAMPLE_RATE, subtype="PCM_24")
    stem = StemRecord(
        stem_id=path.stem,
        piece_id="02_Silent",
        instrument_code="vc",
        instrument_name="cello",
        path=path,
        sample_rate=SAMPLE_RATE,
        channels=1,
        frames=samples,
    )
    config = _config(
        tmp_path, tmp_path / "out", retry_quiet_excerpts=True, max_excerpt_retries=3
    )


    picked = _pick_start_seconds(stem, stem.frames / stem.sample_rate, config)
    assert isinstance(picked, float)


def test_end_to_end_build_recovers_an_item_the_no_retry_build_drops(tmp_path: Path):
    source = tmp_path / "URMP"
    stem = _write_mostly_silent_stem(source / "AuSep_1_vc_01_Test.wav", loud_start_s=5.0)


    assert stem

    off_config = _config(
        source, tmp_path / "off", retry_quiet_excerpts=False, max_excerpt_retries=50
    )
    on_config = _config(
        source, tmp_path / "on", retry_quiet_excerpts=True, max_excerpt_retries=50
    )

    off_summary = build_benchmark(off_config)
    on_summary = build_benchmark(on_config)

    assert off_summary.rejected_items == off_summary.render_plans
    assert off_summary.accepted_items == 0
    assert on_summary.accepted_items == on_summary.render_plans
    assert on_summary.rejected_items == 0
