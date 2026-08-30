from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark.config import BenchmarkConfig
from benchmark.measure import measure_render
from benchmark.questions import answerable_azimuth_deg
from benchmark.render import render_audio
from benchmark.schemas import ProcessingStep, RenderPlan
from benchmark.validate import validate_render
from spatialize.binaural import binaural_render, woodworth_itd_us

SAMPLE_RATE = 48_000
DURATION_S = 4.0
TOLERANCE_DEG = 5.0
ITD_TOLERANCE_US = 30.0


def config(**overrides) -> BenchmarkConfig:
    base = dict(
        benchmark_name="StereoMusicQA-binaural",
        version="0.3.0-binaural",
        seed=2026,
        sample_rate=SAMPLE_RATE,
        clip_duration_seconds=DURATION_S,
        angles_degrees=(-45.0, 0.0, 45.0),
        azimuth_tolerance_degrees=TOLERANCE_DEG,
        minimum_rms_db=-45.0,
        maximum_peak=0.99,
        source_root=Path("data/URMP"),
        output_root=Path("artifacts/binaural-test"),
        rendering="binaural",
    )
    base.update(overrides)
    return BenchmarkConfig(**base)  # type: ignore[arg-type]


def source(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(DURATION_S * SAMPLE_RATE)) / SAMPLE_RATE
    tone = sum(
        0.3 / (n + 1) * np.sin(2 * np.pi * 220.0 * (n + 1) * time + n)
        for n in range(6)
    )
    return 0.6 * (tone + 0.25 * rng.standard_normal(time.size)) / 2.0


def plan(angle_deg: float, **overrides) -> RenderPlan:
    base = dict(
        render_id=f"bin_{angle_deg:+g}",
        stem_id="AuSep_1_vn_01_Test",
        piece_id="01_Test",
        split="test",
        instrument="violin",
        instrument_code="vn",
        source_path=Path("unused.wav"),
        start_seconds=0.0,
        duration_seconds=DURATION_S,
        sample_rate=SAMPLE_RATE,
        regime="binaural",
        azimuth_deg=angle_deg,
        seed=1,
    )
    base.update(overrides)
    return RenderPlan(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("angle", [-75.0, -45.0, -15.0, 15.0, 45.0, 75.0])
def test_a_planted_binaural_angle_comes_back(angle: float) -> None:
    stereo = binaural_render(source(), SAMPLE_RATE, angle)
    evidence = measure_render(stereo, SAMPLE_RATE)

    assert abs(evidence.binaural_azimuth_deg - angle) <= TOLERANCE_DEG
    assert abs(evidence.itd_us - woodworth_itd_us(angle)) <= ITD_TOLERANCE_US


def test_the_tangent_law_reading_is_not_the_answer_here() -> None:
    angle = 75.0
    evidence = measure_render(binaural_render(source(), SAMPLE_RATE, angle), SAMPLE_RATE)
    assert abs(evidence.binaural_azimuth_deg - angle) <= TOLERANCE_DEG

    assert abs(evidence.estimated_azimuth_deg) <= 30.0
    assert abs(evidence.estimated_azimuth_deg - angle) > TOLERANCE_DEG


@pytest.mark.parametrize("angle", [-60.0, -20.0, 20.0, 60.0])
def test_render_audio_dispatches_on_the_plan_regime(angle: float, tmp_path) -> None:
    import soundfile as sf

    stem = tmp_path / "stem.wav"
    sf.write(stem, source(), SAMPLE_RATE)
    stereo = render_audio(plan(angle, source_path=stem))
    assert stereo.shape[1] == 2
    evidence = measure_render(stereo, SAMPLE_RATE)
    assert abs(evidence.itd_us - woodworth_itd_us(angle)) <= ITD_TOLERANCE_US


@pytest.mark.parametrize("angle", [-60.0, -30.0, 30.0, 60.0])
def test_a_good_binaural_render_is_accepted(angle: float) -> None:
    stereo = binaural_render(source(), SAMPLE_RATE, angle)
    evidence = measure_render(stereo, SAMPLE_RATE)
    assert validate_render(stereo, plan(angle), evidence, config()) == []


def test_an_amplitude_pan_labelled_binaural_is_rejected() -> None:
    from spatialize.amplitude import render_amplitude_pan

    stereo = render_amplitude_pan(source(), 25.0)
    evidence = measure_render(stereo, SAMPLE_RATE)
    failures = validate_render(stereo, plan(60.0), evidence, config())
    assert "binaural_itd_recovery_failed" in failures


def test_a_render_at_the_wrong_angle_is_rejected() -> None:
    stereo = binaural_render(source(), SAMPLE_RATE, 60.0)
    evidence = measure_render(stereo, SAMPLE_RATE)
    failures = validate_render(stereo, plan(20.0), evidence, config())
    assert failures
    assert any("binaural" in failure for failure in failures)


def test_a_centred_binaural_item_is_not_required_to_read_as_delayed() -> None:
    stereo = binaural_render(source(), SAMPLE_RATE, 0.0)
    evidence = measure_render(stereo, SAMPLE_RATE)
    failures = validate_render(stereo, plan(0.0), evidence, config())
    assert "binaural_render_does_not_read_as_delayed" not in failures


def test_the_question_answers_with_the_planted_angle() -> None:
    assert answerable_azimuth_deg(plan(63.0)) == 63.0
    assert answerable_azimuth_deg(plan(-8.5)) == -8.5


def test_binaural_configs_reach_past_the_panning_law_limit() -> None:
    wide = config(angles_degrees=(-80.0, 0.0, 80.0))
    assert wide.maximum_abs_angle_deg == 90.0
    with pytest.raises(ValueError, match="strictly inside"):
        config(angles_degrees=(-95.0, 0.0, 95.0))


def test_amplitude_configs_keep_their_thirty_degree_limit() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        config(rendering="amplitude", angles_degrees=(-45.0, 0.0, 45.0))


def test_processing_recipes_are_refused_for_binaural_configs() -> None:
    with pytest.raises(ValueError, match="not supported with binaural"):
        config(
            processing_recipes=(
                (ProcessingStep(kind="midside", params={"width": 1.2}),),
            )
        )


def test_binaural_items_collapse_to_a_lookup_table_too() -> None:
    angle = 45.0
    measured = [
        measure_render(
            binaural_render(source(seed), SAMPLE_RATE, angle), SAMPLE_RATE
        ).itd_us
        for seed in range(4)
    ]
    assert len({round(value, 3) for value in measured}) == 1
    for value in measured:
        assert abs(value - woodworth_itd_us(angle)) <= ITD_TOLERANCE_US
