from __future__ import annotations

import pytest

from eval.generalization import (
    ON_GRID_EPSILON_DEG,
    choose_eval_angles,
    discriminability,
    on_grid_rate,
    snap_to_trained,
    trained_answer_angles,
)

GRID = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
NAIVE_MIDPOINTS = (-22.5, -17.5, -12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5, 22.5)


@pytest.fixture
def trained() -> tuple[float, ...]:
    return trained_answer_angles(GRID)


def test_the_training_split_holds_twenty_one_distinct_answers(trained) -> None:
    assert len(trained) == 21
    assert set(GRID) <= set(trained)


def test_the_mid_side_angles_are_trained_too(trained) -> None:
    assert any(abs(angle - 17.8246) < 1e-3 for angle in trained)
    assert any(abs(angle - 11.9471) < 1e-3 for angle in trained)


def test_a_lookup_model_answers_with_the_nearest_thing_it_knows(trained) -> None:
    assert snap_to_trained(2.5, trained) in (0.0, 5.0)
    assert snap_to_trained(24.9, trained) == 25.0

    assert snap_to_trained(15.0, trained) == 15.0


def test_on_grid_rate_ignores_unanswered_items(trained) -> None:
    assert on_grid_rate([25.0, None, None], trained) == 1.0
    assert on_grid_rate([None, None], trained) == 0.0


def test_on_grid_rate_counts_only_answers_near_a_trained_angle(trained) -> None:
    assert on_grid_rate([25.0, 20.0], trained) == 1.0
    assert on_grid_rate([2.5, 8.0], trained) == 0.0
    assert on_grid_rate([25.0, 2.5], trained) == 0.5


def test_tolerance_cannot_detect_memorisation_anywhere_in_range(trained) -> None:
    for angles in (NAIVE_MIDPOINTS, choose_eval_angles(trained, 10)):
        result = discriminability(angles, trained)
        assert result.memorises.within_tolerance == 1.0
        assert result.computes.within_tolerance == 1.0
        assert not result.separates_on_tolerance


def test_the_error_and_grid_metrics_do_separate_them(trained) -> None:
    result = discriminability(choose_eval_angles(trained, 10), trained)
    assert result.separates_on_error
    assert result.separates_on_grid_rate
    assert result.memorises.on_grid_rate == 1.0
    assert result.computes.on_grid_rate == 0.0


def test_the_naive_midpoints_are_the_trap_this_module_exists_for(trained) -> None:
    result = discriminability(NAIVE_MIDPOINTS, trained)
    assert result.computes.on_grid_rate == pytest.approx(0.4)


def test_the_chosen_angles_are_clear_of_every_trained_answer(trained) -> None:
    chosen = choose_eval_angles(trained, 10)
    assert len(chosen) == 10
    for angle in chosen:
        nearest = min(abs(angle - candidate) for candidate in trained)
        assert nearest > ON_GRID_EPSILON_DEG


def test_the_chosen_angles_are_balanced_across_the_two_sides(trained) -> None:
    chosen = choose_eval_angles(trained, 10)
    assert sum(1 for angle in chosen if angle < 0) == 5
    assert sorted(-angle for angle in chosen if angle < 0) == sorted(
        angle for angle in chosen if angle > 0
    )


def test_the_chosen_angles_stay_inside_the_panning_law(trained) -> None:
    for angle in choose_eval_angles(trained, 10):
        assert abs(angle) < 30.0


def _config(**overrides):
    from pathlib import Path

    from benchmark.config import BenchmarkConfig

    base = dict(
        benchmark_name="StereoMusicQA",
        version="0.2.0",
        seed=2026,
        sample_rate=48_000,
        clip_duration_seconds=6.0,
        angles_degrees=GRID,
        azimuth_tolerance_degrees=5.0,
        minimum_rms_db=-45.0,
        maximum_peak=0.99,
        source_root=Path("data/URMP"),
        output_root=Path("artifacts/x"),
    )
    base.update(overrides)


    return BenchmarkConfig(**base)  # type: ignore[arg-type]


def test_the_linter_reproduces_the_twenty_one_question_finding() -> None:
    from benchmark.schemas import ProcessingStep

    from eval.generalization import distinct_answer_angles

    v02 = _config(
        processing_recipes=(
            (ProcessingStep(kind="compression", params={"threshold_db": -20.0, "ratio": 4.0}),),
            (ProcessingStep(kind="eq", params={"center_hz": 1000.0, "gain_db": 6.0, "q": 1.0}),),
            (ProcessingStep(kind="midside", params={"width": 1.2}),),
            (ProcessingStep(kind="delay_widening", params={"delay_us": 300.0}),),
        )
    )
    assert len(distinct_answer_angles(v02)) == 21


def test_processing_that_does_not_move_the_angle_adds_no_questions() -> None:
    from benchmark.schemas import ProcessingStep

    from eval.generalization import distinct_answer_angles

    padded = _config(
        processing_recipes=(
            (ProcessingStep(kind="compression", params={"threshold_db": -20.0, "ratio": 4.0}),),
            (ProcessingStep(kind="eq", params={"center_hz": 1000.0, "gain_db": 6.0, "q": 1.0}),),
        )
    )
    assert len(distinct_answer_angles(padded)) == len(GRID)


def test_a_binaural_config_answers_with_its_planted_angles() -> None:
    from eval.generalization import distinct_answer_angles

    binaural = _config(
        rendering="binaural", angles_degrees=(-75.0, -45.0, 45.0, 75.0)
    )
    assert distinct_answer_angles(binaural) == (-75.0, -45.0, 45.0, 75.0)


def test_the_linter_warns_when_tolerance_swallows_the_grid() -> None:
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "lint_config", _Path(__file__).resolve().parents[1] / "scripts/lint_config.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    warnings = module.report(_Path("x.json"), _config())
    assert any("within the 5 deg tolerance" in w for w in warnings)


    sparse = _config(angles_degrees=tuple(float(a) for a in range(-25, 26, 25)))
    assert not any("within the" in w for w in module.report(_Path("y.json"), sparse))
