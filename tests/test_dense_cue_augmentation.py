from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from cues.level import SPEAKER_ANGLE_DEG, azimuth_from_ild_db, ild_db_for_azimuth  # noqa: E402
from model.dense_cue_augmentation import (  # noqa: E402
    OFFGRID_EVAL_ANGLES_DEG,
    DenseAugmentationConfig,
    build_dense_synthetic_examples,
    round_trip_error_deg,
)

REAL_AUDIO_PATHS = [f"audio/train/stem_{index}.wav" for index in range(5)]


def test_round_trip_is_exact_across_the_sampled_range():
    for azimuth in (-29.49, -20.0, -5.5, -0.001, 0.0, 0.001, 5.5, 20.0, 29.49):
        assert round_trip_error_deg(azimuth) < 1e-6


def test_ild_db_for_azimuth_rejects_the_pole_itself():
    with pytest.raises(ValueError):
        ild_db_for_azimuth(SPEAKER_ANGLE_DEG)


def test_generates_the_requested_count_with_unique_item_ids():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=500, seed=1)
    )
    assert len(examples) == 500
    assert len({e.item_id for e in examples}) == 500


def test_sampled_cues_are_overwhelmingly_distinct():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=500, seed=2)
    )
    ild_values = {_stated_ild_db(e.target) for e in examples}
    assert len(ild_values) > 250, (
        f"only {len(ild_values)} distinct displayed cues from 500 draws -- "
        "barely better than the 21-row lookup table this exists to fix"
    )


def test_stays_inside_the_configured_angle_range():
    config = DenseAugmentationConfig(count=300, seed=3, max_angle_deg=10.0)
    examples = build_dense_synthetic_examples(REAL_AUDIO_PATHS, config)
    for example in examples:
        azimuth = _stated_azimuth_deg(example.target)
        assert abs(azimuth) <= 10.0 + 1e-6


def test_reuses_real_audio_paths_cycled_not_matched():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=12, seed=4)
    )
    used = {e.audio_path for e in examples}
    assert used == set(REAL_AUDIO_PATHS)
    assert examples[0].audio_path == REAL_AUDIO_PATHS[0]
    assert examples[5].audio_path == REAL_AUDIO_PATHS[5 % len(REAL_AUDIO_PATHS)]


def test_the_target_derivation_is_internally_consistent_not_fabricated():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=200, seed=5)
    )
    for example in examples:
        stated_ild = _stated_ild_db(example.target)
        stated_r = _stated_r(example.target)
        stated_azimuth = _stated_azimuth_deg(example.target)


        ild_implied_by_r = 20.0 * math.log10(stated_r)
        assert abs(ild_implied_by_r - stated_ild) < 0.5, (
            f"target states ild_db={stated_ild} but its own r={stated_r} "
            f"implies {ild_implied_by_r:.2f} -- this is the fabrication bug"
        )
        recomputed_azimuth = azimuth_from_ild_db(stated_ild)
        assert stated_azimuth == pytest.approx(recomputed_azimuth, abs=0.15)


def test_side_word_matches_the_sign_of_the_azimuth():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=200, seed=6)
    )
    for example in examples:
        azimuth = _stated_azimuth_deg(example.target)
        if azimuth > 0:
            assert "to the left" in example.target
        elif azimuth < 0:
            assert "to the right" in example.target


def test_prompt_omits_the_fundamental_frequency_line():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=5, seed=7)
    )
    for example in examples:
        assert "fundamental frequency" not in example.prompt


def test_rejects_empty_audio_paths():
    with pytest.raises(ValueError):
        build_dense_synthetic_examples([], DenseAugmentationConfig(count=10))


def test_rejects_nonpositive_count():
    with pytest.raises(ValueError):
        build_dense_synthetic_examples(REAL_AUDIO_PATHS, DenseAugmentationConfig(count=0))


def test_rejects_an_out_of_range_max_angle():
    with pytest.raises(ValueError):
        build_dense_synthetic_examples(
            REAL_AUDIO_PATHS, DenseAugmentationConfig(count=10, max_angle_deg=30.0)
        )


def test_no_sampled_angle_lands_near_a_held_out_eval_angle():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=3000, seed=2026)
    )
    for example in examples:
        azimuth = _stated_azimuth_deg(example.target)
        closest = min(abs(azimuth - angle) for angle in OFFGRID_EVAL_ANGLES_DEG)
        assert closest > 0.05, (
            f"{example.item_id} states {azimuth}deg, within display-rounding "
            f"distance of a held-out off-grid eval angle"
        )


def test_default_exclusion_still_leaves_the_full_count_reachable():
    examples = build_dense_synthetic_examples(
        REAL_AUDIO_PATHS, DenseAugmentationConfig(count=2000, seed=8)
    )
    assert len(examples) == 2000


def test_exclusion_zones_are_configurable():
    config = DenseAugmentationConfig(
        count=1,
        max_angle_deg=5.0,
        excluded_angles_deg=(0.0,),
        exclusion_margin_deg=10.0,
    )
    with pytest.raises(RuntimeError):
        build_dense_synthetic_examples(REAL_AUDIO_PATHS, config)


def _stated_ild_db(target: str) -> float:
    match = re.search(r"level difference is (-?[\d.]+) dB", target)
    assert match, target
    return float(match.group(1))


def _stated_r(target: str) -> float:
    match = re.search(r"r = 10\^\([^)]*\) = ([\d.]+)", target)
    assert match, target
    return float(match.group(1))


def _stated_azimuth_deg(target: str) -> float:
    match = re.search(r"azimuth = arctan\([^)]*\) = (-?[\d.]+) degrees", target)
    assert match, target
    return float(match.group(1))
