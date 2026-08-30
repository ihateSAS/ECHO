from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmark.build import build_benchmark
from benchmark.config import BenchmarkConfig, load_config
from benchmark.inventory import build_inventory
from benchmark.measure import measure_render
from benchmark.process import (
    apply_processing_step,
    expected_after_processing,
    processing_label,
)
from benchmark.questions import answerable_azimuth_deg, side_for_angle
from benchmark.render import make_render_plans, render_audio
from benchmark.schemas import ProcessingStep, RenderPlan
from benchmark.splits import assign_piece_splits
from benchmark.validate import validate_render
from spatialize.amplitude import render_amplitude_pan

SAMPLE_RATE = 48_000


def _write_mini_urmp(root: Path, pieces: int = 7) -> None:
    root.mkdir(parents=True)
    samples = 12_000
    for number in range(1, pieces + 1):
        rng = np.random.default_rng(number)
        mono = 0.1 * rng.standard_normal(samples)
        sf.write(
            root / f"AuSep_1_vc_{number:02d}_Piece{number}.wav",
            mono,
            SAMPLE_RATE,
            subtype="PCM_24",
        )


def _config(source_root: Path, output_root: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        benchmark_name="StereoMusicQA",
        version="0.1.0-test",
        seed=2026,
        sample_rate=SAMPLE_RATE,
        clip_duration_seconds=0.2,
        angles_degrees=(-15.0, 0.0, 15.0),
        azimuth_tolerance_degrees=5.0,
        minimum_rms_db=-45.0,
        maximum_peak=0.99,
        source_root=source_root,
        output_root=output_root,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_inventory_uses_only_isolated_mono_stems(tmp_path: Path):
    source = tmp_path / "URMP"
    _write_mini_urmp(source, pieces=1)
    sf.write(source / "AuMix_01_Piece1.wav", np.zeros((100, 2)), SAMPLE_RATE)

    inventory = build_inventory(source)

    assert len(inventory) == 1
    assert inventory[0].stem_id.startswith("AuSep_")
    assert inventory[0].instrument_name == "cello"


def test_splits_are_deterministic_and_piece_level(tmp_path: Path):
    source = tmp_path / "URMP"
    _write_mini_urmp(source)
    inventory = build_inventory(source)

    first = assign_piece_splits(inventory, seed=8)
    second = assign_piece_splits(inventory, seed=8)

    assert first == second
    assert set(first.values()) == {"train", "dev", "test"}
    assert len(first) == len({record.piece_id for record in inventory})


@pytest.mark.parametrize(
    "angle,expected", [(-15.0, "right"), (0.0, "center"), (15.0, "left")]
)
def test_side_convention(angle: float, expected: str):
    assert side_for_angle(angle) == expected


def test_builder_recovers_planted_angles_and_hides_test_labels(tmp_path: Path):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_urmp(source)

    summary = build_benchmark(_config(source, output))

    assert summary.render_plans == 21
    assert summary.accepted_items == 21
    assert summary.rejected_items == 0
    assert _read_jsonl(output / "rejected_items.jsonl") == []

    private = _read_jsonl(output / "test_private_labels.jsonl")
    public = _read_jsonl(output / "test_public.jsonl")
    assert private
    assert len(private) == len(public)
    assert all("answer_azimuth_deg" not in item for item in public)
    assert all("measured_evidence" not in item for item in public)

    private_by_id = {item["item_id"]: item for item in private}
    for item in public:
        label = private_by_id[item["item_id"]]
        error = abs(
            label["answer_azimuth_deg"]
            - label["measured_evidence"]["estimated_azimuth_deg"]
        )
        assert error < 5.0
        assert (output / item["audio_path"]).exists()


def test_build_is_deterministic(tmp_path: Path):
    source = tmp_path / "URMP"
    _write_mini_urmp(source)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    build_benchmark(_config(source, first_root))
    build_benchmark(_config(source, second_root))

    for filename in (
        "train.jsonl",
        "dev.jsonl",
        "test_public.jsonl",
        "test_private_labels.jsonl",
        "rejected_items.jsonl",
    ):
        assert (first_root / filename).read_bytes() == (
            second_root / filename
        ).read_bytes()


def _write_config_file(path: Path, source: Path, output: Path, recipes: list) -> Path:
    path.write_text(
        json.dumps(
            {
                "benchmark_name": "StereoMusicQA",
                "version": "0.1.0-test",
                "seed": 2026,
                "sample_rate": SAMPLE_RATE,
                "clip_duration_seconds": 0.2,
                "angles_degrees": [-15, 0, 15],
                "azimuth_tolerance_degrees": 5.0,
                "minimum_rms_db": -45.0,
                "maximum_peak": 0.99,
                "source_root": str(source),
                "output_root": str(output),
                "processing_recipes": recipes,
            }
        )
    )
    return path


def test_a_config_file_can_ask_for_reverb_at_a_known_rt60(tmp_path: Path):
    path = _write_config_file(
        tmp_path / "config.json",
        tmp_path / "URMP",
        tmp_path / "build",
        [
            [{"kind": "reverb", "params": {"rt60_s": 0.6, "wet": 0.2}}],
            [{"kind": "midside", "params": {"width": 1.5}}],
        ],
    )
    config = load_config(path)

    assert len(config.processing_recipes) == 2
    (reverb,), (width,) = config.processing_recipes
    assert reverb.kind == "reverb"
    assert reverb.params == {"rt60_s": 0.6, "wet": 0.2}
    assert width.kind == "midside"
    assert width.params == {"width": 1.5}


def test_a_config_without_recipes_still_loads_as_dry_only(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_name": "StereoMusicQA",
                "version": "0.1.0-test",
                "seed": 2026,
                "sample_rate": SAMPLE_RATE,
                "clip_duration_seconds": 0.2,
                "angles_degrees": [0],
                "azimuth_tolerance_degrees": 5.0,
                "minimum_rms_db": -45.0,
                "maximum_peak": 0.99,
                "source_root": str(tmp_path),
                "output_root": str(tmp_path),
            }
        )
    )
    assert load_config(path).processing_recipes == ()


def test_the_shipped_v0_1_config_requests_no_processing():
    config = load_config(
        Path(__file__).parent.parent / "benchmark/configs/stereomusicqa_v0.1.json"
    )
    assert config.processing_recipes == ()


def test_a_recipe_adds_renders_without_disturbing_the_dry_ones(tmp_path: Path):
    source = tmp_path / "URMP"
    _write_mini_urmp(source, pieces=2)
    inventory = build_inventory(source)
    assignments = assign_piece_splits(inventory, seed=2026)

    dry_config = _config(source, tmp_path / "dry")
    processed_config = replace(
        dry_config,
        processing_recipes=((ProcessingStep("midside", {"width": 1.5}),),),
    )

    dry_plans = make_render_plans(inventory, assignments, dry_config)
    processed_plans = make_render_plans(inventory, assignments, processed_config)

    assert len(processed_plans) == 2 * len(dry_plans)
    assert len({plan.render_id for plan in processed_plans}) == len(processed_plans)

    dry_ids = {plan.render_id for plan in dry_plans}
    assert dry_ids <= {plan.render_id for plan in processed_plans}
    for plan in processed_plans:
        if plan.render_id in dry_ids:
            assert plan.processing == ()
        else:
            assert plan.processing == (ProcessingStep("midside", {"width": 1.5}),)


def test_the_manifest_records_the_processing_parameters_next_to_the_azimuth(
    tmp_path: Path,
):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_urmp(source)

    config = replace(
        _config(source, output),
        processing_recipes=(
            (ProcessingStep("reverb", {"rt60_s": 0.4, "wet": 0.2}),),
            (ProcessingStep("midside", {"width": 1.5}),),
        ),
    )
    summary = build_benchmark(config)
    assert summary.render_plans == 63

    accepted = [
        item["render"]
        for name in ("train.jsonl", "dev.jsonl")
        for item in _read_jsonl(output / name)
    ] + [label["render"] for label in _read_jsonl(output / "test_private_labels.jsonl")]
    rejected = _read_jsonl(output / "rejected_items.jsonl")


    assert len(accepted) == 42
    assert len(rejected) == 21
    assert {failure for item in rejected for failure in item["failures"]} == {
        "azimuth_not_recoverable_after_processing"
    }

    by_kind: dict[str, list[dict]] = {}
    for record in accepted + [item["render"] for item in rejected]:
        assert "azimuth_deg" in record
        for step in record["processing"]:
            by_kind.setdefault(step["kind"], []).append(step)

    assert len(by_kind["reverb"]) == 21
    assert len(by_kind["midside"]) == 21
    assert by_kind["reverb"][0]["params"] == {"rt60_s": 0.4, "wet": 0.2}
    assert by_kind["midside"][0]["params"] == {"width": 1.5}


    assert sum(1 for record in accepted if record["processing"] == []) == 21


def test_a_reverb_item_is_rejected_because_its_azimuth_cannot_be_recovered():
    mono = 0.2 * np.random.default_rng(11).standard_normal(SAMPLE_RATE)
    planted = 25.0
    panned = render_amplitude_pan(mono, planted)
    reverberant = apply_processing_step(
        panned, SAMPLE_RATE, ProcessingStep("reverb", {"rt60_s": 0.6, "wet": 0.3})
    )

    plan = RenderPlan(
        render_id="pinned",
        stem_id="stem",
        piece_id="piece",
        split="train",
        instrument="cello",
        instrument_code="vc",
        source_path=Path("unused.wav"),
        start_seconds=0.0,
        duration_seconds=1.0,
        sample_rate=SAMPLE_RATE,
        regime="amplitude",
        azimuth_deg=planted,
        seed=0,
        processing=(ProcessingStep("reverb", {"rt60_s": 0.6, "wet": 0.3}),),
    )
    config = _config(Path("unused"), Path("unused"))
    evidence = measure_render(reverberant, SAMPLE_RATE)

    assert (
        abs(evidence.estimated_azimuth_deg - planted)
        > config.azimuth_tolerance_degrees
    )
    assert validate_render(reverberant, plan, evidence, config) == [
        "azimuth_not_recoverable_after_processing"
    ]

    with pytest.raises(ValueError, match="no recoverable azimuth"):
        answerable_azimuth_deg(plan)


def test_a_processed_render_actually_sounds_different(tmp_path: Path):
    source = tmp_path / "URMP"
    _write_mini_urmp(source, pieces=1)
    inventory = build_inventory(source)
    assignments = assign_piece_splits(inventory, seed=2026)
    config = replace(
        _config(source, tmp_path / "build"),
        processing_recipes=((ProcessingStep("midside", {"width": 0.0}),),),
    )
    plans = make_render_plans(inventory, assignments, config)

    dry = next(plan for plan in plans if not plan.processing and plan.azimuth_deg == 15.0)
    processed = next(
        plan for plan in plans if plan.processing and plan.azimuth_deg == 15.0
    )


    dry_audio = render_audio(dry)
    processed_audio = render_audio(processed)
    assert abs(dry_audio[:, 0] - dry_audio[:, 1]).max() > 1e-6
    np.testing.assert_allclose(processed_audio[:, 0], processed_audio[:, 1])


def test_an_unknown_processing_kind_is_refused_rather_than_ignored():
    stereo = np.zeros((1000, 2))
    step = ProcessingStep("chorus", {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown processing kind"):
        apply_processing_step(stereo, SAMPLE_RATE, step)


def test_a_recipe_label_is_stable_and_collision_free():
    reverb = (ProcessingStep("reverb", {"rt60_s": 0.6, "wet": 0.2}),)
    width = (ProcessingStep("midside", {"width": 1.5}),)

    assert processing_label(()) == "dry"
    assert processing_label(reverb) == processing_label(reverb)
    assert processing_label(reverb) != processing_label(width)

    assert set(processing_label(reverb + width)) <= set(
        "abcdefghijklmnopqrstuvwxyz0123456789_"
    )


def _measured_azimuth(azimuth_deg: float, steps: tuple[ProcessingStep, ...]) -> float:
    mono = 0.2 * np.random.default_rng(5).standard_normal(SAMPLE_RATE)
    stereo = render_amplitude_pan(mono, azimuth_deg)
    for step in steps:
        stereo = apply_processing_step(stereo, SAMPLE_RATE, step)
    return measure_render(stereo, SAMPLE_RATE).estimated_azimuth_deg


@pytest.mark.parametrize(
    "step",
    [
        ProcessingStep("compression", {"threshold_db": -18.0, "ratio": 4.0}),
        ProcessingStep("eq", {"center_hz": 800.0, "gain_db": 6.0}),
    ],
)
def test_compression_and_eq_leave_the_answer_where_it_was(step: ProcessingStep):
    expected = expected_after_processing(20.0, (step,))
    assert expected.azimuth_deg == 20.0
    assert expected.coherence_stays_high
    assert _measured_azimuth(20.0, (step,)) == pytest.approx(20.0, abs=0.5)


@pytest.mark.parametrize("width,planted", [(1.5, 10.0), (1.3, 15.0), (0.5, 25.0)])
def test_mid_side_moves_the_answer_by_a_computable_amount(
    width: float, planted: float
):
    step = ProcessingStep("midside", {"width": width})
    expected = expected_after_processing(planted, (step,))

    assert expected.azimuth_deg is not None
    assert expected.azimuth_deg != planted
    assert _measured_azimuth(planted, (step,)) == pytest.approx(
        expected.azimuth_deg, abs=0.5
    )


def test_a_width_that_inverts_polarity_has_no_answer_at_all():
    expected = expected_after_processing(25.0, (ProcessingStep("midside", {"width": 1.5}),))

    assert expected.azimuth_deg is None
    assert any("inverts polarity" in note for note in expected.notes)


def test_delay_widening_installs_the_delay_it_was_asked_for():
    step = ProcessingStep("delay_widening", {"delay_us": 400.0})
    expected = expected_after_processing(10.0, (step,))

    assert expected.azimuth_deg == 10.0
    assert expected.itd_us == 400.0
    assert expected.regime == "delayed"

    mono = 0.2 * np.random.default_rng(6).standard_normal(SAMPLE_RATE)
    measured = measure_render(
        apply_processing_step(render_amplitude_pan(mono, 10.0), SAMPLE_RATE, step),
        SAMPLE_RATE,
    )
    assert measured.itd_us == pytest.approx(400.0, abs=30.0)
    assert measured.estimated_regime == "delayed"


def test_a_dry_reverb_send_is_not_treated_as_reverb():
    expected = expected_after_processing(
        20.0, (ProcessingStep("reverb", {"rt60_s": 0.6, "wet": 0.0}),)
    )
    assert expected.azimuth_deg == 20.0
    assert expected.coherence_stays_high


def test_a_chain_carries_the_answer_through_every_step_in_order():
    chain = (
        ProcessingStep("compression", {"threshold_db": -18.0, "ratio": 4.0}),
        ProcessingStep("midside", {"width": 1.3}),
        ProcessingStep("eq", {"center_hz": 500.0, "gain_db": -3.0}),
    )
    expected = expected_after_processing(10.0, chain)
    width_only = expected_after_processing(
        10.0, (ProcessingStep("midside", {"width": 1.3}),)
    )

    assert expected.azimuth_deg == width_only.azimuth_deg
    assert _measured_azimuth(10.0, chain) == pytest.approx(
        expected.azimuth_deg, abs=0.5
    )


def test_reverb_anywhere_in_a_chain_poisons_the_answer():
    expected = expected_after_processing(
        10.0,
        (
            ProcessingStep("midside", {"width": 1.2}),
            ProcessingStep("reverb", {"rt60_s": 0.6, "wet": 0.3}),
            ProcessingStep("compression", {"threshold_db": -18.0, "ratio": 4.0}),
        ),
    )
    assert expected.azimuth_deg is None
    assert expected.regime is None
    assert not expected.coherence_stays_high


def test_a_widened_item_is_answered_with_the_angle_it_actually_has(tmp_path: Path):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_urmp(source)
    config = replace(
        _config(source, output),
        angles_degrees=(15.0,),
        processing_recipes=((ProcessingStep("midside", {"width": 1.3}),),),
    )
    build_benchmark(config)

    labels = [
        item
        for name in ("train.jsonl", "dev.jsonl")
        for item in _read_jsonl(output / name)
    ] + _read_jsonl(output / "test_private_labels.jsonl")
    widened = [item for item in labels if item["render"]["processing"]]
    dry = [item for item in labels if not item["render"]["processing"]]
    assert widened and dry


    def answer_of(item: dict) -> float:
        if "answer_azimuth_deg" in item:
            return item["answer_azimuth_deg"]
        return item["question"]["answer_azimuth_deg"]

    for item in dry:
        assert answer_of(item) == pytest.approx(15.0)
    for item in widened:
        assert answer_of(item) == pytest.approx(19.205, abs=0.01)
        assert item["render"]["azimuth_deg"] == 15.0


ANGLE_IN_NAME = re.compile(r"_(p|neg)\d+")


def test_no_public_test_identifier_spells_out_the_angle(tmp_path: Path):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_urmp(source)
    build_benchmark(_config(source, output))

    public = _read_jsonl(output / "test_public.jsonl")
    assert public
    for item in public:
        assert not ANGLE_IN_NAME.search(item["item_id"]), item["item_id"]
        assert not ANGLE_IN_NAME.search(item["audio_path"]), item["audio_path"]
        assert item["item_id"].startswith("smq_test_")


        assert (output / item["audio_path"]).is_file()

    for name in (output / "audio" / "test").iterdir():
        assert not ANGLE_IN_NAME.search(name.name), name.name


def test_train_and_dev_keep_their_descriptive_identifiers(tmp_path: Path):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_urmp(source)
    build_benchmark(_config(source, output))

    train = _read_jsonl(output / "train.jsonl")
    assert train
    assert all(ANGLE_IN_NAME.search(item["item_id"]) for item in train)


def test_the_private_labels_are_the_only_way_back_to_the_angle(tmp_path: Path):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_urmp(source)
    build_benchmark(_config(source, output))

    public = _read_jsonl(output / "test_public.jsonl")
    private = {item["item_id"]: item for item in _read_jsonl(output / "test_private_labels.jsonl")}

    assert set(private) == {item["item_id"] for item in public}
    for item in public:
        label = private[item["item_id"]]
        assert "answer_azimuth_deg" in label
        assert ANGLE_IN_NAME.search(label["render"]["render_id"])


        assert item["item_id"] == f"{label['render']['public_id']}_absolute"


def test_the_opaque_numbering_is_stable_across_builds(tmp_path: Path):
    source = tmp_path / "URMP"
    _write_mini_urmp(source)
    first, second = tmp_path / "first", tmp_path / "second"
    build_benchmark(_config(source, first))
    build_benchmark(_config(source, second))

    assert (first / "test_public.jsonl").read_bytes() == (
        second / "test_public.jsonl"
    ).read_bytes()


def test_the_numbering_does_not_follow_the_angle_order(tmp_path: Path):
    source = tmp_path / "URMP"
    output = tmp_path / "build"
    _write_mini_urmp(source)
    config = replace(
        _config(source, output),
        angles_degrees=(-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    )
    build_benchmark(config)

    private = _read_jsonl(output / "test_private_labels.jsonl")
    by_id = sorted(private, key=lambda item: item["item_id"])
    angles = [item["answer_azimuth_deg"] for item in by_id]
    assert len(angles) >= 11
    assert angles != sorted(angles)
    assert angles != sorted(angles, reverse=True)
