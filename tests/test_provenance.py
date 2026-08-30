from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.config import BenchmarkConfig
from benchmark.provenance import (
    BUILD_ID_LENGTH,
    BuildMismatch,
    build_id,
    check_build_ids,
    fingerprint_fields,
)
from benchmark.schemas import ProcessingStep
from eval.score import read_build_id


def make_config(**overrides) -> BenchmarkConfig:
    base = dict(
        benchmark_name="StereoMusicQA",
        version="0.2.0",
        seed=2026,
        sample_rate=48_000,
        clip_duration_seconds=6.0,
        angles_degrees=(-25.0, 0.0, 25.0),
        azimuth_tolerance_degrees=5.0,
        minimum_rms_db=-45.0,
        maximum_peak=0.99,
        source_root=Path("data/URMP"),
        output_root=Path("artifacts/x"),
    )
    base.update(overrides)
    return BenchmarkConfig(**base)  # type: ignore[arg-type]


def test_the_id_is_short_and_deterministic() -> None:
    config = make_config()
    assert build_id(config) == build_id(make_config())
    assert len(build_id(config)) == BUILD_ID_LENGTH


def test_changing_the_angles_changes_the_id() -> None:
    assert build_id(make_config()) != build_id(
        make_config(angles_degrees=(-22.5, 0.0, 22.5))
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 7),
        ("version", "0.3.0"),
        ("benchmark_name", "Other"),
        ("clip_duration_seconds", 4.0),
        ("sample_rate", 16_000),
        ("azimuth_tolerance_degrees", 2.5),
    ],
)
def test_anything_that_changes_the_questions_changes_the_id(field, value) -> None:
    assert build_id(make_config()) != build_id(make_config(**{field: value}))


def test_adding_a_processing_recipe_changes_the_id() -> None:
    widened = make_config(
        processing_recipes=(
            (ProcessingStep(kind="midside", params={"width": 1.2}),),
        )
    )
    assert build_id(make_config()) != build_id(widened)


def test_where_the_build_was_written_does_not_change_the_id() -> None:
    assert build_id(make_config()) == build_id(
        make_config(output_root=Path("/somewhere/else"))
    )


def test_the_fingerprint_is_json_serialisable() -> None:
    json.dumps(fingerprint_fields(make_config()), sort_keys=True)


def test_matching_ids_pass() -> None:
    check_build_ids("abc123", "abc123")


def test_conflicting_ids_are_refused() -> None:
    with pytest.raises(BuildMismatch, match="different benchmarks"):
        check_build_ids("abc123", "def456")


def test_the_refusal_names_both_files_and_the_way_out() -> None:
    with pytest.raises(BuildMismatch) as error:
        check_build_ids("aaa", "bbb", labels_name="labels.jsonl", answers_name="ans.jsonl")
    message = str(error.value)
    assert "labels.jsonl" in message and "ans.jsonl" in message
    assert "--allow-build-mismatch" in message


@pytest.mark.parametrize(
    "labels,answers",
    [(None, "abc"), ("abc", None), (None, None)],
)
def test_a_missing_id_is_tolerated(labels, answers) -> None:
    check_build_ids(labels, answers)


def test_read_build_id_finds_it(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"item_id": "a", "answer": "x", "build_id": "deadbeef"}) + "\n"
        + json.dumps({"item_id": "b", "answer": "y", "build_id": "deadbeef"}) + "\n"
    )
    assert read_build_id(path) == "deadbeef"


def test_read_build_id_returns_none_for_an_older_file(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"item_id": "a", "answer": "x"}) + "\n")
    assert read_build_id(path) is None


def test_read_build_id_handles_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text("")
    assert read_build_id(path) is None


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _labels(build: str | None) -> list[dict]:
    records = []
    for index, angle in enumerate((-25.0, 0.0, 25.0)):
        record = {
            "item_id": f"smq_test_{index:05d}_absolute",
            "answer_azimuth_deg": angle,
            "answer_side": "left" if angle > 0 else "right" if angle < 0 else "center",
            "render": {"instrument": "cello"},
        }
        if build is not None:
            record["build_id"] = build
        records.append(record)
    return records


def _answers(build: str | None) -> list[dict]:
    records = []
    for index in range(3):
        record = {"item_id": f"smq_test_{index:05d}_absolute", "answer": "centred, 0 degrees"}
        if build is not None:
            record["build_id"] = build
        records.append(record)
    return records


def _run_scorer(tmp_path: Path, labels_build, answers_build, *extra: str):
    import subprocess
    import sys

    labels = tmp_path / "labels.jsonl"
    answers = tmp_path / "answers.jsonl"
    _write(labels, _labels(labels_build))
    _write(answers, _answers(answers_build))
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo / "scripts/score_benchmark.py"),
         "--labels", str(labels), "--answers", str(answers), *extra],
        capture_output=True, text=True, cwd=repo,
    )


def test_the_scorer_refuses_a_mismatched_pair(tmp_path: Path) -> None:
    result = _run_scorer(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    assert result.returncode != 0
    assert "different benchmarks" in (result.stderr + result.stdout)


def test_the_scorer_accepts_a_matching_pair(tmp_path: Path) -> None:
    result = _run_scorer(tmp_path, "aaaaaaaaaaaa", "aaaaaaaaaaaa")
    assert result.returncode == 0, result.stderr


def test_the_escape_hatch_works_when_asked_for(tmp_path: Path) -> None:
    result = _run_scorer(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb", "--allow-build-mismatch")
    assert result.returncode == 0, result.stderr


def test_older_answers_without_an_id_still_score(tmp_path: Path) -> None:
    result = _run_scorer(tmp_path, "aaaaaaaaaaaa", None)
    assert result.returncode == 0, result.stderr
