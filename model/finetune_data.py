from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from cues.level import SPEAKER_ANGLE_DEG
from model.qwen_backbone import (
    DEFAULT_STEREO_QUESTION,
    STEREO_PREAMBLE,
    describe_cues,
)
from benchmark.questions import side_for_angle
from benchmark.schemas import CueMeasurements


MAX_GROUND_TRUTH_DRIFT_DEG = 1.0


@dataclass(frozen=True)
class SFTExample:
    item_id: str
    audio_path: str
    prompt: str
    target: str
    condition: str


def faithful_derivation(ild_db: float, azimuth_deg: float, instrument: str) -> str:
    displayed_ild_db = round(ild_db, 1)
    r = 10.0 ** (displayed_ild_db / 20.0)
    d = (r - 1.0) / (r + 1.0)
    computed_azimuth = math.degrees(
        math.atan(d * math.tan(math.radians(SPEAKER_ANGLE_DEG)))
    )
    drift = abs(computed_azimuth - azimuth_deg)
    if drift > MAX_GROUND_TRUTH_DRIFT_DEG:
        raise ValueError(
            f"self-consistent azimuth {computed_azimuth:.2f}deg is {drift:.2f}deg "
            f"from ground truth {azimuth_deg:.2f}deg for ild_db={ild_db:.4f} -- "
            "ild_db and azimuth_deg likely do not describe the same render"
        )
    side = side_for_angle(computed_azimuth)
    louder = "left" if displayed_ild_db > 0 else "right"
    return (
        f"The inter-channel level difference is {displayed_ild_db:.1f} dB, so the "
        f"{louder} channel is louder. Converting with the panning law: "
        f"r = 10^({displayed_ild_db:.1f}/20) = {r:.3f}; "
        f"d = (r - 1)/(r + 1) = {d:.3f}; "
        f"azimuth = arctan(d * tan {SPEAKER_ANGLE_DEG:.0f}deg) = {computed_azimuth:.1f} degrees. "
        f"The {instrument} is at {abs(computed_azimuth):.1f} degrees to the {side}."
    )


def audio_only_target(azimuth_deg: float, side: str, instrument: str) -> str:
    if side == "center":
        return f"The {instrument} is centred, at 0 degrees."
    return f"The {instrument} is at {abs(azimuth_deg):.1f} degrees to the {side}."


def _cues_from_record(evidence: dict) -> CueMeasurements:
    return CueMeasurements(
        **{k: evidence[k] for k in CueMeasurements.__dataclass_fields__ if k in evidence}
    )


def build_example(record: dict, condition: str) -> SFTExample:
    question = record["question"]
    instrument = question["target_instrument"]
    azimuth = float(question["answer_azimuth_deg"])
    side = question["answer_side"]

    if condition == "cues":
        evidence = _cues_from_record(record["measured_evidence"])
        f0 = evidence.f0_hz if evidence.f0_hz == evidence.f0_hz else None
        prompt = (
            describe_cues(evidence, instrument=instrument, f0_hz=f0)
            + "\n"
            + question["question"]
        )
        target = faithful_derivation(evidence.ild_db, azimuth, instrument)
    elif condition == "audio_only":
        prompt = f"{STEREO_PREAMBLE}\n{DEFAULT_STEREO_QUESTION}"
        target = audio_only_target(azimuth, side, instrument)
    else:
        raise ValueError(f"unknown condition {condition!r}; expected 'cues' or 'audio_only'")

    return SFTExample(
        item_id=record["item_id"],
        audio_path=record["audio_path"],
        prompt=prompt,
        target=target,
        condition=condition,
    )


def build_examples(manifest: Path, condition: str) -> list[SFTExample]:
    return [
        build_example(json.loads(line), condition)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_examples(examples: list[SFTExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        for ex in examples:
            file.write(
                json.dumps(
                    {
                        "item_id": ex.item_id,
                        "audio_path": ex.audio_path,
                        "prompt": ex.prompt,
                        "target": ex.target,
                        "condition": ex.condition,
                    }
                )
                + "\n"
            )
