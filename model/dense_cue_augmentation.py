from __future__ import annotations

from dataclasses import dataclass

from benchmark.schemas import CueMeasurements
from cues.level import SPEAKER_ANGLE_DEG, ild_db_for_azimuth
from model.finetune_data import SFTExample, faithful_derivation
from model.qwen_backbone import describe_cues


MAX_SAMPLED_ANGLE_DEG = 29.5


OFFGRID_EVAL_ANGLES_DEG = (
    -21.8, -16.4, -13.5, -8.0, -2.5, 2.5, 8.0, 13.5, 16.4, 21.8,
)


EXCLUSION_MARGIN_DEG = 0.2

INSTRUMENTS = (
    "violin", "viola", "cello", "double bass", "flute", "oboe",
    "clarinet", "saxophone", "bassoon", "trumpet", "horn", "trombone", "tuba",
)


def _synthetic_evidence(azimuth_deg: float) -> CueMeasurements:
    ild_db = ild_db_for_azimuth(azimuth_deg)
    return CueMeasurements(
        ild_db=ild_db,
        high_frequency_ild_db=ild_db,
        ild_flatness_db=0.0,
        itd_us=0.0,
        itd_peak_strength=0.99,
        coherence=1.0,
        estimated_azimuth_deg=azimuth_deg,
        estimated_regime="amplitude",
        active_bins=600,
    )


@dataclass(frozen=True)
class DenseAugmentationConfig:
    count: int
    seed: int = 2026
    max_angle_deg: float = MAX_SAMPLED_ANGLE_DEG
    excluded_angles_deg: tuple[float, ...] = OFFGRID_EVAL_ANGLES_DEG
    exclusion_margin_deg: float = EXCLUSION_MARGIN_DEG


def _sample_angle_excluding(rng, config: "DenseAugmentationConfig") -> float:
    for _ in range(10_000):
        azimuth_deg = rng.uniform(-config.max_angle_deg, config.max_angle_deg)
        if all(
            abs(azimuth_deg - excluded) > config.exclusion_margin_deg
            for excluded in config.excluded_angles_deg
        ):
            return azimuth_deg
    raise RuntimeError(
        "could not sample an angle outside the excluded set after 10,000 "
        "tries -- excluded_angles_deg/exclusion_margin_deg likely leaves "
        "too little of [-max_angle_deg, max_angle_deg] open"
    )


def build_dense_synthetic_examples(
    audio_paths: list[str], config: DenseAugmentationConfig
) -> list[SFTExample]:
    if not audio_paths:
        raise ValueError("audio_paths must not be empty")
    if config.count <= 0:
        raise ValueError("config.count must be positive")
    if not 0.0 < config.max_angle_deg < SPEAKER_ANGLE_DEG:
        raise ValueError(f"max_angle_deg must be in (0, {SPEAKER_ANGLE_DEG:g})")

    import random

    rng = random.Random(config.seed)
    examples: list[SFTExample] = []
    for index in range(config.count):
        azimuth_deg = _sample_angle_excluding(rng, config)
        instrument = rng.choice(INSTRUMENTS)
        audio_path = audio_paths[index % len(audio_paths)]
        evidence = _synthetic_evidence(azimuth_deg)

        prompt = (
            describe_cues(evidence, instrument=instrument, f0_hz=None)
            + "\n"
            + f"Where is the {instrument} positioned in the stereo image? "
            "Return its side and azimuth in degrees."
        )
        target = faithful_derivation(evidence.ild_db, azimuth_deg, instrument)

        examples.append(
            SFTExample(
                item_id=f"synthetic_dense_{index:06d}_absolute",
                audio_path=audio_path,
                prompt=prompt,
                target=target,
                condition="cues",
            )
        )
    return examples


def round_trip_error_deg(azimuth_deg: float) -> float:
    from cues.level import azimuth_from_ild_db

    ild_db = ild_db_for_azimuth(azimuth_deg)
    return abs(azimuth_deg - azimuth_from_ild_db(ild_db))
