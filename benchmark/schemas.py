from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

Split = Literal["train", "dev", "test"]


Regime = Literal["amplitude", "binaural"]
QuestionType = Literal["absolute_localization"]
MixtureQuestionType = Literal["mixture_localization"]
ProcessingKind = Literal["reverb", "compression", "eq", "midside", "delay_widening"]


@dataclass(frozen=True)
class ProcessingStep:
    kind: ProcessingKind
    params: dict[str, float]


@dataclass(frozen=True)
class StemRecord:
    stem_id: str
    piece_id: str
    instrument_code: str
    instrument_name: str
    path: Path
    sample_rate: int
    channels: int
    frames: int


@dataclass(frozen=True)
class RenderPlan:
    render_id: str
    stem_id: str
    piece_id: str
    split: Split
    instrument: str
    instrument_code: str
    source_path: Path
    start_seconds: float
    duration_seconds: float
    sample_rate: int
    regime: Regime
    azimuth_deg: float
    seed: int


    processing: tuple[ProcessingStep, ...] = ()


    public_id: str | None = None

    @property
    def published_id(self) -> str:
        return self.public_id or self.render_id


@dataclass(frozen=True)
class CueMeasurements:
    ild_db: float
    high_frequency_ild_db: float
    ild_flatness_db: float
    itd_us: float
    itd_peak_strength: float
    coherence: float
    estimated_azimuth_deg: float
    estimated_regime: str
    active_bins: int


    f0_hz: float = float("nan")


    binaural_azimuth_deg: float = float("nan")


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    question_type: QuestionType
    question: str
    target_instrument: str
    answer_side: Literal["left", "center", "right"]
    answer_azimuth_deg: float


@dataclass(frozen=True)
class BenchmarkItem:
    schema_version: str
    item_id: str
    audio_path: str
    audio_sha256: str
    split: Split
    question: QuestionRecord
    render: RenderPlan
    measured_evidence: CueMeasurements


@dataclass(frozen=True)
class MixtureStemPlan:
    stem_id: str
    label: str
    instrument_code: str
    azimuth_deg: float


@dataclass(frozen=True)
class MixtureRenderPlan:
    render_id: str
    piece_id: str
    split: Split
    stems: tuple[MixtureStemPlan, ...]
    start_seconds: float
    duration_seconds: float
    sample_rate: int
    seed: int


    public_id: str | None = None

    @property
    def published_id(self) -> str:
        return self.public_id or self.render_id


@dataclass(frozen=True)
class MixtureCueMeasurements:
    source_azimuths_deg: tuple[float, ...]
    source_energy_fractions: tuple[float, ...]
    active_bins: int

    @property
    def source_count(self) -> int:
        return len(self.source_azimuths_deg)


@dataclass(frozen=True)
class MixtureQuestionRecord:
    question_id: str
    question_type: MixtureQuestionType
    question: str
    answer_count: int


    answer_azimuths_deg: tuple[float, ...]
    answer_sides: tuple[Literal["left", "center", "right"], ...]


@dataclass(frozen=True)
class MixtureBenchmarkItem:
    schema_version: str
    item_id: str
    audio_path: str
    audio_sha256: str
    split: Split
    question: MixtureQuestionRecord
    render: MixtureRenderPlan
    measured_evidence: MixtureCueMeasurements


def to_json_dict(value: object) -> dict[str, object]:
    raw = asdict(value)  # type: ignore[arg-type]

    def convert(item: object) -> object:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        return item

    return cast(dict[str, object], convert(raw))
