from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from benchmark.measure import measure_render
from benchmark.process import apply_processing_chain
from benchmark.schemas import CueMeasurements, ProcessingStep
from cues.pitch import track_pitch
from spatialize.amplitude import render_amplitude_pan


ANALYSIS_SAMPLE_RATE = 48_000


DEFAULT_WINDOW_SECONDS = 5.0


QUIET_WINDOW_FLOOR_DB = -40.0


MIX_PITCH_RANGE_HZ = (55.0, 1000.0)


DEFAULT_CONTROL_ANGLES_DEG = (-25.0, -10.0, 0.0, 10.0, 25.0)


CONTROL_CHAINS: dict[str, tuple[ProcessingStep, ...]] = {
    "dry": (),
    "reverb": (ProcessingStep("reverb", {"rt60_s": 0.6, "wet": 0.3}),),
    "compression": (
        ProcessingStep("compression", {"threshold_db": -18.0, "ratio": 4.0}),
    ),
    "width": (ProcessingStep("midside", {"width": 1.5}),),
    "mastered": (
        ProcessingStep("compression", {"threshold_db": -18.0, "ratio": 4.0}),
        ProcessingStep("reverb", {"rt60_s": 0.6, "wet": 0.3}),
        ProcessingStep("midside", {"width": 1.3}),
    ),
}


AZIMUTH_TOLERANCE_DEG = 5.0


@dataclass(frozen=True)
class WindowReading:
    start_seconds: float
    rms_dbfs: float
    evidence: CueMeasurements


@dataclass(frozen=True)
class ControlReading:
    start_seconds: float
    chain: str
    planted_azimuth_deg: float
    recovered_azimuth_deg: float
    ild_db: float
    itd_us: float
    coherence: float
    regime: str

    @property
    def azimuth_error_deg(self) -> float:
        return abs(self.recovered_azimuth_deg - self.planted_azimuth_deg)


@dataclass(frozen=True)
class TrackSurvey:
    name: str
    sample_rate: int
    duration_seconds: float
    channels: int
    side_to_total_db: float
    windows: list[WindowReading]
    controls: list[ControlReading] = field(default_factory=list)
    mix_f0_hz: float = float("nan")
    mix_voiced_fraction: float = float("nan")

    def spread(self, attribute: str) -> tuple[float, float, float]:
        values = np.array(
            [getattr(window.evidence, attribute) for window in self.windows],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        return (
            float(np.mean(finite)),
            float(np.std(finite)),
            float(np.max(finite) - np.min(finite)),
        )

    @property
    def regimes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for window in self.windows:
            label = window.evidence.estimated_regime
            counts[label] = counts.get(label, 0) + 1
        return counts

    @property
    def worst_control_error_deg(self) -> float:
        if not self.controls:
            return float("nan")
        return max(control.azimuth_error_deg for control in self.controls)

    def worst_control_error_by_chain(self) -> dict[str, float]:
        worst: dict[str, float] = {}
        for control in self.controls:
            previous = worst.get(control.chain, 0.0)
            worst[control.chain] = max(previous, control.azimuth_error_deg)
        return worst


def read_stereo(path: Path, sample_rate: int = ANALYSIS_SAMPLE_RATE) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected stereo")
    if source_rate != sample_rate:
        gcd = math.gcd(int(source_rate), int(sample_rate))
        audio = resample_poly(audio, sample_rate // gcd, source_rate // gcd, axis=0)
    return np.ascontiguousarray(audio, dtype=np.float64)


def side_to_total_db(stereo: np.ndarray) -> float:
    audio = np.asarray(stereo, dtype=np.float64)
    total = float(np.mean(audio**2))
    if total <= 0.0:
        return float("-inf")
    side = float(np.mean((audio[:, 0] - audio[:, 1]) ** 2))
    return 10.0 * math.log10(side / total) if side > 0.0 else float("-inf")


def _rms_dbfs(window: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(window, dtype=np.float64) ** 2)))
    return 20.0 * math.log10(rms) if rms > 0.0 else float("-inf")


def window_starts(
    num_samples: int, sample_rate: int, window_seconds: float
) -> list[int]:
    length = round(window_seconds * sample_rate)
    if length <= 0:
        raise ValueError("window_seconds must be positive")
    return list(range(0, max(0, num_samples - length + 1), length))


def loud_windows(
    stereo: np.ndarray,
    sample_rate: int,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    floor_db: float = QUIET_WINDOW_FLOOR_DB,
) -> list[tuple[int, float]]:
    length = round(window_seconds * sample_rate)
    levels = [
        (start, _rms_dbfs(stereo[start : start + length]))
        for start in window_starts(stereo.shape[0], sample_rate, window_seconds)
    ]
    finite = [level for _, level in levels if math.isfinite(level)]
    if not finite:
        return []
    loudest = max(finite)
    return [(start, level) for start, level in levels if level >= loudest + floor_db]


def mix_pitch(
    stereo: np.ndarray, sample_rate: int
) -> tuple[float, float]:
    mono = np.asarray(stereo, dtype=np.float64).mean(axis=1)
    fmin_hz, fmax_hz = MIX_PITCH_RANGE_HZ
    try:
        track = track_pitch(mono, sample_rate, fmin_hz, fmax_hz)
    except ValueError:
        return (float("nan"), float("nan"))
    voiced = track.f0_hz[np.isfinite(track.f0_hz)]
    if track.f0_hz.size == 0:
        return (float("nan"), float("nan"))
    fraction = float(voiced.size / track.f0_hz.size)
    if voiced.size == 0:
        return (float("nan"), fraction)
    return (float(np.median(voiced)), fraction)


def control_recovery(
    stereo: np.ndarray,
    sample_rate: int,
    start: int,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    angles_deg: tuple[float, ...] = DEFAULT_CONTROL_ANGLES_DEG,
    chains: dict[str, tuple[ProcessingStep, ...]] | None = None,
) -> list[ControlReading]:
    if chains is None:
        chains = CONTROL_CHAINS
    length = round(window_seconds * sample_rate)
    window = np.asarray(stereo[start : start + length], dtype=np.float64)
    mono = window.mean(axis=1)
    readings: list[ControlReading] = []
    for angle in angles_deg:
        panned = render_amplitude_pan(mono, angle)
        for name, steps in chains.items():
            processed = apply_processing_chain(panned, sample_rate, steps)
            evidence = measure_render(processed, sample_rate)
            readings.append(
                ControlReading(
                    start_seconds=start / sample_rate,
                    chain=name,
                    planted_azimuth_deg=float(angle),
                    recovered_azimuth_deg=evidence.estimated_azimuth_deg,
                    ild_db=evidence.ild_db,
                    itd_us=evidence.itd_us,
                    coherence=evidence.coherence,
                    regime=evidence.estimated_regime,
                )
            )
    return readings


def survey_track(
    path: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    control_angles_deg: tuple[float, ...] = DEFAULT_CONTROL_ANGLES_DEG,
    control_windows: int = 1,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    control_chains: dict[str, tuple[ProcessingStep, ...]] | None = None,
) -> TrackSurvey:
    info = sf.info(path)
    stereo = read_stereo(path, sample_rate)
    selected = loud_windows(stereo, sample_rate, window_seconds)
    length = round(window_seconds * sample_rate)

    windows = [
        WindowReading(
            start_seconds=start / sample_rate,
            rms_dbfs=level,
            evidence=measure_render(stereo[start : start + length], sample_rate),
        )
        for start, level in selected
    ]

    controls: list[ControlReading] = []
    for start, _ in selected[: max(0, control_windows)]:
        controls.extend(
            control_recovery(
                stereo,
                sample_rate,
                start,
                window_seconds,
                control_angles_deg,
                control_chains,
            )
        )

    median_f0_hz, voiced_fraction = mix_pitch(stereo, sample_rate)
    return TrackSurvey(
        name=path.name,
        sample_rate=int(info.samplerate),
        duration_seconds=float(info.frames / info.samplerate),
        channels=int(info.channels),
        side_to_total_db=side_to_total_db(stereo),
        windows=windows,
        controls=controls,
        mix_f0_hz=median_f0_hz,
        mix_voiced_fraction=voiced_fraction,
    )
