from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.measure import measure_render  # noqa: E402
from cues.level import estimate_level_cues  # noqa: E402
from spatialize.mixture import StemPlacement, mix_stems  # noqa: E402

SAMPLE_RATE = 48_000
MEDLEYDB_ROOT = Path(__file__).parent.parent / "data" / "MedleyDB" / "MedleyDB_sample"
AUDIO_ROOT = MEDLEYDB_ROOT / "Audio"
ACTIVATION_ROOT = MEDLEYDB_ROOT / "Annotations" / "Instrument_Activations" / "ACTIVATION_CONF"
ACTIVATION_THRESHOLD = 0.3


FOLK_TRIO = [
    ("Phoenix_ScotchMorris", "Phoenix_ScotchMorris_RAW_01_01.wav", "S01", "acoustic_guitar"),
    ("Phoenix_ScotchMorris", "Phoenix_ScotchMorris_RAW_02_01.wav", "S02", "flute"),
    ("Phoenix_ScotchMorris", "Phoenix_ScotchMorris_RAW_03_01.wav", "S03", "violin"),
]
POP_TRIO = [
    ("LizNelson_Rainfall", "LizNelson_Rainfall_RAW_01_01.wav", "S01", "female_singer"),
    ("LizNelson_Rainfall", "LizNelson_Rainfall_RAW_04_01.wav", "S04", "acoustic_guitar"),
    ("LizNelson_Rainfall", "LizNelson_Rainfall_RAW_05_01.wav", "S05", "clean_electric_guitar"),
]


def _medleydb_available() -> bool:
    return AUDIO_ROOT.exists()


def _first_all_active_window(piece: str, columns: list[str], duration_s: float) -> float:
    path = ACTIVATION_ROOT / f"{piece}_ACTIVATION_CONF.lab"
    rows = list(csv.DictReader(path.open()))
    times = [float(row["time"]) for row in rows]
    active = [all(float(row[c]) > ACTIVATION_THRESHOLD for c in columns) for row in rows]

    for i, (t, ok) in enumerate(zip(times, active)):
        if not ok:
            continue
        end_t = t + duration_s
        j = i
        while j < len(times) and times[j] < end_t:
            if not active[j]:
                break
            j += 1
        else:
            return t
    raise ValueError(f"{piece}: no {duration_s}s window has {columns} all active together")


def _read_raw_resampled(
    piece: str, filename: str, start_s: float, duration_s: float = 8.0
) -> np.ndarray:
    path = AUDIO_ROOT / piece / f"{piece}_RAW" / filename
    audio, source_rate = sf.read(path, dtype="float64", always_2d=True)
    assert audio.shape[1] == 1, f"{path} is not mono"
    mono = audio[:, 0]
    start_frame = int(start_s * source_rate)
    frames = int(duration_s * source_rate)
    mono = mono[start_frame : start_frame + frames]
    if source_rate != SAMPLE_RATE:
        gcd = int(np.gcd(source_rate, SAMPLE_RATE))
        mono = resample_poly(mono, SAMPLE_RATE // gcd, source_rate // gcd)
    return mono


@pytest.mark.skipif(not _medleydb_available(), reason="MedleyDB sample not present at data/MedleyDB/")
@pytest.mark.parametrize(
    "trio,angles,name",
    [
        (FOLK_TRIO, (-25.0, 0.0, 25.0), "folk_trio"),
        (POP_TRIO, (-20.0, 5.0, 20.0), "pop_trio"),
    ],
)
def test_explore_real_medleydb_trio(trio, angles, name):
    piece = trio[0][0]
    columns = [column for _, _, column, _ in trio]
    duration_s = 8.0
    start_s = _first_all_active_window(piece, columns, duration_s)

    placements = [
        StemPlacement(_read_raw_resampled(piece, filename, start_s, duration_s), angle, label)
        for (_, filename, _, label), angle in zip(trio, angles)
    ]
    mix, ground_truth = mix_stems(placements)
    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        mix = mix / peak

    level = estimate_level_cues(mix, SAMPLE_RATE)
    evidence = measure_render(mix, SAMPLE_RATE)

    print(f"\n[medleydb explore: {name}] window: {piece} @ {start_s:.2f}s-{start_s+duration_s:.2f}s")
    print(f"[medleydb explore: {name}] planted: {ground_truth}")
    print(
        f"[medleydb explore: {name}] measured: azimuth={level.azimuth_deg:.2f} deg, "
        f"ild={level.broadband_ild_db:.2f} dB, active_bins={level.active_bins}, "
        f"regime={evidence.estimated_regime}, coherence={evidence.coherence:.3f}"
    )
    assert np.isfinite(level.azimuth_deg)
    assert -30.0 <= level.azimuth_deg <= 30.0
