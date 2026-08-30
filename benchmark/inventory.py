from __future__ import annotations

import re
from pathlib import Path

import soundfile as sf

from benchmark.schemas import StemRecord

INSTRUMENT_NAMES = {
    "vn": "violin",
    "va": "viola",
    "vc": "cello",
    "db": "double bass",
    "fl": "flute",
    "ob": "oboe",
    "cl": "clarinet",
    "sax": "saxophone",
    "bn": "bassoon",
    "tpt": "trumpet",
    "hn": "horn",
    "tbn": "trombone",
    "tba": "tuba",
}
STEM_PATTERN = re.compile(
    r"AuSep_(?P<track>\d+)_(?P<instrument>[a-z]+)_"
    r"(?P<piece_number>\d+)_(?P<piece_name>.+)"
)


def parse_stem(path: Path) -> StemRecord:
    match = STEM_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"unrecognized URMP filename: {path.name}")
    instrument_code = match.group("instrument")
    if instrument_code not in INSTRUMENT_NAMES:
        raise ValueError(f"unknown instrument code: {instrument_code}")

    info = sf.info(path)
    return StemRecord(
        stem_id=path.stem,
        piece_id=f"{match.group('piece_number')}_{match.group('piece_name')}",
        instrument_code=instrument_code,
        instrument_name=INSTRUMENT_NAMES[instrument_code],
        path=path.resolve(),
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
    )


def build_inventory(source_root: Path) -> list[StemRecord]:
    paths = sorted(source_root.rglob("AuSep_*.wav"))
    if not paths:
        raise FileNotFoundError(f"no AuSep WAV files found under {source_root}")

    records = [parse_stem(path) for path in paths]
    invalid = [record.path for record in records if record.channels != 1]
    if invalid:
        raise ValueError(f"AuSep inputs must be mono; invalid files: {invalid}")
    return records
