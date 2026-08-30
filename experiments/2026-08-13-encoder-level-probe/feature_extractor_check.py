#!/usr/bin/env python3


from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model.qwen_backbone import BACKBONE_MODEL_ID, BACKBONE_SAMPLE_RATE  # noqa: E402


REAL_ILDS_DB = (2.7, 3.2, 5.5, 6.7, 8.7, 10.9, 12.9, 17.2, 19.5, 36.1)
OUT = Path(__file__).resolve().parent / "feature_extractor.json"


def main() -> None:
    from transformers import AutoProcessor

    extractor = AutoProcessor.from_pretrained(BACKBONE_MODEL_ID).feature_extractor
    rate = BACKBONE_SAMPLE_RATE
    rng = np.random.default_rng(2026)
    time = np.arange(int(4.0 * rate)) / rate


    signal = (
        0.5 * np.sin(2 * np.pi * 220 * time)
        + 0.15 * np.sin(2 * np.pi * 660 * time)
        + 0.05 * rng.standard_normal(time.size)
    ).astype(np.float32)

    rows = []
    for ild_db in REAL_ILDS_DB:
        ratio = 10.0 ** (ild_db / 20.0)
        left, right = 0.9, 0.9 / ratio
        features = extractor(
            [signal * left, signal * right], sampling_rate=rate, return_tensors="np"
        )["input_features"]
        difference = features[0] - features[1]


        above_floor = features[1] > features[1].max() - 7.5
        rows.append(
            {
                "ild_db": ild_db,
                "predicted_offset": math.log10(ratio**2) / 4.0,
                "measured_offset": float(difference[above_floor].mean()),
            }
        )

    worst = max(abs(row["predicted_offset"] - row["measured_offset"]) for row in rows)
    result = {
        "extractor": type(extractor).__name__,
        "sampling_rate": int(extractor.sampling_rate),
        "feature_size": int(extractor.feature_size),
        "prediction": "uniform additive offset of log10(power ratio) / 4",
        "worst_absolute_deviation": worst,
        "survives": worst < 0.02,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
