#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.measure import measure_render  # noqa: E402
from benchmark.process import apply_processing_chain  # noqa: E402
from benchmark.schemas import ProcessingStep  # noqa: E402
from eval.real_stereo import ANALYSIS_SAMPLE_RATE, read_stereo  # noqa: E402
from spatialize.amplitude import render_amplitude_pan  # noqa: E402

WET_LEVELS = (0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
ANGLES_DEG = (25.0, 10.0)
RT60_S = 0.6
WINDOW_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("data/real_stereo/fma_02700.mp3"))
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stereo = read_stereo(args.audio, ANALYSIS_SAMPLE_RATE)
    length = round(WINDOW_SECONDS * ANALYSIS_SAMPLE_RATE)
    mono = stereo[:length].mean(axis=1)

    rows: list[dict] = []
    for angle in ANGLES_DEG:
        panned = render_amplitude_pan(mono, angle)
        for wet in WET_LEVELS:
            steps = (
                ()
                if wet == 0.0
                else (ProcessingStep("reverb", {"rt60_s": RT60_S, "wet": wet}),)
            )
            evidence = measure_render(
                apply_processing_chain(panned, ANALYSIS_SAMPLE_RATE, steps),
                ANALYSIS_SAMPLE_RATE,
            )
            row = {
                "planted_azimuth_deg": angle,
                "wet": wet,
                "rt60_s": RT60_S,
                "recovered_azimuth_deg": evidence.estimated_azimuth_deg,
                "azimuth_error_deg": abs(evidence.estimated_azimuth_deg - angle),
                "ild_db": evidence.ild_db,
                "coherence": evidence.coherence,
                "regime": evidence.estimated_regime,
            }
            rows.append(row)
            print(
                f"angle {angle:+5.1f}  wet {wet:5.3f}  "
                f"az {row['recovered_azimuth_deg']:+7.2f}  "
                f"err {row['azimuth_error_deg']:6.2f}  "
                f"coh {row['coherence']:.3f}  {row['regime']}",
                flush=True,
            )

    if args.json is not None:
        args.json.write_text(
            json.dumps({"source": args.audio.name, "rows": rows}, indent=1)
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
