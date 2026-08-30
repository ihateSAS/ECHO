from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cues.level import SPEAKER_ANGLE_DEG, azimuth_from_ild_db, ild_db_for_azimuth  # noqa: E402
from model.finetune_data import (  # noqa: E402
    audio_only_target,
    build_example,
    faithful_derivation,
)


def _record(azimuth_deg: float, side: str, instrument: str = "violin") -> dict:
    ild = ild_db_for_azimuth(azimuth_deg) if abs(azimuth_deg) < 30 else 0.0
    return {
        "item_id": "x",
        "audio_path": "audio/train/x.wav",
        "measured_evidence": {
            "ild_db": ild,
            "high_frequency_ild_db": ild,
            "ild_flatness_db": 0.0,
            "itd_us": 0.0,
            "itd_peak_strength": 1.0,
            "coherence": 1.0,
            "estimated_azimuth_deg": azimuth_deg,
            "estimated_regime": "amplitude",
            "active_bins": 500,
            "f0_hz": float("nan"),
        },
        "question": {
            "answer_azimuth_deg": azimuth_deg,
            "answer_side": side,
            "question": f"Where is the {instrument}? Return side and azimuth.",
            "target_instrument": instrument,
        },
    }


@pytest.mark.parametrize(
    "azimuth,side", [(-25.0, "right"), (10.0, "left"), (20.0, "left"), (-5.0, "right")]
)
def test_reasoning_target_arithmetic_is_correct(azimuth, side):
    ild = ild_db_for_azimuth(azimuth)
    target = faithful_derivation(ild, azimuth, "violin")

    assert f" degrees to the {side}." in target
    assert azimuth_from_ild_db(ild) == pytest.approx(azimuth, abs=1e-6)


def test_reasoning_target_shows_every_intermediate_step():
    target = faithful_derivation(ild_db_for_azimuth(15.0), 15.0, "flute")
    assert "r = 10^" in target
    assert "d = (r - 1)/(r + 1)" in target
    assert "arctan" in target


_R_RE = r"r = 10\^\((-?[\d.]+)/20\) = ([\d.]+)"
_D_RE = r"d = \(r - 1\)/\(r \+ 1\) = (-?[\d.]+)"
_AZIMUTH_RE = r"azimuth = arctan\([^)]*\) = (-?[\d.]+) degrees"
_ILD_RE = r"level difference is (-?[\d.]+) dB"


def _parse_chain(target: str) -> tuple[float, float, float, float]:
    import re

    ild = float(re.search(_ILD_RE, target).group(1))
    formula_ild, r = re.search(_R_RE, target).groups()
    formula_ild, r = float(formula_ild), float(r)
    d = float(re.search(_D_RE, target).group(1))
    azimuth = float(re.search(_AZIMUTH_RE, target).group(1))
    assert ild == formula_ild, "opening cue and the formula's own x disagree"
    return ild, r, d, azimuth


@pytest.mark.parametrize(
    "azimuth", [-29.4, -25.0, -20.0, -15.0, -10.0, -5.0, -0.03, 0.0, 0.04, 5.0, 15.0, 25.0, 29.4]
)
def test_every_displayed_step_recomputes_from_the_one_before_it(azimuth):
    ild = ild_db_for_azimuth(azimuth) if abs(azimuth) < 29.999 else 0.0
    target = faithful_derivation(ild, azimuth, "oboe")
    displayed_ild, r, d, computed_azimuth = _parse_chain(target)

    assert r == pytest.approx(10.0 ** (displayed_ild / 20.0), abs=5e-4)
    assert d == pytest.approx((r - 1.0) / (r + 1.0), abs=5e-4)
    assert computed_azimuth == pytest.approx(
        math.degrees(math.atan(d * math.tan(math.radians(SPEAKER_ANGLE_DEG)))), abs=0.05
    )


def test_stated_side_matches_the_sign_of_the_stated_azimuth():
    for azimuth in (-20.0, 20.0):
        ild = ild_db_for_azimuth(azimuth)
        target = faithful_derivation(ild, azimuth, "tuba")
        _, _, _, computed_azimuth = _parse_chain(target)
        expected_side = "left" if computed_azimuth > 0 else "right"
        assert f"degrees to the {expected_side}" in target


def test_raises_when_ild_db_and_azimuth_deg_do_not_describe_the_same_render():
    with pytest.raises(ValueError, match="do not describe the same render"):
        faithful_derivation(ild_db_for_azimuth(25.0), -25.0, "cello")


def test_audio_only_target_is_just_the_placement():
    assert "0.0 degrees to the" not in audio_only_target(-25.0, "right", "cello")
    assert "25.0 degrees to the right" in audio_only_target(-25.0, "right", "cello")
    assert "centred" in audio_only_target(0.0, "center", "cello")


def test_build_example_cues_condition_prompt_has_cues_target_has_derivation():
    ex = build_example(_record(-25.0, "right"), "cues")
    assert ex.condition == "cues"
    assert "dB" in ex.prompt
    assert "arctan" in ex.target
    assert "25.0 degrees to the right" in ex.target


def test_build_example_audio_only_prompt_has_no_cue_numbers():
    ex = build_example(_record(-25.0, "right"), "audio_only")
    assert ex.condition == "audio_only"
    assert "dB" not in ex.prompt
    assert "left channel" in ex.prompt
    assert "arctan" not in ex.target


def test_build_example_rejects_unknown_condition():
    with pytest.raises(ValueError, match="unknown condition"):
        build_example(_record(10.0, "left"), "nonsense")
