import numpy as np
import pytest

THETA0_DEG = 30.0
AZIMUTH_TOL_DEG = 5.0
ILD_TOL_DB = 1.5


def _gains_for_angle(angle_deg: float) -> tuple[float, float]:
    if abs(angle_deg) > THETA0_DEG:
        raise ValueError(f"amplitude panning limited to +/-{THETA0_DEG} deg")
    d = np.tan(np.radians(angle_deg)) / np.tan(np.radians(THETA0_DEG))
    gL, gR = (1 + d) / 2, (1 - d) / 2
    k = 1.0 / np.sqrt(gL**2 + gR**2)
    return gL * k, gR * k


def _angle_from_ild(ild_db: float) -> float:
    ratio = 10 ** (ild_db / 20)
    d = (ratio - 1) / (ratio + 1)
    return np.degrees(np.arctan(d * np.tan(np.radians(THETA0_DEG))))


@pytest.mark.parametrize("angle", [-25, -15, 0, 15, 25])
def test_angle_survives_roundtrip(angle):
    gL, gR = _gains_for_angle(angle)
    ild_db = 20 * np.log10(gL / gR) if gR > 0 else np.inf
    assert abs(_angle_from_ild(ild_db) - angle) < AZIMUTH_TOL_DEG


def test_center_is_equal_gain():
    gL, gR = _gains_for_angle(0.0)
    assert gL == pytest.approx(gR)
    assert gL == pytest.approx(1 / np.sqrt(2), abs=1e-9)


def test_six_db_is_eleven_degrees():
    assert _angle_from_ild(6.0) == pytest.approx(10.9, abs=0.1)


def test_constant_power_preserved():
    for angle in (-25, -10, 0, 10, 25):
        gL, gR = _gains_for_angle(angle)
        assert gL**2 + gR**2 == pytest.approx(1.0, abs=1e-9)


def test_beyond_thirty_degrees_rejected():
    with pytest.raises(ValueError):
        _gains_for_angle(45.0)


def test_sign_convention_positive_is_left():
    gL, gR = _gains_for_angle(+20.0)
    assert gL > gR, "positive azimuth must put more energy in channel 0 (left)"
    gL, gR = _gains_for_angle(-20.0)
    assert gR > gL, "negative azimuth must put more energy in channel 1 (right)"


def test_woodworth_max_itd():
    a_over_c = 0.0875 / 343.0
    theta = np.pi / 2
    itd_us = a_over_c * (theta + np.sin(theta)) * 1e6
    assert itd_us == pytest.approx(656, abs=2)
