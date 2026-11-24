"""Solar position (conformance/sunpos.py) against NREL SPA reference values (stored, generated with
pvlib; see spa_reference.json): within 0.02 deg (direction and elevation; apparent elevation when
the sun is up). Azimuth is compared as a direction (it is ill-defined near the zenith)."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance.sunpos import sun_position  # noqa: E402

REF = json.loads((Path(__file__).parent / "spa_reference.json").read_text())["points"]
TOL = 0.02


def _vec(az, el):
    a, e = math.radians(az), math.radians(el)
    return (math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e))


@pytest.mark.parametrize("p", REF, ids=lambda p: f"{p['t']:.0f}_{p['lat']:.1f}_{p['lon']:.1f}")
def test_matches_spa(p):
    az, eapp, etrue = sun_position(p["lat"], p["lon"], p["t"])
    assert abs(etrue - p["elevation"]) < TOL
    u, v = _vec(az, etrue), _vec(p["azimuth"], p["elevation"])
    ang = math.degrees(math.acos(min(1.0, sum(a * b for a, b in zip(u, v)))))
    assert ang < TOL
    if p["apparent_elevation"] > 0:
        assert abs(eapp - p["apparent_elevation"]) < TOL
    assert 0.0 <= az < 360.0


def test_rejects_bad_input():
    for bad in ((91.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, float("inf"))):
        with pytest.raises(ValueError):
            sun_position(*bad)
