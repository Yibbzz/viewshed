"""The crate's exact horizon maps (viewshed::horizon_map / horizon_rise_vec) against the exact
horizon oracle: within the certified tolerance (1e-9 res + 1e-12 |Hs|) on every cell, with
VIEWSHED_HORIZON_CHECK on; the angle output; consistency with the crate's shadow maps."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformance import horizon_oracle as ho  # noqa: E402
import cases  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_BIN = None


def _bin(name="raw_horizon"):
    global _BIN
    if _BIN is None:
        cargo = str(Path.home() / ".cargo/bin/cargo") if (Path.home() / ".cargo/bin/cargo").exists() else shutil.which("cargo")
        if not cargo:
            return None
        r = subprocess.run([cargo, "build", "--release", "-q", "--examples"], cwd=REPO, capture_output=True, text=True)
        _BIN = r.returncode == 0 and (REPO / "target/release/examples/raw_horizon").exists()
    return REPO / f"target/release/examples/{name}" if _BIN else None


pytestmark = pytest.mark.skipif(_bin() is None, reason="Rust toolchain not available")


def num(x):
    return repr(float(x))


def run(binary, z, gt, th, curv, tail, dtype="<f8", env=None):
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "z.f64"), Path(d, "o")
        np.ascontiguousarray(z, "<f8").tofile(src)
        rows, cols = z.shape
        subprocess.run([str(binary), str(src), str(rows), str(cols), num(gt[0]), num(gt[1]), num(gt[3]), str(dst),
                        num(th), "1" if curv else "0", "0.13", num(50.0 * gt[1] * max(rows, cols)), *tail],
                       check=True, capture_output=True, env={**os.environ, "VIEWSHED_HORIZON_CHECK": "1", "VIEWSHED_SHADOW_CHECK": "1", **(env or {})})
        return np.fromfile(dst, dtype).reshape(rows, cols)


VECS = [(1.0, 0.0), (0.0, -1.0), (1.0, 1.0), (-1.0, 2.0), (0.8, 0.6), (-0.7, -0.3), (0.31, -0.95), (2.0, 1.0)]


def rasters():
    seen = set()
    for name, z, gt, *_ in cases.CASES:
        key = (z.shape, float(np.nansum(z)))
        if key in seen or z.size > 300:
            continue
        seen.add(key)
        yield name, np.ascontiguousarray(z, float), (100.0, abs(gt[1]), 0.0, 500.0, 0.0, -abs(gt[1]))
    rng = np.random.default_rng(12)
    for k in range(6):
        m = int(rng.integers(7, 15))
        z = rng.normal(0, rng.uniform(0.3, 3.0), (m, m)).cumsum(0) + 0.1
        if k % 2:
            z[rng.random(z.shape) < 0.1] = np.nan
        res = [1.0, 10.0, 0.5][k % 3]
        yield f"rand{k}", z, (500000.37, res, 0.0, 4000000.61, 0.0, -res)


PATHS = {"main": None, "fallback": {"VIEWSHED_HORIZON_FALLBACK_ALL": "1"},
         "exact": {"VIEWSHED_HORIZON_FALLBACK_ALL": "1", "VIEWSHED_HORIZON_EXACT_ALL": "1"}}


@pytest.mark.parametrize("fallback", list(PATHS))
@pytest.mark.parametrize("name,z,gt", list(rasters()), ids=lambda v: v if isinstance(v, str) else "")
def test_rise_within_tolerance_of_exact(name, z, gt, fallback):
    res = abs(gt[1])
    for i, (dx, dy) in enumerate(VECS):
        th, curv = (0.0, False) if i % 2 == 0 else (0.5, True)
        got = run(_bin(), z, gt, th, curv, ["vec", num(dx), num(dy)], env=PATHS[fallback])
        ex, want = ho.horizon_map(z, gt, (dx, dy), th, curv, earth_radius=50.0 * res * max(z.shape))
        ok = ~np.isnan(z)
        assert np.array_equal(np.isnan(got), ~ok), name
        for g, w in zip(got[ok], want[ok]):
            if w == -math.inf:
                assert g == -math.inf, (name, (dx, dy), g)
                continue
            assert abs(g - w) <= 1e-9 * res + 1e-12 * abs(w) + 1e-13 * max(1.0, abs(w)), (name, (dx, dy), th, g, w)


def test_angles_and_nothing_ahead():
    z = np.array([[0.0, 1.0, 2.0, np.nan], [0.0, 1.0, 5.0, np.nan], [np.nan, np.nan, np.nan, np.nan]])
    gt = (0.0, 2.0, 0.0, 6.0, 0.0, -2.0)
    for az in (90.0, 270.0, 45.0, 200.0):
        ang = run(_bin(), z, gt, 0.0, False, ["az", num(az)])
        r = az * math.pi / 180.0
        dx, dy = {90.0: (1.0, 0.0), 270.0: (-1.0, 0.0)}.get(az, (math.sin(r), -math.cos(r)))
        rise = run(_bin(), z, gt, 0.0, False, ["vec", num(dx), num(dy)])
        want = np.where(rise == -np.inf, -90.0, np.degrees(np.arctan(rise / (math.hypot(dx, dy) * 2.0))))
        assert np.allclose(ang, want, equal_nan=True, rtol=0, atol=1e-12)
    # cell (1, 2) looking east: only no-data ahead -> -90
    assert run(_bin(), z, gt, 0.0, False, ["az", "90"])[1, 2] == -90.0


@pytest.mark.parametrize("name,z,gt", list(rasters())[::4], ids=lambda v: v if isinstance(v, str) else "")
def test_consistent_with_shadows(name, z, gt):
    res = abs(gt[1])
    for dx, dy in VECS[:6]:
        h = run(_bin(), z, gt, 0.3, True, ["vec", num(dx), num(dy)])
        for dz in (0.05, 0.4, 1.3, 3.0):
            lit = run(_bin("raw_shadow"), z, gt, 0.3, True, ["vec", num(dx), num(dy), num(dz)], dtype=np.uint8).astype(bool)
            ok = ~np.isnan(z) & (np.abs(h - dz) > 1e-9 * res + 1e-12 * np.abs(h))
            assert np.array_equal(lit[ok], ~(h[ok] > dz)), (name, (dx, dy, dz))


@pytest.mark.parametrize("fallback", list(PATHS))
def test_far_horizon(fallback):
    # a 200 m ridge 2500 cells away on flat ground with small non-dyadic noise, an inexact shallow
    # direction (arbitrary near-vertex crossings): the horizon is set ~2500 cells out, where the
    # global closed form would cancel badly (review 2026-11-24); checked against the exact oracle
    rng = np.random.default_rng(21)
    # the forced rational branch is costly per cell: a shorter raster there (ridge 650 cells out)
    cols, ridge = (701, 650) if fallback == "exact" else (2601, 2500)
    z = np.round(rng.uniform(-0.3, 0.3, (7, cols)), 3) + 0.1
    z[:, ridge] = 200.0
    gt = (0.0, 1.0, 0.0, 7.0, 0.0, -1.0)
    dx, dy = 0.9999980000019999, 0.002
    got = run(_bin(), z, gt, 0.0, False, ["vec", num(dx), num(dy)], env=PATHS[fallback])
    for c in (0, 3, 7, 11, 40):
        r = 1
        w = ho.horizon_exact(z, r, c, dx, dy, 0.0, 1.0, 0.0)
        wf = float(w.approx())
        assert wf > 0.07                                    # set by the ridge, far away
        assert abs(got[r, c] - wf) <= 1e-9 + 1e-12 * abs(wf), (c, got[r, c], wf)


def test_flat_is_exactly_zero():
    z = np.full((6, 9), 12.5)
    got = run(_bin(), z, (0.0, 1.0, 0.0, 6.0, 0.0, -1.0), 0.0, False, ["az", "63.0"])
    # every cell with ground ahead reports exactly 0 (not a tiny negative from rounding)
    assert ((got == 0.0) | (got == -90.0)).all() and (got == 0.0).any(), got
