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
    # tilted planes (the lateral detrend: exact ties after it; tie pruning), with noise and holes
    for k, (gx, gy, sd) in enumerate([(0.37, -0.21, 0.0), (-0.13, 0.29, 0.0), (0.3, 0.1, 1e-3), (0.05, -0.4, 0.05)]):
        m = 13 + k
        y, x = np.mgrid[0:m, 0:m]
        z = 7.25 + gx * x + gy * y + np.round(rng.normal(0, sd, (m, m)), 4)
        if k == 3:
            z[rng.random(z.shape) < 0.08] = np.nan
        res = [1.0, 10.0, 0.5, 2.0][k]
        yield f"plane{k}", z, (500000.37, res, 0.0, 4000000.61, 0.0, -res)


PATHS = {"main": None, "fallback": {"VIEWSHED_HORIZON_FALLBACK_ALL": "1"},
         "exact": {"VIEWSHED_HORIZON_FALLBACK_ALL": "1", "VIEWSHED_HORIZON_EXACT_ALL": "1"},
         "strict": {"VIEWSHED_HORIZON_NODETREND": "1", "VIEWSHED_HORIZON_NOTIE": "1"}}


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


@pytest.mark.parametrize("seed", [2, 3, 6])
def test_near_planar_patches_certified(seed):
    # a tilted plane with 1e-3 m noise: f' near 0 at both ends of many stretches, where the reach
    # bound alone left enclosures ~1e-4 wide (VIEWSHED_HORIZON_CHECK tripped in the fallback before the
    # two-tangent bound, 2026-11-24); run() sets CHECK, and values must match the oracle
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:15, 0:15]
    z = 7.25 + 0.3 * x + 0.1 * y + np.round(rng.normal(0, 1e-3, (15, 15)), 4)
    gt = (500000.37, 0.5, 0.0, 4000000.61, 0.0, -0.5)
    for dx, dy in [(1.0, 1.0), (0.8, 0.6), (-1.0, 2.0)]:
        got = run(_bin(), z, gt, 0.0, False, ["vec", num(dx), num(dy)])
        _, want = ho.horizon_map(z, gt, (dx, dy), 0.0, False, earth_radius=50.0 * 0.5 * 15)
        for g, w in zip(got.ravel(), want.ravel()):
            if w == -math.inf:
                assert g == -math.inf, (seed, (dx, dy), g)
                continue
            assert abs(g - w) <= 1e-9 * 0.5 + 1e-12 * abs(w) + 1e-13 * max(1.0, abs(w)), (seed, (dx, dy), g, w)


CUT_VECS = [(1.0, 0.0), (0.8, 0.6), (-1.0, 2.0), (0.31, -0.95)]


@pytest.mark.parametrize("fallback", list(PATHS))
@pytest.mark.parametrize("name,z,gt", [x for i, x in enumerate(rasters()) if i % 3 == 0], ids=lambda v: v if isinstance(v, str) else "")
def test_cut_within_tolerance_of_exact(name, z, gt, fallback):
    # max_distance R (map units): own patch only (0.4 res), mid-patch, whole cells; curvature on half
    res = abs(gt[1])
    for i, (dx, dy) in enumerate(CUT_VECS):
        th, curv = (0.0, False) if i % 2 == 0 else (0.5, True)
        for R in (0.4 * res, 1.7 * res, 3.0 * res):
            env = {**(PATHS[fallback] or {}), "RAW_HORIZON_MAX_DISTANCE": repr(R)}
            got = run(_bin(), z, gt, th, curv, ["vec", num(dx), num(dy)], env=env)
            _, want = ho.horizon_map(z, gt, (dx, dy), th, curv, earth_radius=50.0 * res * max(z.shape), max_dist=R)
            ok = ~np.isnan(z)
            assert np.array_equal(np.isnan(got), ~ok), name
            for g, w in zip(got[ok], want[ok]):
                if w == -math.inf:
                    assert g == -math.inf, (name, (dx, dy), R, g)
                    continue
                assert abs(g - w) <= 1e-9 * res + 1e-12 * abs(w) + 1e-13 * max(1.0, abs(w)), (name, (dx, dy), R, th, g, w)


@pytest.mark.parametrize("fallback", list(PATHS))
def test_cut_inclusive_boundaries(fallback):
    gt = (0.0, 2.0, 0.0, 30.0, 0.0, -2.0)
    # (a) patch-edge distance: east ray along a row, flat then a 30 m wall 4 cells out (the patch edge at
    # s = 3 is where the ramp up to the wall starts): R = 3 cells sees only flat ground (0), 3.5 the ramp
    z = np.zeros((7, 15)); z[:, 7] = 30.0
    for R, want in ((3 * 2.0, 0.0), (3.5 * 2.0, 15.0 / 3.5), (4 * 2.0, 30.0 / 4.0)):
        got = run(_bin(), z, gt, 0.0, False, ["vec", "1.0", "0.0"], env={**(PATHS[fallback] or {}), "RAW_HORIZON_MAX_DISTANCE": repr(R)})
        assert abs(got[3, 3] - want) <= 1e-9 * 2.0 + 1e-12 * want, (R, got[3, 3], want)
    # (b) vertex at exactly R with the horizon attained there (a single peak cell): direction (3, 4)
    # (L = 5) from (1, 1) meets vertex (5, 4) at s = 1, distance 5 cells = 10 m; inclusive -> peak counts
    z = np.zeros((9, 9)); z[5, 4] = 50.0
    ex_in = ho.horizon_exact(z, 1, 1, 3.0, 4.0, 0.0, 2.0, 0.0, 10.0).approx()
    ex_out = ho.horizon_exact(z, 1, 1, 3.0, 4.0, 0.0, 2.0, 0.0, 9.999999).approx()
    assert ex_in == 50.0 and ex_out < 50.0                      # the tie at the boundary is included
    for R, want in ((10.0, ex_in), (9.999999, ex_out)):
        got = run(_bin(), z, gt[:3] + (18.0,) + gt[4:], 0.0, False, ["vec", "3.0", "4.0"], env={**(PATHS[fallback] or {}), "RAW_HORIZON_MAX_DISTANCE": repr(R)})
        assert abs(got[1, 1] - want) <= 1e-9 * 2.0 + 1e-12 * abs(want), (R, got[1, 1], want)


def test_cut_nonpositive_and_huge():
    z = np.round(np.random.default_rng(5).normal(0, 3.0, (8, 9)).cumsum(0), 2)
    z[2, 3] = np.nan
    gt = (0.0, 1.0, 0.0, 8.0, 0.0, -1.0)
    free = run(_bin(), z, gt, 0.0, False, ["vec", "0.8", "0.6"])
    huge = run(_bin(), z, gt, 0.0, False, ["vec", "0.8", "0.6"], env={"RAW_HORIZON_MAX_DISTANCE": "1e9"})
    assert np.array_equal(free, huge, equal_nan=True)
    for R in ("0.0", "-3.0", "nan"):
        got = run(_bin(), z, gt, 0.0, False, ["vec", "0.8", "0.6"], env={"RAW_HORIZON_MAX_DISTANCE": R})
        ok = ~np.isnan(z)
        assert np.array_equal(np.isnan(got), ~ok) and (got[ok] == -math.inf).all(), R
