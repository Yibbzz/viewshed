"""viewshed::sky (sky view factor, openness) vs the sky oracle (exact horizons per azimuth, same
aggregation), plus analytic checks. Runs the raw_sky example with VIEWSHED_HORIZON_CHECK set."""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformance import sky_oracle as so  # noqa: E402
import test_rust_horizon as th  # noqa: E402  (raster sets, binary build)

pytestmark = pytest.mark.skipif(th._bin() is None, reason="Rust toolchain not available")


def run(z, gt, kind, K, extra=(), env=None, check=True):
    b = th._bin("raw_sky")
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "z.f64"), Path(d, "o")
        np.ascontiguousarray(z, "<f8").tofile(src)
        rows, cols = z.shape
        p = subprocess.run([str(b), str(src), str(rows), str(cols), th.num(gt[0]), th.num(gt[1]), th.num(gt[3]), str(dst), kind, str(K), *extra],
                           capture_output=True, env={**os.environ, "VIEWSHED_HORIZON_CHECK": "1", **(env or {})})
        if not check:
            return p.returncode
        assert p.returncode == 0, p.stderr.decode()[-400:]
        return np.fromfile(dst, "<f8").reshape(rows, cols)


def rasters():
    out = [x for i, x in enumerate(th.rasters()) if i % 4 == 0]
    return out


@pytest.mark.parametrize("K", [5, 8])
@pytest.mark.parametrize("kind", ["vs", "iso"])
@pytest.mark.parametrize("name,z,gt", rasters(), ids=lambda v: v if isinstance(v, str) else "")
def test_svf_matches_oracle(name, z, gt, kind, K):
    res = abs(gt[1])
    for th_, curv, R in ((0.0, False, None), (0.5, True, None), (0.0, False, 2.5 * res)):
        R_env = {"RAW_SKY_MAX_DISTANCE": repr(R)} if R else None
        er = 50.0 * res * max(z.shape)
        got = run(z, gt, kind, K, (th.num(th_), "1" if curv else "0", "0.13", th.num(er)), env=R_env)
        want = so.sky_view_factor(z, gt, kind, K, th_, curv, 0.13, er, R)
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
        ok = ~np.isnan(want)
        assert np.abs(got[ok] - want[ok]).max() <= 1e-8, (name, kind, K, th_, curv, R, np.abs(got[ok] - want[ok]).max())
        # visible-sky is in [0, 1] by construction; the isotropic mean of K per-direction terms (above
        # 1 downslope, below upslope) can overshoot 1 by azimuth sampling: measured max 0.022 at K = 8,
        # 0.0026 at 16, 1e-4 at 32, 0 at 64 on these rasters (not clamped: it is the sampling error)
        top = 1 + 1e-12 if kind == "vs" else 1.05
        assert (got[ok] >= -1e-12).all() and (got[ok] <= top).all()


@pytest.mark.parametrize("K", [5, 8])
@pytest.mark.parametrize("kind", ["opos", "oneg"])
@pytest.mark.parametrize("name,z,gt", rasters(), ids=lambda v: v if isinstance(v, str) else "")
def test_openness_matches_oracle(name, z, gt, kind, K):
    res = abs(gt[1])
    for R in (None, 1.5 * res, 4.0 * res):
        got = run(z, gt, kind, K, env={"RAW_SKY_MAX_DISTANCE": repr(R)} if R else None)
        want = so.openness(z, gt, kind, K, R)
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
        ok = ~np.isnan(want)
        assert np.abs(got[ok] - want[ok]).max() <= 1e-7, (name, kind, K, R, np.abs(got[ok] - want[ok]).max())


def test_flat_ground():
    z = np.full((7, 9), 3.25)
    gt = (0.0, 2.0, 0.0, 14.0, 0.0, -2.0)
    for kind in ("vs", "iso"):
        assert (run(z, gt, kind, 8) == 1.0).all(), kind
    for kind in ("opos", "oneg"):
        o = run(z, gt, kind, 8)
        assert np.allclose(o, 90.0, atol=1e-12), (kind, o)


def test_tilted_plane():
    # an unobstructed tilted plane: its own surface is the horizon, nothing rises above it, so the
    # isotropic (sloped) sky view factor is 1; (1 + cos S) / 2 would be a tilted element in HORIZONTAL
    # surroundings (topocalc, clamping horizons at the horizontal, gives ~cos S here)
    y, x = np.mgrid[0:21, 0:21]
    gx, gy = 0.3, -0.2
    res = 1.0
    z = 5.0 + gx * x * res + gy * y * res
    gt = (0.0, res, 0.0, 21.0, 0.0, -res)
    # radius 8 cells: stay off the raster's edge, where the surface extends flat at the edge height
    # (half a cell beyond the outer centres) and genuinely raises downslope horizons
    within = {"RAW_SKY_MAX_DISTANCE": "8.0"}
    iso = run(z, gt, "iso", 64, ("0.0", "0", "0.13", "6371008.8"), env=within)
    assert abs(iso[10, 10] - 1.0) < 1e-9, iso[10, 10]
    # visible-sky: 1 - mean sin(max(H, 0)), H the plane's own slope towards each azimuth
    vs = run(z, gt, "vs", 720, ("0.0", "0", "0.13", "6371008.8"), env=within)
    g = math.hypot(gx, gy)
    phis = np.linspace(0, 2 * math.pi, 100000, endpoint=False)
    want_vs = 1 - np.mean(np.maximum(np.sin(np.arctan(g * np.cos(phis))), 0.0))
    assert abs(vs[10, 10] - want_vs) < 1e-4, (vs[10, 10], want_vs)
    # openness on a plane: 90 - mean atan(g cos phi) = 90 (symmetric), both signs
    for kind in ("opos", "oneg"):
        o = run(z, gt, kind, 64, env=within)
        assert abs(o[10, 10] - 90.0) < 1e-9, (kind, o[10, 10])


def test_negative_openness_is_positive_of_flipped():
    rng = np.random.default_rng(4)
    z = np.round(rng.normal(0, 2.0, (9, 11)).cumsum(1), 2)
    gt = (0.0, 1.5, 0.0, 13.5, 0.0, -1.5)
    a = run(z, gt, "oneg", 8, env={"RAW_SKY_MAX_DISTANCE": "6.0"})
    b = run(-z, gt, "opos", 8, env={"RAW_SKY_MAX_DISTANCE": "6.0"})
    assert np.array_equal(a, b)


def test_zero_azimuths_is_an_error():
    z = np.zeros((4, 4))
    assert run(z, (0.0, 1.0, 0.0, 4.0, 0.0, -1.0), "vs", 0, check=False) != 0
