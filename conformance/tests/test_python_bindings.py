"""The Python package must give the r.viewshed-model oracle's answer on every
cell (flat earth and with curvature), and the same ZTV as the Rust crate."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import oracle_rviewshed as orv  # noqa: E402
from conformance.adapters import PythonBindings  # noqa: E402

PY = PythonBindings()
pytestmark = pytest.mark.skipif(not PY.available(),
                                reason="viewshed bindings not built (maturin develop in python/)")

from test_curvature import curved_jobs  # noqa: E402
from test_rust_sweep import fixed_jobs, random_jobs  # noqa: E402


@pytest.mark.parametrize("job", [*fixed_jobs(), *random_jobs(20, seed=3), *curved_jobs(12, seed=4)])
def test_bindings_equal_oracle(job):
    theirs = orv.viewshed(job.dem, job.gt, job.observer, job.observer_height, job.target_height,
                          job.max_distance, **job.curve_kwargs())
    assert (PY.run(job) == theirs).all()


def test_ztv_forward_equals_viewshed_from_every_cell():
    import viewshed
    rng = np.random.default_rng(8)
    n = 15
    z = rng.normal(0, 2, (n, n)).cumsum(1)
    gt = (0.0, 1.0, 0.0, float(n), 0.0, -1.0)
    targets = [(3.5, 11.5, 6.0), (10.5, 2.5, 0.0)]
    kw = dict(curvature=True, refraction=0.13, earth_radius=40.0)
    counts = viewshed.ztv(z, gt, targets, observer_height=1.5, **kw)
    for r in range(n):
        for c in range(n):
            expect = 0
            for tx, ty, h in targets:
                seen = viewshed.viewshed(z, gt, (c + 0.5, n - r - 0.5), observer_height=1.5,
                                         target_height=h, **kw)
                expect += seen[int(n - ty), int(tx)]
            assert counts[r, c] == expect


def _reverse_scenes():
    rng = np.random.default_rng(21)
    for k in range(10):
        n = 9
        z = rng.integers(0, 4, (n, n)).astype(float)                 # ties everywhere
        if k % 2:
            z[rng.random((n, n)) < 0.08] = np.nan
        tr, tc = int(rng.integers(1, n - 1)), int(rng.integers(1, n - 1))
        z[tr, tc] = 1.0
        z[tr + 1, tc] = z[tr, tc + 1] = z[tr + 1, tc + 1] = 1.0      # target patch has a surface
        # target on its cell centre, or off it (the bilinear target keeps its exact point)
        off = (0.0, 0.0) if k < 5 else (0.25, -0.375)
        yield k, z, (tr, tc), off, float(rng.integers(0, 3)), float(rng.integers(0, 4)), k % 3 == 2



def _blocked_between(z, gt, a, b, drop):
    """Exact: does the segment from point a = (x, y, height) to b pass strictly below
    the bilinear surface (curvature drop as the oracle's, from a) on a stretch of
    positive length? The oracle's own procedure, with b any point (not a cell centre)."""
    from fractions import Fraction as F
    from conformance import oracle
    uv = lambda x, y: ((F(x) - F(gt[0])) / F(gt[1]) - F(1, 2), (F(y) - F(gt[3])) / F(gt[5]) - F(1, 2))
    (ue, ve), (ub, vb) = uv(a[0], a[1]), uv(b[0], b[1])
    zf = np.vectorize(lambda v: v if np.isnan(v) else F(v), otypes=[object])(z)
    du, dv = ub - ue, vb - ve
    bend = F(drop) * (du * du + dv * dv) * F(abs(gt[1])) ** 2
    dz = b[2] - bend - a[2]
    ts = sorted({F(0), F(1), *oracle._crossings(ue, du), *oracle._crossings(ve, dv)})
    for ta, tb in zip(ts, ts[1:]):
        m = oracle._segment_min(zf, ue, ve, a[2], du, dv, dz, ta, tb, bend=bend)
        if m is not None and m < 0:
            return True
    return False


@pytest.mark.parametrize("scene", list(_reverse_scenes()), ids=lambda s: f"scene{s[0]}")
def test_reverse_viewshed_bilinear_equals_exact_oracle_from_every_cell(scene):
    """Bilinear reverse viewshed == the oracle's exact procedure run forwards: from the eye
    at every cell centre (its height + observer_height) to the target point (+ target_height).
    Tie-heavy integer rasters, holes, the target on and off a cell centre, with curvature."""
    from fractions import Fraction as F
    import viewshed
    from conformance import oracle
    _, z, (tr, tc), (ox, oy), oh, th, curved = scene
    n = z.shape[0]
    gt = (0.0, 1.0, 0.0, float(n), 0.0, -1.0)
    tx, ty = tc + 0.5 + ox, n - tr - 0.5 + oy
    kw = dict(curvature=curved, refraction=0.13, earth_radius=60.0)
    rev = viewshed.reverse_viewshed(z, gt, (tx, ty), target_height=th, observer_height=oh,
                                    surface="bilinear", **kw)
    drop = oracle.drop_factor(**kw)
    tz = oracle._ground_exact(z, gt, tx, ty) + F(th)
    seen = 0
    for r in range(n):
        for c in range(n):
            if np.isnan(z[r, c]):
                assert not rev[r, c]
                continue
            eye = (c + 0.5, n - r - 0.5, F(z[r, c]) + F(oh))   # surface at a centre = z[r, c]
            want = not _blocked_between(z, gt, eye, (tx, ty, tz), drop)
            assert rev[r, c] == want, (r, c)
            seen += want
    assert 0 < seen
