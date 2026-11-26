"""fresnel_clearance (Rust, via the Python package) against the exact Fresnel oracle.

The oracle (conformance/fresnel_oracle.py) computes the worst clearance ratio in rational
arithmetic (Sturm-isolated roots of the per-piece cubic, 60-digit evaluation); the crate
must be within 1e-9 + 1e-12 |ratio| of it, and its reported position must be a point whose
ratio is that close too.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformance import fresnel_oracle as fo  # noqa: E402
from conformance import oracle  # noqa: E402

viewshed = pytest.importorskip("viewshed")


def tol(v):
    return 1e-9 + 1e-12 * abs(v)


def _scenes(n_scenes=60, seed=43):
    """Tie-heavy integer rasters, random walks, no-data, strong curvature (earth radius 30
    cells), ends on cell centres, cell corners, grid lines and anywhere."""
    rng = np.random.default_rng(seed)
    for k in range(n_scenes):
        n = int(rng.integers(5, 16))
        z = rng.integers(0, 4, (n, n)).astype(float) if k % 2 else \
            np.round(rng.normal(0, 2, (n, n)).cumsum(0), 1)
        if k % 3 == 1:
            z[rng.random((n, n)) < 0.12] = np.nan
        res = [1.0, 0.5, 2.0][k % 3]
        gt = (100.0 * (k % 4), res, 0.0, 300.0 + n * res, 0.0, -res)

        def point():
            kind = int(rng.integers(0, 4))
            if kind == 0:
                c, r = rng.integers(0, n, 2) + 0.5
            elif kind == 1:
                c, r = rng.integers(1, n, 2).astype(float)
            elif kind == 2:
                c, r = rng.integers(0, n) + 0.5, rng.uniform(0.5, n - 0.5)
            else:
                c, r = rng.uniform(0, n, 2)
            return (gt[0] + c * res, gt[3] - r * res, float(rng.choice([0.5, 1.0, 2.0, 3.0])))

        pairs = [(point(), point()) for _ in range(30)]
        freq = float([1e8, 9e8, 6e9][k % 3])
        yield k, z, gt, pairs, bool(k % 2), 30.0 * res, freq


@pytest.fixture(params=["float", "exact"])
def path(request, monkeypatch):
    """Both paths: the float search, and every link through the exact fallback
    (VIEWSHED_FRESNEL_EXACT_ALL; read per call, so it applies to the next call)."""
    if request.param == "exact":
        monkeypatch.setenv("VIEWSHED_FRESNEL_EXACT_ALL", "1")
    return request.param


@pytest.mark.parametrize("scene", list(_scenes()), ids=lambda s: f"scene{s[0]}")
def test_fresnel_equals_exact_oracle(scene, path):
    _, z, gt, pairs, curved, radius, freq = scene
    kw = dict(curvature=curved, refraction=0.13, earth_radius=radius)
    checked = 0
    for a, b in pairs:
        try:
            _, _, want = fo.fresnel_clearance(z, gt, a, b, freq, **kw)
        except ValueError:
            with pytest.raises(ValueError):
                viewshed.fresnel_clearance(z, gt, a, b, freq, **kw)
            continue
        for p, q in ((a, b), (b, a)):
            got, t = viewshed.fresnel_clearance(z, gt, p, q, freq, return_position=True, **kw)
            if math.isinf(want):
                assert got == want and math.isnan(t), (p, q, got)
                continue
            assert abs(got - want) <= tol(want), (p, q, got, want)
            assert 0 < t < 1, t
        checked += 1
    assert checked >= 3


def test_fresnel_random_walk_links():
    """Longer links on rough random walks with holes, all three frequencies, curvature."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for k in range(40):
        n = 40
        z = rng.normal(0, 2, (n, n)).cumsum(0) + rng.normal(0, 0.5, (n, n))
        if k % 4 == 0:
            z[rng.random((n, n)) < 0.05] = np.nan
        res = 10.0
        gt = (0.0, res, 0.0, n * res, 0.0, -res)
        a = (rng.uniform(5, n * res - 5), rng.uniform(5, n * res - 5), float(rng.uniform(1, 20)))
        b = (rng.uniform(5, n * res - 5), rng.uniform(5, n * res - 5), float(rng.uniform(1, 20)))
        if oracle._ground_exact(z, gt, a[0], a[1]) is None or oracle._ground_exact(z, gt, b[0], b[1]) is None:
            continue
        freq = float([1e8, 2.4e9, 6e9][k % 3])
        _, _, want = fo.fresnel_clearance(z, gt, a, b, freq, curvature=True)
        got = viewshed.fresnel_clearance(z, gt, a, b, freq)
        assert abs(got - want) <= tol(want), (k, got, want)
        worst = max(worst, abs(got - want) / tol(want))
    assert worst <= 1.0


def test_fresnel_flat_closed_form_and_batch():
    z = np.zeros((20, 120))
    gt = (0.0, 10.0, 0.0, 200.0, 0.0, -10.0)
    got, t = viewshed.fresnel_clearance(z, gt, (15.0, 105.0, 10.0), (1015.0, 105.0, 10.0), 1e9,
                                        curvature=False, return_position=True)
    want = 2 * 10 / math.sqrt(299792458.0 / 1e9 * 1000)
    assert abs(got - want) <= tol(want) and abs(t - 0.5) < 1e-6
    a = np.array([[15.0, 105.0, 10.0], [15.0, 55.0, 5.0]])
    batch = viewshed.fresnel_clearance(z, gt, a, (1015.0, 105.0, 10.0), 1e9, curvature=False)
    assert batch.shape == (2,) and abs(batch[0] - want) <= tol(want)


def test_fresnel_errors():
    z = np.zeros((10, 10))
    gt = (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="frequency"):
        viewshed.fresnel_clearance(z, gt, (1.5, 1.5, 1.0), (5.5, 5.5, 1.0), 0.0)
    with pytest.raises(ValueError, match="> 0"):
        viewshed.fresnel_clearance(z, gt, (1.5, 1.5, 0.0), (5.5, 5.5, 1.0), 1e9)
    with pytest.raises(ValueError, match="coincide"):
        viewshed.fresnel_clearance(z, gt, (1.5, 1.5, 1.0), (1.5, 1.5, 2.0), 1e9)
    with pytest.raises(ValueError, match="outside"):
        viewshed.fresnel_clearance(z, gt, (1.5, 1.5, 1.0), (50.5, 5.5, 1.0), 1e9)
