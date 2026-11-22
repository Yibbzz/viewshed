"""oracle_fast and oracle_numba must be bit-identical to oracle.py: same visibility, same clearance
(every bit, NaN where oracle.py has NaN), same count of exact-arithmetic decisions.

The cases lean on ties: integer heights, observers on cell centres, corners and
edges (sightlines through lattice points), plateaus that graze, no-data, curvature
with tiny earth radii, max_distance, and non-unit cells."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import datasets, oracle, oracle_fast  # noqa: E402

FAST = {"numpy": oracle_fast}
try:
    from conformance import oracle_numba  # noqa: E402
    FAST["numba"] = oracle_numba
except ImportError:                          # numba is optional
    pass


def _cases():
    rng = np.random.default_rng(42)
    for i in range(40):
        n = int(rng.integers(5, 24))
        m = int(rng.integers(5, 24))
        kind = i % 5
        if kind == 0:
            z = rng.normal(0, 2, (n, m)).cumsum(0)
        elif kind == 1:
            z = rng.integers(0, 4, (n, m)).astype(float)               # ties everywhere
        elif kind == 2:
            z = np.zeros((n, m)); z[: n // 2] = 2.0                     # grazing plateau
        elif kind == 3:
            z = rng.integers(0, 3, (n, m)).astype(float) * 0.5
            z[rng.random(z.shape) < 0.15] = np.nan
        else:
            z = rng.normal(0, 1, (n, m)).cumsum(1).cumsum(0) / 3
        res = [1.0, 2.0, 0.5, 30.0][i % 4]
        gt = (100.0, res, 0.0, 500.0 + n * res, 0.0, -res)
        # observer on a cell centre, a corner, an edge or anywhere
        c = int(rng.integers(0, m)); r = int(rng.integers(0, n))
        fx, fy = [(0.5, 0.5), (0.0, 0.0), (0.5, 0.0), tuple(rng.random(2))][(i // 5) % 4]
        x = gt[0] + (c + fx) * res
        y = gt[3] - (r + fy) * res
        oh = [0.0, 1.6, 2.0, 5.0][i % 4]
        th = [0.0, 0.0, 1.5, 2.0][(i // 3) % 4]
        md = None if i % 3 else float(rng.integers(3, 12)) * res
        curve = dict(curvature=bool(i % 2), refraction=0.13,
                     earth_radius=[6371008.8, 50.0 * res, 500.0 * res][i % 3])
        yield f"case{i}", z, gt, (x, y), oh, th, md, curve
    yield "rough", datasets.rough(), datasets.GT, datasets.OBSERVER, 1.6, 0.0, None, {}
    yield "holes", datasets.nodata_holes(), datasets.GT, datasets.OBSERVER, 1.6, 1.5, 20.0, \
        dict(curvature=True, refraction=0.13, earth_radius=2000.0)


CASES = list(_cases())


@pytest.mark.parametrize("impl", sorted(FAST))
@pytest.mark.parametrize("name,z,gt,obs,oh,th,md,curve", CASES, ids=[c[0] for c in CASES])
def test_bit_identical_to_reference(impl, name, z, gt, obs, oh, th, md, curve):
    fast = FAST[impl]
    try:
        want = oracle.viewshed(z, gt, obs, oh, th, md, **curve)
    except ValueError as e:                       # observer on no-data: both must refuse
        with pytest.raises(ValueError, match=str(e)):
            fast.viewshed(z, gt, obs, oh, th, md, **curve)
        return
    got = fast.viewshed(z, gt, obs, oh, th, md, **curve)
    assert (got[0] == want[0]).all(), f"visibility differs at {np.argwhere(got[0] != want[0])[:5]}"
    a, b = got[1].view(np.uint64), want[1].view(np.uint64)
    assert (a == b).all(), f"clearance bits differ at {np.argwhere(a != b)[:5]}"
    assert got[2] == want[2]
