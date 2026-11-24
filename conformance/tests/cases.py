"""Tie-heavy viewshed cases shared by the oracle tests (numba-free, so importing it never skips).

The cases lean on ties: integer heights, observers on cell centres, corners and
edges (sightlines through lattice points), plateaus that graze, no-data, curvature
with tiny earth radii, max_distance, and non-unit cells."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import datasets  # noqa: E402


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
