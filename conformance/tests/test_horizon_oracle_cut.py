"""The exact horizon oracle's distance cut (max_dist): ridge cases and dense sampling of the ray."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformance import horizon_oracle as ho  # noqa: E402


def test_ridge_cut():
    z = np.zeros((9, 41)); z[:, 25] = 30.0            # ridge 15 cells east of column 10
    for R, want in ((10.0, 0.0), (14.9, 30 * 0.9 / 14.9), (15.0, 2.0), (15.1, 2.0), (None, 2.0)):
        h = ho.horizon_exact(z, 4, 10, 1.0, 0.0, 0.0, 1.0, 0.0, R)
        assert h is not None and abs(h.approx() - want) < 1e-12, (R, h.approx(), want)
    # res 2 m: the cut is in map units (R = 29.8 m = 14.9 cells)
    h = ho.horizon_exact(z, 4, 10, 1.0, 0.0, 0.0, 2.0, 0.0, 29.8)
    assert abs(h.approx() - 30 * 0.9 / 14.9) < 1e-12


def test_short_cuts_on_flat_ground():
    # the surface extends one patch past the outer cell centres (edge heights repeated: the crate's
    # convention), so even an edge cell has ground ahead; any R > 0 sees flat ground: exactly 0
    z = np.zeros((5, 5))
    for r, c, R in ((2, 4, None), (2, 2, 0.5), (2, 2, 1e-6), (2, 4, 0.3)):
        h = ho.horizon_exact(z, r, c, 1.0, 0.0, 0.0, 1.0, 0.0, R)
        assert h is not None and h.approx() == 0.0, (r, c, R)


@pytest.mark.parametrize("seed", range(6))
def test_cut_matches_dense_sampling(seed):
    rng = np.random.default_rng(seed)
    z = np.round(rng.normal(0, 2.0, (9, 9)).cumsum(1), 3)
    ang = rng.uniform(0, 2 * math.pi)
    dx, dy = math.cos(ang) * 1.3, math.sin(ang) * 1.3                       # non-unit L on purpose
    res, drop = 0.7, 0.0
    L = math.hypot(dx, dy)
    for R in (0.9, 2.3, 4.1):
        r, c = 4, 4
        h = ho.horizon_exact(z, r, c, dx, dy, drop, res, 0.0, R)
        smax = R / (res * L)
        # dense samples of f(s) = (B(X(s)) - zT) / s over (0, smax], bilinear B, inside the raster
        best = -math.inf
        zc = lambda i, j: z[min(max(i, 0), 8), min(max(j, 0), 8)]           # edge heights repeated
        for s in np.linspace(smax * 1e-4, smax, 20001):
            x, y = c + s * dx, r + s * dy
            if not (-1 <= x <= 9 and -1 <= y <= 9):                           # patches -1..8 (convention)
                break
            c0, r0 = min(int(math.floor(x)), 8), min(int(math.floor(y)), 8)
            u, v = x - c0, y - r0
            B = (zc(r0, c0) * (1 - u) * (1 - v) + zc(r0, c0 + 1) * u * (1 - v) + zc(r0 + 1, c0) * (1 - u) * v + zc(r0 + 1, c0 + 1) * u * v)
            best = max(best, (B - z[r, c]) / s)
        if best == -math.inf:
            assert h is None
            continue
        ex = h.approx()
        assert ex >= best - 1e-9, (R, ex, best)                              # sup >= every sample
        assert ex - best < 5e-3 * max(1.0, abs(ex)), (R, ex, best)           # and close to the densest one
