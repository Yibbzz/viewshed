"""The cast-shadow oracle (conformance/shadow_oracle.py): the fast float pass + rational fallback must
equal the pure rational procedure on every cell, and the semantics are pinned by small scenes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformance import shadow_oracle as so  # noqa: E402
import test_oracle_fast as tof  # noqa: E402

# lattice-aligned and general directions (dx, dy, dz): axis and diagonal rays through grid points,
# slopes equal to integer height steps (exact ties on integer rasters), a horizontal sun
VECS = [(1.0, 0.0, 0.5), (0.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 0.25), (0.8, 0.6, 0.3),
        (0.6, -0.8, 2.0), (1.0, 0.0, 0.0), (-0.7, -0.3, 0.05), (2.0, 1.0, 1.0)]


def _small_cases():
    seen = set()
    for name, z, gt, obs, oh, th, md, cv in tof.CASES:
        key = (z.shape, float(np.nansum(z)), float(gt[1]))
        if key in seen or z.size > 400:
            continue
        seen.add(key)
        yield name, np.ascontiguousarray(z, float), gt, bool(cv.get("curvature", False))


@pytest.mark.skipif(not so.HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("case", list(_small_cases()), ids=lambda c: c[0])
def test_fast_equals_exact(case):
    name, z, gt, curv = case
    for vec in VECS:
        for th in (0.0, 0.5):
            fast, _ = so.shadow_vec(z, gt, vec, th, curv, fast=True)
            slow, _ = so.shadow_vec(z, gt, vec, th, curv, fast=False)
            assert (fast == slow).all(), f"{name} {vec} th {th}: {np.argwhere(fast != slow)[:4].tolist()}"


def test_point_touch_does_not_block():
    # the sun ray from (6, 0) to the north-east passes exactly through the 100 m vertex (3, 3);
    # the patches it crosses on either side have no data (NaN at (4, 2) and (2, 4)), and the patches
    # containing the vertex only at a point don't count (positive-length stretches only)
    z = np.zeros((7, 7)); z[3, 3] = 100.0; z[4, 2] = np.nan; z[2, 4] = np.nan
    gt = (0.0, 1.0, 0.0, 7.0, 0.0, -1.0)
    lit, _ = so.shadow_vec(z, gt, (1.0, -1.0, 0.1), fast=False)
    assert lit[6, 0]


def test_blocker_shadows():
    z = np.zeros((5, 9)); z[:, 6] = 10.0       # a wall east of the cells
    gt = (0.0, 1.0, 0.0, 5.0, 0.0, -1.0)
    lit, _ = so.shadow(z, gt, 90.0, 20.0)      # sun due east, 20 degrees up
    assert not lit[2, 3] and not lit[2, 5]     # behind the wall: shadowed
    assert lit[2, 8]                           # east of the wall: lit
    assert lit[2, 6]                           # the wall top itself: nothing higher beyond


def test_flat_lit():
    z = np.full((6, 6), 3.0)
    gt = (500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0)
    lit, n = so.shadow(z, gt, 135.0, 5.0)
    assert lit.all()
