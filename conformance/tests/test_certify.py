"""certify.segment against exact arithmetic: every segment it calls blocking must
have an exact minimum < 0, every one it calls not blocking an exact minimum >= 0
where it matters (> 0, or on the target's own segment with target height 0, no dip
below 0 before the target). Segments are oracle.py's own float segments, on rasters
built to stress the bounds: large projected coordinates, non-dyadic cell sizes,
large heights with small relief, integer ties, tiny earth radii."""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import certify, oracle, oracle_fast  # noqa: E402


def sightline_segments(z, transform, x, y, oh, th, drop, r, c):
    """oracle.py's float segments for cell (r, c), with the sliver flags."""
    ue = (x - transform[0]) / transform[1] - 0.5
    ve = (y - transform[3]) / transform[5] - 0.5
    ground = oracle._ground_exact(z, transform, x, y)
    ze = float(ground) + oh
    res = abs(transform[1])
    raster = certify.Raster(transform, x, y, ground, oh, ue, ve, ze)
    cell = certify.Cell(raster, th, drop, res, z[r, c], r, c, certify.raster_slope(z))
    ts, close = certify.boundaries(cell, r, c)
    out = []
    for i, (ta, tb) in enumerate(zip(ts, ts[1:])):
        flagged = ta in close or tb in close
        out.append((ta, tb, i == len(ts) - 2, flagged))
    return cell, ground, out


def exact_min(z, transform, x, y, ground, oh, th, drop, r, c, r0, c0):
    """Exact minimum of the gap over oracle.py's exact segment in patch (r0, c0),
    closed interval; None when the sightline has no segment there."""
    ue, ve, ze, du, dv, dz, bend = oracle_fast._exact_setup(z, transform, x, y, ground, oh,
                                                            r, c, th, drop)
    seg = oracle_fast._exact_patch_segment(ue, ve, du, dv, r0, c0)
    if seg is None:
        return None
    zf = oracle_fast._LazyFractions(z)
    return oracle._segment_min(zf, ue, ve, ze, du, dv, dz, seg[0], seg[1], bend=bend)


def rasters(seed, count):
    rng = np.random.default_rng(seed)
    for i in range(count):
        n = int(rng.integers(6, 16))
        kind = i % 6
        if kind == 0:
            z = rng.integers(0, 4, (n, n)).astype(float)
        elif kind == 1:
            z = 1234.5 + rng.normal(0, 0.01, (n, n)).cumsum(0)        # big heights, tiny relief
        elif kind == 2:
            z = np.zeros((n, n)); z[: n // 2] = 2.0
        elif kind == 3:
            z = rng.normal(0, 3, (n, n)).cumsum(1)
        elif kind == 4:
            z = rng.integers(0, 3, (n, n)) * 0.1                          # non-dyadic ties
        else:
            z = rng.normal(0, 1, (n, n)) * 50
        res = [1.0, 0.5, 2.0, 30.0, 0.1, 0.3][i % 6]
        x0 = [0.0, 412345.0, 650000.5, 100.0][(i // 6) % 4]
        y0 = [float(n) * res, 1234567.0, 5e5 + 0.25, 999.0][(i // 6) % 4]
        gt = (x0, res, 0.0, y0, 0.0, -res)
        cc, rr = rng.integers(0, n, 2)
        fx, fy = [(0.5, 0.5), (0.0, 0.0), (0.5, 0.0), tuple(rng.random(2))][i % 4]
        x, y = x0 + (cc + fx) * res, y0 - (rr + fy) * res
        oh = [1.6, 0.0, 2.0, 10.0][i % 4]
        th = [0.0, 1.5, 0.0, 2.0][(i // 4) % 4]
        drop = oracle.drop_factor(bool(i % 2), 0.13, [6371008.8, 40.0 * res][(i // 2) % 2])
        yield z, gt, x, y, oh, th, drop


def check(seed, count, cells_per_raster=40):
    rng = np.random.default_rng(seed + 1000)
    stats = {0: 0, 1: 0, 2: 0, 3: 0, "flag": 0}
    for z, gt, x, y, oh, th, drop in rasters(seed, count):
        if oracle._ground_exact(z, gt, x, y) is None:
            continue
        n = z.shape[0]
        for _ in range(cells_per_raster):
            r, c = (int(v) for v in rng.integers(0, n, 2))
            cell, ground, segs = sightline_segments(z, gt, x, y, oh, th, drop, r, c)
            for ta, tb, last, flagged in segs:
                if flagged:
                    stats["flag"] += 1
                    continue
                state, r0, c0 = certify.segment(cell, z, ta, tb, last)
                stats[state] += 1
                if state in (2, 3):
                    continue
                m = exact_min(z, gt, x, y, ground, oh, th, drop, r, c, r0, c0)
                where = f"cell {(r, c)} patch {(r0, c0)} t {ta}..{tb} last={last}"
                assert m is not None, f"certified a segment the exact sightline lacks: {where}"
                if state == 1:
                    assert m < 0, f"certified blocking, exact minimum {float(m)}: {where}"
                elif last and th == 0:
                    assert m >= 0, f"certified clear, exact minimum {float(m)}: {where}"
                else:
                    assert m > 0, f"certified clear, exact minimum {float(m)}: {where}"
    return stats


@pytest.mark.parametrize("seed", range(4))
def test_certified_segments_agree_with_exact_arithmetic(seed):
    stats = check(seed, 24)
    assert stats[0] + stats[1] > 0.5 * sum(stats.values())   # it does decide most segments
