"""Exact-arithmetic helpers for the bilinear oracles, and the choice of oracle.

The NumPy float pass that used to live here (about 10x oracle.py) was retired when
the oracles moved to certified float decisions (certify.py): oracle_numba compiles
the certifier itself, and without numba oracle.py is the fallback.

What stays: oracle._blocked_exact without converting the whole raster to Fractions
on each call, and the exact check restricted to given patches, which is how the
fast oracle settles a cell: every other segment is certified clear, so checking
the undecided ones (and their neighbours) exactly decides what checking the whole
sightline would.
"""

import math
from fractions import Fraction

from . import oracle


class _LazyFractions:
    """z as oracle._blocked_exact sees it (Fraction per value, NaN kept as the float),
    converting only the values read: _segment_min reads four corners per patch."""

    def __init__(self, z):
        self.z = z
        self.shape = z.shape

    def __getitem__(self, idx):
        v = float(self.z[idx])
        return v if math.isnan(v) else Fraction(v)


def blocked_exact(z, transform, x, y, ground, oh, r, c, th, drop=0.0):
    """oracle._blocked_exact line for line, except that z goes in as _LazyFractions
    instead of converting all of it (1.8 s for 1001 x 1001) on every call."""
    F = Fraction
    ue = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
    ve = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
    zf = _LazyFractions(z)
    ze = ground + F(oh)
    du, dv = F(c) - ue, F(r) - ve
    res = abs(F(transform[1]))
    bend = F(drop) * (du * du + dv * dv) * res * res
    dz = F(z[r, c]) - bend + F(th) - ze
    ts = sorted({F(0), F(1), *oracle._crossings(ue, du), *oracle._crossings(ve, dv)})
    for ta, tb in zip(ts, ts[1:]):
        m = oracle._segment_min(zf, ue, ve, ze, du, dv, dz, ta, tb, bend=bend)
        if m is not None and m < 0:
            return True
    return False


def _exact_setup(z, transform, x, y, ground, oh, r, c, th, drop):
    F = Fraction
    ue = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
    ve = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
    ze = ground + F(oh)
    du, dv = F(c) - ue, F(r) - ve
    res = abs(F(transform[1]))
    bend = F(drop) * (du * du + dv * dv) * res * res
    dz = F(z[r, c]) - bend + F(th) - ze
    return ue, ve, ze, du, dv, dz, bend


def _exact_patch_segment(ue, ve, du, dv, r0, c0):
    """oracle.py's (exact) segment of the sightline inside patch (r0, c0), or None."""
    ta, tb = Fraction(0), Fraction(1)
    for p0, d, k0 in ((ue, du, c0), (ve, dv, r0)):
        if d == 0:
            if not (k0 <= p0 < k0 + 1):            # floor(p0) is the only patch
                return None
            continue
        a, b = (k0 - p0) / d, (k0 + 1 - p0) / d
        ta, tb = max(ta, min(a, b)), min(tb, max(a, b))
    return (ta, tb) if ta < tb else None


def blocked_exact_patches(z, transform, x, y, ground, oh, r, c, th, drop, patches):
    """oracle._blocked_exact, restricted to the given patches (r0, c0)."""
    ue, ve, ze, du, dv, dz, bend = _exact_setup(z, transform, x, y, ground, oh, r, c, th, drop)
    zf = _LazyFractions(z)
    for r0, c0 in patches:
        seg = _exact_patch_segment(ue, ve, du, dv, r0, c0)
        if seg is None:
            continue
        m = oracle._segment_min(zf, ue, ve, ze, du, dv, dz, seg[0], seg[1], bend=bend)
        if m is not None and m < 0:
            return True
    return False


def fastest():
    """The fastest oracle available, all with oracle.py's visibility and clearance:
    oracle_numba when numba is installed (pip install .[fast]), else oracle.py."""
    try:
        from . import oracle_numba
        return oracle_numba
    except ImportError:
        return oracle
