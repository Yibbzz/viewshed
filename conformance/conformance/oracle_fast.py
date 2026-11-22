"""The bilinear oracle (oracle.py), vectorised over cells with NumPy.

Same model, same method, and the same float operations in the same order, so the
result is bit-identical to oracle.viewshed: visibility, clearance and the count of
cells decided in exact arithmetic (tests/test_oracle_fast.py holds it to that).
NumPy's elementwise float64 operations are the IEEE operations Python floats use,
so only the evaluation order has to match. Near-ties still go to oracle.py's
rational-arithmetic check, unchanged.

How: every sightline's segment boundaries (0, 1 and its crossings of the lines
joining cell centres) go into one row of a padded array, sorted, and each
segment's minimum gap is computed for all rows at once.
"""

import math
from fractions import Fraction

import numpy as np

from . import oracle

VERSION = oracle.VERSION     # identical results, so the same cache key

_CHUNK = 1 << 21             # padded segment slots per batch (memory bound)


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


# Exact check on candidate patches only ------------------------------------------
#
# oracle._blocked_exact re-runs the whole sightline in Fraction arithmetic. But a
# cell only reaches it when its float minimum is within tol of 0, and the patches
# that can decide it are the ones whose float minimum is near 0 too: elsewhere the
# float minimum is far from 0 and its error is many orders of magnitude smaller.
# oracle.py already trusts the float minimum whenever it is beyond tol; here it is
# trusted only beyond WINDOW * tol, and each candidate's 8 neighbours are checked
# too (a float segment's patch comes from its midpoint and can be one off on a
# sliver). The candidates are then checked exactly, on exactly the segment
# oracle.py would use: inside a patch there are no other boundaries, so the segment
# runs from the entry to the exit crossing (or 0 / 1), with no list to sort.

WINDOW = 1000


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


def near_zero_patches(z, ue, ve, ze, th, drop, res, r, c, window):
    """Patches (with their 8 neighbours) where the float pass's minimum for cell
    (r, c) is within `window` of 0: oracle.py's float loop, in Python."""
    du, dv = c - ue, r - ve
    bend = drop * (du * du + dv * dv) * res * res
    dz = z[r, c] - bend + th - ze
    ts = sorted({0.0, 1.0, *oracle._crossings(ue, du), *oracle._crossings(ve, dv)})
    out = set()
    for i, (ta, tb) in enumerate(zip(ts, ts[1:])):
        last = th if i == len(ts) - 2 else None
        m = oracle._segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, last, bend)
        if m is not None and m <= window:
            mid = (ta + tb) / 2
            r0, c0 = math.floor(ve + mid * dv), math.floor(ue + mid * du)
            out.update((r0 + a, c0 + b) for a in (-1, 0, 1) for b in (-1, 0, 1))
    return sorted(out)


def _crossings(p0, d):
    """oracle._crossings for many sightlines: (n, k) array of t, +inf padded.
    lo, hi and (k - p0) / d are computed exactly as there."""
    p1 = p0 + d
    lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
    first = np.floor(lo) + 1
    count = np.where(d == 0, 0, np.ceil(hi) - first).astype(np.int64)
    k = first[:, None] + np.arange(max(int(count.max(initial=0)), 0))[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (k - p0[:, None]) / d[:, None]
    return np.where(np.arange(k.shape[1])[None, :] < count[:, None], t, np.inf)


def _gather(z, r0, c0):
    rows, cols = z.shape
    ra, rb = np.clip(r0, 0, rows - 1), np.clip(r0 + 1, 0, rows - 1)
    ca, cb = np.clip(c0, 0, cols - 1), np.clip(c0 + 1, 0, cols - 1)
    return z[ra, ca], z[ra, cb], z[rb, ca], z[rb, cb]


def _segment_min(z, ue, ve, ze, du, dv, dz, bend, ta, tb, last):
    """oracle._segment_min elementwise. Arrays broadcast; `last` drops gap(tb).
    Returns the minimum, NaN where the patch has no data."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _segment_min_(z, ue, ve, ze, du, dv, dz, bend, ta, tb, last)


def _segment_min_(z, ue, ve, ze, du, dv, dz, bend, ta, tb, last):
    mid = (ta + tb) / 2
    c0 = np.floor(ue + mid * du)
    r0 = np.floor(ve + mid * dv)
    z00, z01, z10, z11 = _gather(z, r0.astype(np.int64), c0.astype(np.int64))
    nodata = np.isnan(z00) | np.isnan(z01) | np.isnan(z10) | np.isnan(z11)

    def gap(t):
        tc = ue + t * du - c0
        tr = ve + t * dv - r0
        bil = (z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr
        return ze + t * dz - bil + bend * t * t

    m = gap(ta)
    m = np.where(last, m, np.minimum(m, gap(tb)))
    e3 = z00 - z01 - z10 + z11
    g2 = -e3 * du * dv + bend
    tca, trb = ue + ta * du - c0, ve + ta * dv - r0
    g1 = dz - ((z01 - z00) * du + (z10 - z00) * dv + e3 * (tca * dv + trb * du)) + 2 * bend * ta
    s = -g1 / (2 * g2)
    vertex = (g2 > 0) & (0 < s) & (s < tb - ta)
    m = np.where(vertex, np.minimum(m, gap(ta + s)), m)
    return np.where(nodata, np.nan, m)


def _batch(z, ue, ve, ze, du, dv, L, bend, dz):
    """Float pass for a batch of cells: (min gap, clearance) per cell."""
    t = np.concatenate([np.zeros((len(du), 1)), np.ones((len(du), 1)),
                        _crossings(np.full(len(du), ue), du),
                        _crossings(np.full(len(dv), ve), dv)], axis=1)
    t.sort(axis=1)
    ta, tb = t[:, :-1], t[:, 1:]
    real = np.isfinite(tb) & (tb > ta)            # the set() in oracle.py drops repeats
    # the last real segment of each row (its end value is the target, left out)
    idx = np.arange(ta.shape[1])[None, :]
    last_i = np.where(real, idx, -1).max(axis=1)
    last = idx == last_i[:, None]
    # unused slots get harmless values so nothing below divides by inf
    ta = np.where(real, ta, 0.0)
    tb = np.where(real, tb, 1.0)

    col = lambda a: a[:, None]  # noqa: E731
    m = _segment_min(z, ue, ve, ze, col(du), col(dv), col(dz), col(bend), ta, tb, last)
    ok = real & ~np.isnan(m)
    mn = np.where(ok, m, np.inf).min(axis=1)

    t_far = np.where(L > 0.5, 1 - 0.5 / np.where(L > 0.5, L, 1.0), 0.0)
    tf = col(t_far)
    near = ok & (ta < tf)
    split = near & (tb > tf)
    mf = m
    if split.any():
        rr, kk = np.nonzero(split)
        part = _segment_min(z, ue, ve, ze, du[rr], dv[rr], dz[rr], bend[rr],
                            ta[rr, kk], t_far[rr], np.zeros(len(rr), bool))
        mf = m.copy()
        mf[rr, kk] = part
    # oracle.py would fail on a NaN here (it can't happen: same patch as the full
    # segment); skipping keeps the minimum over real values either way
    far = np.where(near & ~np.isnan(mf), mf, np.inf).min(axis=1)
    return mn, far


def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None, curvature=False, refraction=0.13, earth_radius=6371008.8):
    """Same signature and result as oracle.viewshed: (visible, clearance, n_exact)."""
    z = np.asarray(z, dtype=float)
    if transform[2] != 0 or transform[4] != 0 or abs(abs(transform[1]) - abs(transform[5])) > 0:
        raise ValueError("needs a north-up raster with square cells")
    if observer_height < 0 or target_height < 0:
        raise ValueError("heights must be >= 0")
    rows, cols = z.shape
    x, y = float(observer[0]), float(observer[1])
    ue = (x - transform[0]) / transform[1] - 0.5
    ve = (y - transform[3]) / transform[5] - 0.5
    if not (-0.5 <= ue <= cols - 0.5 and -0.5 <= ve <= rows - 0.5):
        raise ValueError("the observer is outside the raster")
    ground = oracle._ground_exact(z, transform, x, y)
    if ground is None:
        raise ValueError("there is no height data under the observer")
    ze = float(ground) + observer_height
    drop = oracle.drop_factor(curvature, refraction, earth_radius)
    tol = 1e-9 * (1.0 + np.nanmax(np.abs(z)) + abs(ze))
    res = abs(transform[1])

    rr, cc = np.nonzero(~np.isnan(z))
    du = cc - ue
    dv = rr - ve
    # math.hypot, not np.hypot: they can differ in the last bit
    L = np.array([math.hypot(a, b) for a, b in zip(du.tolist(), dv.tolist())])
    if max_distance is not None:
        keep = ~(L * res > max_distance)
        rr, cc, du, dv, L = rr[keep], cc[keep], du[keep], dv[keep], L[keep]
    bend = drop * (du * du + dv * dv) * res * res
    dz = z[rr, cc] - bend + target_height - ze

    # batches of similar sightline length, so the padding stays small
    width = np.abs(du) + np.abs(dv) + 4          # >= segment slots per row
    order = np.argsort(width, kind="stable")
    mn = np.empty(len(rr))
    far = np.empty(len(rr))
    i = 0
    while i < len(order):
        j = i + max(1, int(_CHUNK // width[order[min(i + _CHUNK, len(order)) - 1]]))
        j = min(j, len(order))
        while j - i > 1 and (j - i) * width[order[j - 1]] > _CHUNK:
            j = i + max(1, (j - i) // 2)
        sel = order[i:j]
        mn[sel], far[sel] = _batch(z, ue, ve, ze, du[sel], dv[sel], L[sel], bend[sel], dz[sel])
        i = j

    visible = np.zeros(z.shape, dtype=bool)
    clearance = np.full(z.shape, np.nan)
    blocked = mn < -tol
    unsure = np.nonzero(~(mn < -tol) & ~(mn > tol))[0]
    for k in unsure:
        r, c = int(rr[k]), int(cc[k])
        patches = near_zero_patches(z, ue, ve, ze, target_height, drop, res, r, c, WINDOW * tol)
        blocked[k] = blocked_exact_patches(z, transform, x, y, ground, observer_height,
                                           r, c, target_height, drop, patches)
    visible[rr, cc] = ~blocked
    clearance[rr, cc] = far
    return visible, clearance, len(unsure)


def fastest():
    """The fastest bit-identical oracle available: oracle_numba when numba is
    installed (pip install numba), else this module."""
    try:
        from . import oracle_numba
        return oracle_numba
    except ImportError:
        import sys
        return sys.modules[__name__]
