"""Exact line-of-sight oracle for the bilinear surface model.

Model (the one `viewshed` declares): heights at cell centres, bilinearly
interpolated between them, clamped at the raster border. NaN never blocks
and is never visible. A cell is visible when the segment from the eye
(surface + observer_height) to its centre (surface + target_height) is
nowhere strictly below the surface.

Method: the sightline is cut where it crosses the lines joining cell centres
(Amanatides-Woo style, so no patch is skipped). Inside one bilinear patch the
gap (line minus surface) is a quadratic in t, so its minimum is found in
closed form. Float first; near-ties are redone in exact rational arithmetic.

Per cell it also returns the clearance: the smallest gap over the sightline,
leaving out the last half cell before the target (which with target_height
= 0 always touches 0 and says nothing about distant blockers).
"""

import math
from fractions import Fraction

import numpy as np

VERSION = "bilinear-1"


def _corners(z, r0, c0):
    rows, cols = z.shape
    ra, rb = min(max(r0, 0), rows - 1), min(max(r0 + 1, 0), rows - 1)
    ca, cb = min(max(c0, 0), cols - 1), min(max(c0 + 1, 0), cols - 1)
    return z[ra, ca], z[ra, cb], z[rb, ca], z[rb, cb]


def _bilinear(cs, tc, tr):
    z00, z01, z10, z11 = cs
    return (z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr


def _crossings(p0, d):
    """Parameters t in (0, 1) where p0 + t*d crosses an integer."""
    if d == 0:
        return []
    lo, hi = sorted((p0, p0 + d))
    return [(k - p0) / d for k in range(math.floor(lo) + 1, math.ceil(hi))]


def _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, last_th=None):
    """Smallest gap on [ta, tb] (None if the patch has no data). Generic over float/Fraction."""
    mid = (ta + tb) / 2
    c0, r0 = math.floor(ue + mid * du), math.floor(ve + mid * dv)
    cs = _corners(z, r0, c0)
    if any(isinstance(v, float) and math.isnan(v) for v in cs):
        return None

    def gap(t):
        return ze + t * dz - _bilinear(cs, ue + t * du - c0, ve + t * dv - r0)

    # On the last segment the end value is target_height >= 0 exactly; a dip below 0
    # before it must show up at ta or at the vertex, so the end is left out.
    vals = [gap(ta)] if last_th is not None else [gap(ta), gap(tb)]
    z00, z01, z10, z11 = cs
    e3 = z00 - z01 - z10 + z11
    g2 = -e3 * du * dv
    if g2 > 0:
        tca, trb = ue + ta * du - c0, ve + ta * dv - r0
        g1 = dz - ((z01 - z00) * du + (z10 - z00) * dv + e3 * (tca * dv + trb * du))
        s = -g1 / (2 * g2)
        if 0 < s < tb - ta:
            vals.append(gap(ta + s))
    return min(vals)


def _blocked_exact(z, transform, x, y, ground, oh, r, c, th):
    F = Fraction
    ue = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
    ve = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
    zf = np.vectorize(lambda v: v if math.isnan(v) else F(v), otypes=[object])(z)
    ze = ground + F(oh)
    du, dv, dz = F(c) - ue, F(r) - ve, F(z[r, c]) + F(th) - ze
    ts = sorted({F(0), F(1), *_crossings(ue, du), *_crossings(ve, dv)})
    for ta, tb in zip(ts, ts[1:]):
        m = _segment_min(zf, ue, ve, ze, du, dv, dz, ta, tb)
        if m is not None and m < 0:
            return True
    return False


def _ground_exact(z, transform, x, y):
    F = Fraction
    u = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
    v = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
    c0, r0 = math.floor(u), math.floor(v)
    cs = _corners(z, r0, c0)
    if any(math.isnan(q) for q in cs):
        return None
    return _bilinear(tuple(F(q) for q in cs), u - c0, v - r0)


def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None):
    """Exact visibility and clearance for every cell. Returns (visible, clearance, n_exact)."""
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
    ground = _ground_exact(z, transform, x, y)
    if ground is None:
        raise ValueError("there is no height data under the observer")
    ze = float(ground) + observer_height
    tol = 1e-9 * (1.0 + np.nanmax(np.abs(z)) + abs(ze))
    res = abs(transform[1])
    visible = np.zeros(z.shape, dtype=bool)
    clearance = np.full(z.shape, np.nan)
    n_exact = 0
    for r in range(rows):
        for c in range(cols):
            if math.isnan(z[r, c]):
                continue
            du, dv = c - ue, r - ve
            L = math.hypot(du, dv)
            if max_distance is not None and L * res > max_distance:
                continue
            dz = z[r, c] + target_height - ze
            ts = sorted({0.0, 1.0, *_crossings(ue, du), *_crossings(ve, dv)})
            t_far = 1 - 0.5 / L if L > 0.5 else 0.0
            mn, far = math.inf, math.inf
            for i, (ta, tb) in enumerate(zip(ts, ts[1:])):
                last = target_height if i == len(ts) - 2 else None
                m = _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, last)
                if m is None:
                    continue
                mn = min(mn, m)
                if ta < t_far:
                    mf = m if tb <= t_far else _segment_min(z, ue, ve, ze, du, dv, dz, ta, t_far)
                    far = min(far, mf)
            if mn < -tol:
                blocked = True
            elif mn > tol:
                blocked = False
            else:
                n_exact += 1
                blocked = _blocked_exact(z, transform, x, y, ground, observer_height,
                                         r, c, target_height)
            visible[r, c] = not blocked
            clearance[r, c] = far
    return visible, clearance, n_exact
