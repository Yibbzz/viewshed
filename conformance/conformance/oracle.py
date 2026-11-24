"""Exact line-of-sight oracle for the bilinear surface model.

Model (the one `viewshed` declares): heights at cell centres, bilinearly
interpolated between them, clamped at the raster border. NaN never blocks
and is never visible. A cell is visible when the segment from the eye
(surface + observer_height) to its centre (surface + target_height) is
nowhere strictly below the surface.

The surface there is made of patches (the squares between four cell centres);
a patch with a NaN corner has no surface. The sightline meets a patch only
along a stretch of positive length: where it merely touches a grid vertex
(a cell centre) between two patches without surface, that point doesn't
block, even if a neighbouring patch with data contains it. Wherever the sightline crosses patches with data, every point of
those stretches counts, ends included. (Only lattice-exact sightlines at the
edge of no-data areas can tell the difference; the crate does the same.)

Method: the sightline is cut where it crosses the lines joining cell centres
(Amanatides-Woo style, so no patch is skipped). Inside one bilinear patch the
gap (line minus surface) is a quadratic in t, so its minimum is found in
closed form. Float first, with every segment certified blocking or clear by a
rigorous bound on its rounding error (certify.py); a cell with no certainly
blocking segment and some undecided one is redone in exact rational arithmetic.

Per cell it also returns the clearance: the smallest gap over the sightline,
leaving out the last half cell before the target (which with target_height
= 0 always touches 0 and says nothing about distant blockers).
"""

import math
from fractions import Fraction

import numpy as np

from . import certify

VERSION = "bilinear-3"   # 2: earth curvature and refraction; 3: certified float decisions


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


def _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, last_th=None, bend=0):
    """Smallest gap on [ta, tb] (None if the patch has no data). Generic over float/Fraction.

    `bend` is the earth-curvature term: the surface at parameter t is lowered by
    bend * t^2 (= drop factor * horizontal distance^2), which keeps the gap quadratic."""
    mid = (ta + tb) / 2
    c0, r0 = math.floor(ue + mid * du), math.floor(ve + mid * dv)
    cs = _corners(z, r0, c0)
    if any(isinstance(v, float) and math.isnan(v) for v in cs):
        return None

    def gap(t):
        return ze + t * dz - _bilinear(cs, ue + t * du - c0, ve + t * dv - r0) + bend * t * t

    # On the last segment the end value is target_height >= 0 exactly; a dip below 0
    # before it must show up at ta or at the vertex, so the end is left out.
    vals = [gap(ta)] if last_th is not None else [gap(ta), gap(tb)]
    z00, z01, z10, z11 = cs
    e3 = z00 - z01 - z10 + z11
    g2 = -e3 * du * dv + bend
    if g2 > 0:
        tca, trb = ue + ta * du - c0, ve + ta * dv - r0
        g1 = dz - ((z01 - z00) * du + (z10 - z00) * dv + e3 * (tca * dv + trb * du)) + 2 * bend * ta
        s = -g1 / (2 * g2)
        if 0 < s < tb - ta:
            vals.append(gap(ta + s))
    return min(vals)


def _blocked_exact(z, transform, x, y, ground, oh, r, c, th, drop=0.0):
    F = Fraction
    ue = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
    ve = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
    zf = np.vectorize(lambda v: v if math.isnan(v) else F(v), otypes=[object])(z)
    ze = ground + F(oh)
    du, dv = F(c) - ue, F(r) - ve
    res = abs(F(transform[1]))
    bend = F(drop) * (du * du + dv * dv) * res * res
    dz = F(z[r, c]) - bend + F(th) - ze
    ts = sorted({F(0), F(1), *_crossings(ue, du), *_crossings(ve, dv)})
    for ta, tb in zip(ts, ts[1:]):
        m = _segment_min(zf, ue, ve, ze, du, dv, dz, ta, tb, bend=bend)
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


def drop_factor(curvature, refraction, earth_radius):
    """The curvature drop at horizontal distance d is d * d * drop_factor (as in the crate)."""
    return (1.0 - refraction) / (2.0 * earth_radius) if curvature else 0.0


def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None, curvature=False, refraction=0.13, earth_radius=6371008.8):
    """Exact visibility and clearance for every cell. Returns (visible, clearance, n_exact).

    With `curvature`, every surface point and the target are lowered by
    d^2 / (2 R) * (1 - refraction) at horizontal distance d from the observer."""
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
    drop = drop_factor(curvature, refraction, earth_radius)
    res = abs(transform[1])
    raster = certify.Raster(transform, x, y, ground, observer_height, ue, ve, ze)
    visible = np.zeros(z.shape, dtype=bool)
    clearance = np.full(z.shape, np.nan)
    n_exact = 0
    for r in range(rows):
        for c in range(cols):
            if math.isnan(z[r, c]):
                continue
            du, dv = c - ue, r - ve
            L = math.hypot(du, dv)
            if max_distance is not None and not certify.within(raster, ue, ve, res, r, c,
                                                               max_distance):
                continue
            cell = certify.make_cell(ue, ve, ze, raster.e_ue, raster.e_ve, raster.e_ze,
                                     target_height, drop, res, z[r, c], r, c)
            bend, dz = cell.bend, cell.dz
            ts, flagged = certify.boundaries(cell, r, c)
            t_far = 1 - 0.5 / L if L > 0.5 else 0.0
            far = math.inf
            certain_block = undecided = False
            for i, (ta, tb) in enumerate(zip(ts, ts[1:])):
                last = i == len(ts) - 2
                m = _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, target_height if last else None,
                                 bend)
                if m is not None and ta < t_far:
                    mf = m if tb <= t_far else _segment_min(z, ue, ve, ze, du, dv, dz, ta, t_far,
                                                            bend=bend)
                    far = min(far, mf)
                if ta in flagged or tb in flagged:
                    undecided = True
                    continue
                state = certify.segment(cell, z, ta, tb, last)[0]
                if state == certify.BLOCKING:
                    certain_block = True
                elif state == certify.UNDECIDED:
                    undecided = True
            if certain_block:
                blocked = True
            elif undecided:
                n_exact += 1
                blocked = _blocked_exact(z, transform, x, y, ground, observer_height,
                                         r, c, target_height, drop)
            else:
                blocked = False
            visible[r, c] = not blocked
            clearance[r, c] = far
    return visible, clearance, n_exact
