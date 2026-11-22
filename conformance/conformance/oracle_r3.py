"""Exact oracle for the gridlines model (Franklin & Ray's R3), on our setup.

The literature's standard "exact" viewshed (R3; Haverkort & Toma's gridlines model)
checks the terrain only where the sightline crosses a grid line, the lines joining
cell centres, with the height there linearly interpolated between the two grid
points on that line. Along a grid line the bilinear surface is exactly that linear
interpolation, so this is the bilinear model sampled only at grid-line crossings:
between crossings the terrain is treated as a polyline. The full bilinear model
(oracle.py) also checks each cell's interior, where the height along the sightline
is a quadratic that can rise above both crossings, so a cell hidden here is hidden
there too, never the other way round.

Everything else is as oracle.py: the eye observer_height above the bilinear
surface at the observer's exact point, the target target_height above the cell
centre, every point lowered by the curvature drop at its horizontal distance,
no-data never blocks and is never visible (a crossing counts only when the patch
just before or just after it on the sightline has data, as in oracle.py),
max_distance compared exactly (certify.within).

viewshed() computes in exact rational arithmetic throughout: simple and slow, the
reference for oracle_r3_numba.py (when it exists) and for small comparisons.
"""

import math
from fractions import Fraction

import numpy as np

from . import certify, oracle

VERSION = "r3-1"


def _cell_index(x, d, after):
    """The patch index along one axis just before (after=False) or just after the
    point x, moving with velocity d."""
    k = math.floor(x)
    if x == k and d != 0 and (d > 0) != after:
        return k - 1
    return k


def _patch_ok(zf, r0, c0, rows, cols):
    """Whether patch (r0, c0) has data at all four (clamped) corners."""
    ra, rb = min(max(r0, 0), rows - 1), min(max(r0 + 1, 0), rows - 1)
    ca, cb = min(max(c0, 0), cols - 1), min(max(c0 + 1, 0), cols - 1)
    return None not in (zf[ra][ca], zf[ra][cb], zf[rb][ca], zf[rb][cb])


def _blocked(zf, ue, ve, ze, du, dv, dz, bend, rows, cols):
    """True if some grid-line crossing strictly inside the sightline has the terrain
    strictly above it (everything exact). A crossing counts when the patch just
    before or just after it on the sightline has data, as in oracle.py, where the
    crossing is an end of those patches' segments."""
    for p0, d, axis in ((ue, du, 0), (ve, dv, 1)):
        if d == 0:
            continue
        lo, hi = min(p0, p0 + d), max(p0, p0 + d)
        for k in range(math.floor(lo) + 1, math.ceil(hi)):
            t = (k - p0) / d
            u, v = ue + t * du, ve + t * dv
            if not (_patch_ok(zf, _cell_index(v, dv, False), _cell_index(u, du, False), rows, cols)
                    or _patch_ok(zf, _cell_index(v, dv, True), _cell_index(u, du, True),
                                 rows, cols)):
                continue
            # the height on grid line k: linear between its two grid points
            q = v if axis == 0 else u
            a = math.floor(q)
            f = q - a
            n_along = rows if axis == 0 else cols
            ia, ib = min(max(a, 0), n_along - 1), min(max(a + 1, 0), n_along - 1)
            h0, h1 = (zf[ia][k], zf[ib][k]) if axis == 0 else (zf[k][ia], zf[k][ib])
            h = h0 if f == 0 else h0 * (1 - f) + h1 * f
            if ze + t * dz - h + bend * t * t < 0:
                return True
    return False


def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None, curvature=False, refraction=0.13, earth_radius=6371008.8):
    """Exact visibility in the gridlines (R3) model, every cell. Returns a bool array."""
    z = np.asarray(z, dtype=float)
    if transform[2] != 0 or transform[4] != 0 or abs(abs(transform[1]) - abs(transform[5])) > 0:
        raise ValueError("needs a north-up raster with square cells")
    if observer_height < 0 or target_height < 0:
        raise ValueError("heights must be >= 0")
    rows, cols = z.shape
    x, y = float(observer[0]), float(observer[1])
    uef = (x - transform[0]) / transform[1] - 0.5
    vef = (y - transform[3]) / transform[5] - 0.5
    if not (-0.5 <= uef <= cols - 0.5 and -0.5 <= vef <= rows - 0.5):
        raise ValueError("the observer is outside the raster")
    ground = oracle._ground_exact(z, transform, x, y)
    if ground is None:
        raise ValueError("there is no height data under the observer")
    F = Fraction
    ue = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
    ve = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
    ze = ground + F(observer_height)
    res = abs(F(transform[1]))
    drop = F(oracle.drop_factor(curvature, refraction, earth_radius))
    th = F(target_height)
    zf = [[None if math.isnan(v) else F(v) for v in row] for row in z.tolist()]
    raster = None
    if max_distance is not None:
        zef = float(ground) + observer_height
        raster = certify.Raster(transform, x, y, ground, observer_height, uef, vef, zef)
    visible = np.zeros(z.shape, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if zf[r][c] is None:
                continue
            if raster is not None and not certify.within(raster, uef, vef, abs(transform[1]),
                                                         r, c, max_distance):
                continue
            du, dv = c - ue, r - ve
            bend = drop * (du * du + dv * dv) * res * res
            dz = zf[r][c] - bend + th - ze
            visible[r, c] = not _blocked(zf, ue, ve, ze, du, dv, dz, bend, rows, cols)
    return visible
