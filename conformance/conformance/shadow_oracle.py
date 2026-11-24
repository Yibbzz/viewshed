"""Exact cast-shadow oracle for the bilinear surface model.

Model (the viewshed's surface): heights at cell centres, bilinear between them, clamped at the
raster border; no-data never blocks and is never lit. A ray meets a patch only along a stretch of
positive length (a grid vertex touched between two patches without surface doesn't block, as in
oracle.py). The sun is the float direction vector (dx, dy, dz) from `sun_vector` (azimuth clockwise
from north and elevation above the horizontal, in degrees, through the raster's transform); every
float is taken at its exact value, so the oracle is exact for that vector. Curvature/refraction:
points lowered by k e^2 at horizontal distance e from the cell (k = drop_factor). A cell T (its
centre, height z_T + target_height) is SHADOWED iff some s > 0 on a positive-length stretch of an
existing patch has
    q(s) = B(c + s dx, r + s dy) - k res^2 (dx^2 + dy^2) s^2 - (z_T + th) - s dz  >  0.
Per patch q is a quadratic in s on a rational interval: exact decisions without square roots.

Method: a float pass (shadow_oracle_fast, numba, when installed) decides each patch with a margin relative to the size
of its terms; anything within the margin, near-degenerate stretches and the open end at s = 0 send
the cell to the exact rational procedure (`hidden_exact`).
"""

import math
from fractions import Fraction as Q

import numpy as np

from .oracle import drop_factor


def sun_vector(transform, azimuth, elevation):
    """(dx, dy, dz): per unit ray parameter, dx columns, dy rows, dz map units up (horizontal step
    of one cell). Azimuth clockwise from north, elevation above the horizontal, degrees."""
    if not (0.0 < elevation < 90.0):
        raise ValueError("elevation must be in (0, 90) degrees")
    # cardinal azimuths exactly (sin(pi) is 1.2e-16 in floats: a ray a hair off the grid line);
    # the crate's Sun::direction uses the same rule, so both are exact for the same vector
    a = math.fmod(float(azimuth), 360.0)
    if a < 0:
        a += 360.0
    if a in (0.0, 90.0, 180.0, 270.0):
        sa, ca = {0.0: (0.0, 1.0), 90.0: (1.0, 0.0), 180.0: (0.0, -1.0), 270.0: (-1.0, 0.0)}[a]
    else:
        sa, ca = math.sin(math.radians(a)), math.cos(math.radians(a))
    res = abs(transform[1])
    dx = sa * (res / transform[1])
    dy = ca * (res / transform[5])
    return dx, dy, math.tan(math.radians(elevation)) * res


def _corners(z, r0, c0):
    rows, cols = z.shape
    ra, rb = min(max(r0, 0), rows - 1), min(max(r0 + 1, 0), rows - 1)
    ca, cb = min(max(c0, 0), cols - 1), min(max(c0 + 1, 0), cols - 1)
    return z[ra, ca], z[ra, cb], z[rb, ca], z[rb, cb]


def _interval(p0, d, k0):
    if d == 0:
        return (Q(0), None) if k0 <= p0 <= k0 + 1 else None
    a, b = (k0 - p0) / d, (k0 + 1 - p0) / d
    return (min(a, b), max(a, b))


def _candidates(rows, cols, r, c, dx, dy):
    """A superset of the patches the ray from (r, c) meets (float walk widened to 3 x 3)."""
    out = set()
    L = math.hypot(dx, dy)
    if L == 0:
        return out
    ux, uy = dx / L, dy / L
    t = 0.0
    while t <= 2.0 * (rows + cols) + 4.0:
        cx, cy = math.floor(c + t * ux), math.floor(r + t * uy)
        if cx < -2 or cy < -2 or cx > cols + 1 or cy > rows + 1:
            break
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if -1 <= cy + a < rows and -1 <= cx + b < cols:
                    out.add((cy + a, cx + b))
        t += 0.25
    return sorted(out)


def hidden_exact(z, r, c, dx, dy, dz, drop, res, th=0.0):
    """Exact (rational): is cell (r, c) shadowed?"""
    rows, cols = z.shape
    zT = Q(float(z[r, c])) + Q(th)
    dxq, dyq, dzq = Q(dx), Q(dy), Q(dz)
    curv = Q(drop) * Q(res) * Q(res) * (dxq * dxq + dyq * dyq)
    for r0, c0 in _candidates(rows, cols, r, c, float(dx), float(dy)):
        zc = _corners(z, r0, c0)
        if any(v != v for v in zc):
            continue
        iu = _interval(Q(c), dxq, Q(c0))
        iv = _interval(Q(r), dyq, Q(r0))
        if iu is None or iv is None:
            continue
        his = [h for h in (iu[1], iv[1]) if h is not None]
        if not his:
            continue
        lo, hi = max(Q(0), iu[0], iv[0]), min(his)
        if hi <= lo:
            continue          # empty or a single point: no positive-length stretch
        z00, z01, z10, z11 = (Q(float(v)) for v in zc)
        e3 = z00 - z01 - z10 + z11
        u0, v0 = Q(c) - c0, Q(r) - r0
        qa = e3 * dxq * dyq - curv
        qb = (z01 - z00) * dxq + (z10 - z00) * dyq + e3 * (u0 * dyq + v0 * dxq) - dzq
        qc = z00 + (z01 - z00) * u0 + (z10 - z00) * v0 + e3 * u0 * v0 - zT
        q = lambda s: (qa * s + qb) * s + qc
        cands = [hi] + ([lo] if lo > 0 else [])
        if qa < 0:
            sv = -qb / (2 * qa)
            if lo < sv < hi:
                cands.append(sv)
        if any(q(s) > 0 for s in cands):
            return True
        if lo == 0 and qc == 0 and (qb > 0 or (qb == 0 and qa > 0)):
            return True       # positive right after s = 0
    return False


try:
    from .shadow_oracle_fast import _float_pass
    HAVE_NUMBA = True
except ImportError:      # pragma: no cover
    HAVE_NUMBA = False


def shadow(z, transform, azimuth, elevation, target_height=0.0, curvature=False, refraction=0.13,
           earth_radius=6371008.8, fast=True):
    """Lit mask (True = lit; no-data False) and the number of cells decided in rational arithmetic."""
    return shadow_vec(z, transform, sun_vector(transform, azimuth, elevation), target_height, curvature,
                      refraction, earth_radius, fast)


def shadow_vec(z, transform, vec, target_height=0.0, curvature=False, refraction=0.13,
               earth_radius=6371008.8, fast=True):
    """shadow() for a direction vector (dx, dy, dz) given directly (exact for those floats)."""
    z = np.ascontiguousarray(z, dtype=float)
    if transform[2] != 0 or transform[4] != 0 or abs(abs(transform[1]) - abs(transform[5])) > 0:
        raise ValueError("needs a north-up raster with square cells")
    if target_height < 0:
        raise ValueError("target_height must be >= 0")
    dx, dy, dz = (float(v) for v in vec)
    if dx == 0.0 and dy == 0.0:
        raise ValueError("the sun direction needs a horizontal component")
    drop = drop_factor(curvature, refraction, earth_radius)
    res = abs(transform[1])
    rows, cols = z.shape
    lit = np.zeros(z.shape, dtype=bool)
    n_exact = 0
    state = _float_pass(z, dx, dy, dz, drop, res, float(target_height)).reshape(z.shape) \
        if (fast and HAVE_NUMBA) else None
    for r in range(rows):
        for c in range(cols):
            if z[r, c] != z[r, c]:
                continue
            s = 2 if state is None else int(state[r, c])
            if s == 2:
                n_exact += 1
                s = 1 if hidden_exact(z, r, c, dx, dy, dz, drop, res, target_height) else 0
            lit[r, c] = s == 0
    return lit, n_exact
