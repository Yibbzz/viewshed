"""Fast float pass for the cast-shadow oracle (numba): decides each cell from per-patch float
evaluations with a margin relative to the size of their terms; returns 2 (undecided) whenever a
patch is within the margin, a stretch is near-degenerate, or T's own patch starts with a zero
slope, and shadow_oracle then decides that cell in rational arithmetic. Held to the rational
procedure on every cell by tests/test_shadow_oracle.py."""

import math

import numpy as np

from numba import njit, prange


@njit(cache=True)
def _float_cell(z, r, c, dx, dy, dz, drop, res, th):
    """1 shadowed, 0 lit, 2 undecided (exact needed): the float pass for one cell."""
    rows, cols = z.shape
    zT = z[r, c] + th
    curv = drop * res * res * (dx * dx + dy * dy)
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    und = False
    t = 0.0
    last_r, last_c = -99, -99
    while t <= 2.0 * (rows + cols) + 4.0:
        cx, cy = math.floor(c + t * ux), math.floor(r + t * uy)
        t += 0.25
        if cx < -2 or cy < -2 or cx > cols + 1 or cy > rows + 1:
            break
        if cx == last_c and cy == last_r:
            continue
        last_r, last_c = cy, cx
        for a in range(-1, 2):
            for b in range(-1, 2):
                r0, c0 = cy + a, cx + b
                if r0 < -1 or c0 < -1 or r0 >= rows or c0 >= cols:
                    continue
                ra, rb = min(max(r0, 0), rows - 1), min(max(r0 + 1, 0), rows - 1)
                ca, cb = min(max(c0, 0), cols - 1), min(max(c0 + 1, 0), cols - 1)
                z00, z01, z10, z11 = z[ra, ca], z[ra, cb], z[rb, ca], z[rb, cb]
                if not (z00 == z00 and z01 == z01 and z10 == z10 and z11 == z11):
                    continue
                lo, hi = 0.0, np.inf
                ok = True
                for p0, d, k0 in ((float(c), dx, float(c0)), (float(r), dy, float(r0))):
                    if d == 0.0:
                        if not (k0 <= p0 <= k0 + 1.0):
                            ok = False
                        continue
                    s1, s2 = (k0 - p0) / d, (k0 + 1.0 - p0) / d
                    lo, hi = max(lo, min(s1, s2)), min(hi, max(s1, s2))
                if not ok or hi < lo - 1e-9:
                    continue
                if hi - lo <= 1e-9 * (1.0 + hi):
                    und = True          # near-degenerate: let the exact procedure decide
                    continue
                e3 = z00 - z01 - z10 + z11
                u0, v0 = c - c0, r - r0
                qa = e3 * dx * dy - curv
                qb = (z01 - z00) * dx + (z10 - z00) * dy + e3 * (u0 * dy + v0 * dx) - dz
                qc = z00 + (z01 - z00) * u0 + (z10 - z00) * v0 + e3 * u0 * v0 - zT
                # margin relative to the terms (each rounded op contributes <= 2^-53 x its size)
                size = (abs(e3 * dx * dy) + curv) * hi * hi + (abs(z01 - z00) * abs(dx) + abs(z10 - z00) * abs(dy)
                        + abs(e3) * (abs(u0 * dy) + abs(v0 * dx)) + abs(dz)) * hi \
                    + abs(z00) + abs(z01) + abs(z10) + abs(z11) + abs(zT) + 1.0
                m = 1e-11 * size
                if lo == 0.0:
                    # T's own patch (T, a grid vertex, is on its boundary; the near face's
                    # parameter (c - c) / d is exactly 0 in floats too): q(0) = B(T) - z_T - th
                    # = -th exactly, so use that instead of the rounded polynomial
                    qc = -th
                    if th == 0.0:
                        # q(s) = s (qa s + qb): positive right after 0 iff qb > 0 (or qb = 0 < qa)
                        if qb > m:
                            return 1
                        if qb >= -m:
                            und = True
                best = -np.inf
                for s in (lo, hi):
                    if s > 0.0:
                        best = max(best, (qa * s + qb) * s + qc)
                if qa < 0.0:
                    sv = -qb / (2.0 * qa)
                    if lo < sv < hi:
                        best = max(best, (qa * sv + qb) * sv + qc)
                if best > m:
                    return 1
                if best > -m:
                    und = True
    return 2 if und else 0

@njit(cache=True, parallel=True)
def _float_pass(z, dx, dy, dz, drop, res, th):
    rows, cols = z.shape
    out = np.zeros(rows * cols, np.int8)
    for i in prange(rows * cols):
        r, c = i // cols, i % cols
        out[i] = -1 if z[r, c] != z[r, c] else _float_cell(z, r, c, dx, dy, dz, drop, res, th)
    return out
