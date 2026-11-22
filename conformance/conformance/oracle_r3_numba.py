"""The gridlines (R3) oracle, compiled: the same visibility as oracle_r3.viewshed,
far faster (tests/test_oracle_r3.py holds it to that).

Each grid-line crossing's gap is computed in floats with certify.py's bounds on its
distance from the exact gap of the patch next to it (whose polynomial equals the
surface on the shared edge), widened for the float crossing's distance from the
exact one (delta). A crossing is then certainly clear, certainly blocking, or
undecided; undecided ones (ties, crossings within 16 delta of either end, crossings
whose other coordinate is within its error of a grid line) are settled one by one in
exact arithmetic, as are crossings at the observer when its coordinate's error
straddles a grid line (one list might have them and the other not).
"""

import math
from fractions import Fraction

import numpy as np
from numba import njit, prange

from . import certify, oracle, oracle_numba
from .oracle_r3 import _cell_index, _patch_ok

VERSION = "r3-1"
U = certify.U
_C = oracle_numba._C
make_cell = _C["make_cell"]
gap_and_error = _C["gap_and_error"]
derivatives = _C["derivatives"]
near_observer_flag = _C["near_observer_flag"]
_axis = oracle_numba._axis


@njit(cache=True, inline="always")
def _ok(z, r0, c0):
    rows, cols = z.shape
    ra = min(max(r0, 0), rows - 1)
    rb = min(max(r0 + 1, 0), rows - 1)
    ca = min(max(c0, 0), cols - 1)
    cb = min(max(c0 + 1, 0), cols - 1)
    return not (math.isnan(z[ra, ca]) or math.isnan(z[ra, cb]) or math.isnan(z[rb, ca])
                or math.isnan(z[rb, cb]))


@njit(cache=True)
def _cell(z, cell, und_ax, und_k):
    """(state, n): state 0 clear, 1 blocked, 2 undecided; the undecided crossings
    (axis 0 = column line u = k, 1 = row line v = k) go to und_ax/und_k[:n]."""
    rows, cols = z.shape
    delta = cell.delta
    n = 0
    for axis in range(2):
        if axis == 0:
            p0, d, q0, dq, e_q0, e_dq = cell.ue, cell.du, cell.ve, cell.dv, cell.e_ve, cell.e_dv
        else:
            p0, d, q0, dq, e_q0, e_dq = cell.ve, cell.dv, cell.ue, cell.du, cell.e_ue, cell.e_du
        if d == 0:
            continue
        k0, step, cnt = _axis(p0, d)
        for i in range(cnt):
            k = k0 + i * step
            t = (k - p0) / d
            undecided = not math.isfinite(delta) or t <= 16 * delta or t >= 1 - 16 * delta
            if not undecided:
                q = q0 + t * dq
                e_q = (3 * U * (abs(q0) + abs(t * dq)) * (1 + 4 * U) + e_q0 + abs(t) * e_dq
                       + abs(dq) * delta)
                m = math.floor(q)
                if q - m <= 2 * e_q or m + 1 - q <= 2 * e_q:
                    undecided = True
            if undecided:
                und_ax[n] = axis
                und_k[n] = k
                n += 1
                continue
            # patches just before and after the crossing
            kb = k - 1 if d > 0 else k
            ka = k if d > 0 else k - 1
            if axis == 0:
                rb_, cb_, ra_, ca_ = m, kb, m, ka
            else:
                rb_, cb_, ra_, ca_ = kb, m, ka, m
            ok_a = _ok(z, ra_, ca_)
            if not ok_a and not _ok(z, rb_, cb_):
                continue                                    # no data: never blocks
            r0, c0 = (ra_, ca_) if ok_a else (rb_, cb_)
            ra = min(max(r0, 0), rows - 1)
            rb = min(max(r0 + 1, 0), rows - 1)
            ca = min(max(c0, 0), cols - 1)
            cb = min(max(c0 + 1, 0), cols - 1)
            z00, z01, z10, z11 = z[ra, ca], z[ra, cb], z[rb, ca], z[rb, cb]
            g, err, e_tc, e_tr = gap_and_error(cell, z00, z01, z10, z11, c0, r0, t)
            g1, e_g1, g2, e_g2 = derivatives(cell, z00, z01, z10, z11, c0, r0, t, e_tc, e_tr)
            shift = (abs(g1) + e_g1 + 2 * (abs(g2) + e_g2) * delta) * delta
            slop = 16 * U * (abs(g) + err + shift)
            if g - err - shift - slop > 0:
                continue
            if g + err + shift + slop < 0:
                return 1, 0
            und_ax[n] = axis
            und_k[n] = k
            n += 1
    if near_observer_flag(cell):
        for axis in range(2):
            p0 = cell.ue if axis == 0 else cell.ve
            k = math.floor(p0 + 0.5)
            for kk in (k - 1, k, k + 1):
                und_ax[n] = axis
                und_k[n] = kk
                n += 1
    return (2 if n else 0), n


@njit(cache=True, parallel=True)
def _pass(z, ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, cells, state):
    rows, cols = z.shape
    for i in prange(cells.shape[0]):
        und_ax = np.empty(rows + cols + 8, np.int64)
        und_k = np.empty(rows + cols + 8, np.int64)
        r = cells[i, 0]
        c = cells[i, 1]
        cell = make_cell(ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, z[r, c], r, c)
        state[i] = _cell(z, cell, und_ax, und_k)[0]


@njit(cache=True)
def _undecided(z, ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, r, c, und_ax, und_k):
    cell = make_cell(ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, z[r, c], r, c)
    return _cell(z, cell, und_ax, und_k)


def _exact_crossing(zf, ex, axis, k, rows, cols):
    """oracle_r3's test for one crossing, exactly: True if it exists and blocks."""
    ue, ve, ze, du, dv, dz, bend = ex
    p0, d = (ue, du) if axis == 0 else (ve, dv)
    if d == 0 or not (min(p0, p0 + d) < k < max(p0, p0 + d)):
        return False
    t = (k - p0) / d
    u, v = ue + t * du, ve + t * dv
    if not (_patch_ok(zf, _cell_index(v, dv, False), _cell_index(u, du, False), rows, cols)
            or _patch_ok(zf, _cell_index(v, dv, True), _cell_index(u, du, True), rows, cols)):
        return False
    q = v if axis == 0 else u
    a = math.floor(q)
    f = q - a
    n_along = rows if axis == 0 else cols
    ia, ib = min(max(a, 0), n_along - 1), min(max(a + 1, 0), n_along - 1)
    h0, h1 = (zf[ia][k], zf[ib][k]) if axis == 0 else (zf[k][ia], zf[k][ib])
    h = h0 if f == 0 else h0 * (1 - f) + h1 * f
    return ze + t * dz - h + bend * t * t < 0


def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None, curvature=False, refraction=0.13, earth_radius=6371008.8):
    """Same signature and result as oracle_r3.viewshed."""
    z = np.ascontiguousarray(z, dtype=float)
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
    res = abs(transform[1])
    rx = certify.Raster(transform, x, y, ground, observer_height, ue, ve, ze)
    th = float(target_height)
    rr, cc = np.nonzero(~np.isnan(z))
    if max_distance is not None:
        keep = np.array([certify.within(rx, ue, ve, res, r, c, max_distance)
                         for r, c in zip(rr.tolist(), cc.tolist())], dtype=bool)
        rr, cc = rr[keep], cc[keep]
    cells = np.stack([rr, cc], axis=1).astype(np.int64)
    state = np.empty(len(rr), np.int8)
    _pass(z, ue, ve, ze, rx.e_ue, rx.e_ve, rx.e_ze, th, drop, res, cells, state)
    blocked = state == 1
    unsure = np.nonzero(state == 2)[0]
    if len(unsure):
        F = Fraction
        zf = [[None if math.isnan(v) else F(v) for v in row] for row in z.tolist()]
        uex, vex = rx.ue_x, rx.ve_x
        zex = ground + F(observer_height)
        fres, fdrop, fth = F(res), F(drop), F(th)
        und_ax = np.empty(rows + cols + 8, np.int64)
        und_k = np.empty(rows + cols + 8, np.int64)
        for j in unsure:
            r, c = int(rr[j]), int(cc[j])
            _, n = _undecided(z, ue, ve, ze, rx.e_ue, rx.e_ve, rx.e_ze, th, drop, res, r, c,
                              und_ax, und_k)
            du, dv = c - uex, r - vex
            bend = fdrop * (du * du + dv * dv) * fres * fres
            ex = (uex, vex, zex, du, dv, zf[r][c] - bend + fth - zex, bend)
            blocked[j] = any(_exact_crossing(zf, ex, int(a), int(k), rows, cols)
                             for a, k in zip(und_ax[:n], und_k[:n]))
    visible = np.zeros(z.shape, dtype=bool)
    visible[rr, cc] = ~blocked
    return visible, len(unsure)
