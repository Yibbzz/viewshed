"""The bilinear oracle (oracle.py) compiled with numba.

viewshed(..., clearance=True), the full pass, is bit-identical to oracle.viewshed:
the same loops and the same float operations in the same order (numba compiles
without fastmath: no reassociation, no fused multiply-add, so every operation
rounds as CPython's float does), and the same certifier, compiled from certify.py's
own functions. Visibility, clearance and the count of cells sent to exact
arithmetic all match (tests/test_oracle_fast.py). math.hypot stays in Python
(CPython has its own algorithm), and so does the exact arithmetic.

viewshed(..., clearance=False) decides visibility only, much faster: it skips
blocks of patches whose certified lower bound on the exact gap is > 0, stops at the
first certainly blocking segment, generates boundaries as a stream, and tries the
previous cell's blocker first. Its visibility is identical; its exact-check count
can be lower, since a skipped block is certified as a whole and its slivers never
come up.

Either way, a cell that needs exact arithmetic is checked exactly only on the
patches it is undecided on (and their neighbours): every other segment is
certified clear, so this decides exactly what oracle._blocked_exact does.

Python's min(a, b) keeps a unless b < a; the `if v < m` updates below copy that.
"""

import math
import types

import numpy as np
from numba import njit, prange

from . import certify, oracle
from .oracle_fast import blocked_exact_patches

VERSION = oracle.VERSION


def _compile_certify():
    """certify.py's float functions, compiled: the same code objects with their
    globals pointing at the compiled versions of each other."""
    g = dict(certify.__dict__)
    for name in ("two_sum_err", "two_prod", "make_cell", "crossings_equal",
                 "near_observer_flag", "coord_errors", "gap_and_error", "derivatives",
                 "_blocking_at", "certainly_blocking", "segment"):
        f = getattr(certify, name)
        g[name] = njit(cache=True)(types.FunctionType(f.__code__, g, f.__name__, f.__defaults__,
                                                      f.__closure__))
    return g


_C = _compile_certify()
make_cell = _C["make_cell"]
crossings_equal = _C["crossings_equal"]
near_observer_flag = _C["near_observer_flag"]
certainly_blocking = _C["certainly_blocking"]
segment = _C["segment"]
U = certify.U
BLOCKING, UNDECIDED = certify.BLOCKING, certify.UNDECIDED


@njit(cache=True)
def _crossings_sorted(buf, p0, d):
    """oracle._crossings(p0, d) in increasing t (k runs backwards when d < 0);
    the same floats as there. Returns the count."""
    if d == 0:
        return 0
    p1 = p0 + d
    lo = min(p0, p1)
    hi = max(p0, p1)
    first = math.floor(lo) + 1
    end = math.ceil(hi)
    n = 0
    if d > 0:
        k = first
        while k < end:
            buf[n] = (k - p0) / d
            n += 1
            k += 1
    else:
        k = end - 1
        while k >= first:
            buf[n] = (k - p0) / d
            n += 1
            k -= 1
    return n


@njit(cache=True)
def _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, last, bend):
    """oracle._segment_min; NaN for a patch without data."""
    rows, cols = z.shape
    mid = (ta + tb) / 2
    c0 = math.floor(ue + mid * du)
    r0 = math.floor(ve + mid * dv)
    ra = min(max(r0, 0), rows - 1)
    rb = min(max(r0 + 1, 0), rows - 1)
    ca = min(max(c0, 0), cols - 1)
    cb = min(max(c0 + 1, 0), cols - 1)
    z00 = z[ra, ca]
    z01 = z[ra, cb]
    z10 = z[rb, ca]
    z11 = z[rb, cb]
    if math.isnan(z00) or math.isnan(z01) or math.isnan(z10) or math.isnan(z11):
        return np.nan
    c0f = float(c0)
    r0f = float(r0)

    tc = ue + ta * du - c0f
    tr = ve + ta * dv - r0f
    m = ze + ta * dz - ((z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr) \
        + bend * ta * ta
    if not last:
        tc = ue + tb * du - c0f
        tr = ve + tb * dv - r0f
        v = ze + tb * dz - ((z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr) \
            + bend * tb * tb
        if v < m:
            m = v
    e3 = z00 - z01 - z10 + z11
    g2 = -e3 * du * dv + bend
    if g2 > 0:
        tca = ue + ta * du - c0f
        trb = ve + ta * dv - r0f
        g1 = dz - ((z01 - z00) * du + (z10 - z00) * dv + e3 * (tca * dv + trb * du)) + 2 * bend * ta
        s = -g1 / (2 * g2)
        if 0 < s and s < tb - ta:
            t = ta + s
            tc = ue + t * du - c0f
            tr = ve + t * dv - r0f
            v = ze + t * dz - ((z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr) \
                + bend * t * t
            if v < m:
                m = v
    return m


def _pyramid(z):
    """Max-pyramid over patches, for the pruned pass. Patch (r0, c0), r0 in -1..rows-1,
    sits at [r0 + 1, c0 + 1]; its value is the highest of the four corners the oracle
    reads (clamped at the border), -inf when one is no-data (the oracle skips it).
    Each patch is then widened to its 3 x 3 neighbourhood: a segment's patch comes
    from flooring its midpoint, which on a sliver can land one patch over.
    Returns the levels packed in one array, with their offsets and shapes."""
    rows, cols = z.shape
    r = np.clip(np.arange(-1, rows + 1), 0, rows - 1)
    c = np.clip(np.arange(-1, cols + 1), 0, cols - 1)
    zz = z[np.ix_(r, c)]                                   # (rows + 2, cols + 2)
    q = np.stack([zz[:-1, :-1], zz[:-1, 1:], zz[1:, :-1], zz[1:, 1:]])
    patch = np.where(np.isnan(q).any(axis=0), -np.inf, np.nanmax(np.where(np.isnan(q), -np.inf, q), axis=0))
    pad = np.pad(patch, 1, constant_values=-np.inf)
    level = np.max([pad[a:a + patch.shape[0], b:b + patch.shape[1]]
                    for a in range(3) for b in range(3)], axis=0)  # (rows + 1, cols + 1)
    levels = [level]
    while max(levels[-1].shape) > 1:
        a = levels[-1]
        h, w = (a.shape[0] + 1) // 2, (a.shape[1] + 1) // 2
        b = np.pad(a, ((0, 2 * h - a.shape[0]), (0, 2 * w - a.shape[1])), constant_values=-np.inf)
        levels.append(b.reshape(h, 2, w, 2).max(axis=(1, 3)))
    offs = np.cumsum([0] + [a.size for a in levels]).astype(np.int64)
    shapes = np.array([a.shape for a in levels], dtype=np.int64)
    return np.concatenate([a.ravel() for a in levels]), offs, shapes


@njit(cache=True, inline="always")
def _axis(p0, d):
    """One axis's crossings as a stream: (k of the first in increasing t, step, count).
    The i-th is (k0 + i * step - p0) / d, oracle._crossings' formula."""
    if d == 0:
        return 0, 1, 0
    p1 = p0 + d
    lo = min(p0, p1)
    hi = max(p0, p1)
    first = math.floor(lo) + 1
    end = math.ceil(hi)
    n = max(end - first, 0)
    if d > 0:
        return first, 1, n
    return end - 1, -1, n


@njit(cache=True, inline="always")
def _first_above(k0, step, n, i, p0, d, t):
    """Smallest index >= i whose crossing is > t (values rise with the index)."""
    lo = i
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if (k0 + mid * step - p0) / d > t:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def _boundaries(cell, r, c, rt, ra, rk, ts, fl):
    """certify.boundaries in compiled form: fills ts[:n] (sorted, unique) and fl[:n]
    (flagged), returns n. The raw list is merged in oracle.py's order: a stable sort
    of [0, 1, u crossings by k, v crossings by k], so equal values come 0, 1, u, v."""
    ue, ve, du, dv = cell.ue, cell.ve, cell.du, cell.dv
    uk, us, un = _axis(ue, du)
    vk, vs, vn = _axis(ve, dv)
    ui = 0
    vi = 0
    n = 0
    rt[0] = 0.0
    ra[0] = 0
    rk[0] = 0
    n = 1
    one = False
    while ui < un or vi < vn or not one:
        best = np.inf
        src = -1
        if not one:
            best = 1.0
            src = 1
        if ui < un:
            t = (uk + ui * us - ue) / du
            if t < best:
                best = t
                src = 2
        if vi < vn:
            t = (vk + vi * vs - ve) / dv
            if t < best:
                best = t
                src = 3
        rt[n] = best
        ra[n] = src
        if src == 1:
            rk[n] = 0
            one = True
        elif src == 2:
            rk[n] = uk + ui * us
            ui += 1
        else:
            rk[n] = vk + vi * vs
            vi += 1
        n += 1
    # unique values and flags
    lim = 16 * cell.delta
    m = 0
    ts[0] = rt[0]
    fl[0] = False
    for i in range(1, n):
        if rt[i] != rt[i - 1]:
            m += 1
            ts[m] = rt[i]
            fl[m] = False
        if rt[i] - rt[i - 1] <= lim and not (rt[i] == rt[i - 1] and crossings_equal(
                cell, ra[i - 1], rk[i - 1], ra[i], rk[i], r, c)):
            fl[m] = True
            fl[m - 1 if rt[i] != rt[i - 1] else m] = True
    m += 1
    if near_observer_flag(cell):
        fl[0] = True
    if not math.isfinite(cell.delta):
        for i in range(m):
            fl[i] = True
    return m


@njit(cache=True, parallel=True)
def _full_pass(z, ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, cells, L, state_out, far_out):
    """oracle.viewshed's per-cell loop: clearance, and the certified decision
    (0 visible, 1 blocked, 2 exact check)."""
    rows, cols = z.shape
    for i in prange(cells.shape[0]):
        size = rows + cols + 8
        rt = np.empty(size)
        ra = np.empty(size, np.int64)
        rk = np.empty(size, np.int64)
        ts = np.empty(size)
        fl = np.empty(size, np.bool_)
        r = cells[i, 0]
        c = cells[i, 1]
        cell = make_cell(ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, z[r, c], r, c)
        du, dv, dz, bend = cell.du, cell.dv, cell.dz, cell.bend
        k = _boundaries(cell, r, c, rt, ra, rk, ts, fl)
        Li = L[i]
        t_far = 1 - 0.5 / Li if Li > 0.5 else 0.0
        far = np.inf
        block = False
        undecided = False
        for j in range(k - 1):
            ta = ts[j]
            tb = ts[j + 1]
            last = j == k - 2
            m = _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, last, bend)
            if not math.isnan(m) and ta < t_far:
                if tb <= t_far:
                    mf = m
                else:
                    mf = _segment_min(z, ue, ve, ze, du, dv, dz, ta, t_far, False, bend)
                if mf < far:
                    far = mf
            if fl[j] or fl[j + 1]:
                undecided = True
                continue
            st = segment(cell, z, ta, tb, last)[0]
            if st == BLOCKING:
                block = True
            elif st == UNDECIDED:
                undecided = True
        state_out[i] = 1 if block else (2 if undecided else 0)
        far_out[i] = far


@njit(cache=True)
def _undecided_patches(z, ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, r, c, out):
    """(r0, c0) of every segment of cell (r, c) the certifier leaves undecided
    (flagged ones included, patch from the midpoint). Returns the count."""
    rows, cols = z.shape
    size = rows + cols + 8
    rt = np.empty(size)
    ra = np.empty(size, np.int64)
    rk = np.empty(size, np.int64)
    ts = np.empty(size)
    fl = np.empty(size, np.bool_)
    cell = make_cell(ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, z[r, c], r, c)
    k = _boundaries(cell, r, c, rt, ra, rk, ts, fl)
    n = 0
    for j in range(k - 1):
        ta = ts[j]
        tb = ts[j + 1]
        if fl[j] or fl[j + 1]:
            st = UNDECIDED
            mid = (ta + tb) / 2
            r0 = math.floor(ve + mid * cell.dv)
            c0 = math.floor(ue + mid * cell.du)
        else:
            st, r0, c0 = segment(cell, z, ta, tb, j == k - 2)
        if st == UNDECIDED:
            out[n, 0] = r0
            out[n, 1] = c0
            n += 1
    return n


@njit(cache=True, inline="always")
def _peek(uk, us, un, ui, ue, du, vk, vs, vn, vi, ve, dv, one):
    """The next raw boundary value (inf when none)."""
    best = np.inf
    if not one:
        best = 1.0
    if ui < un:
        best = min(best, (uk + ui * us - ue) / du)
    if vi < vn:
        best = min(best, (vk + vi * vs - ve) / dv)
    return best


@njit(cache=True)
def _group_ok(cell, r, c, has1, hu, ku, hv, kv):
    """Whether the raw boundaries sharing one value (in the order 1, u, v) are
    certainly equal exactly, consecutive pair by pair as certify.boundaries checks."""
    pa = -1
    pk = 0
    for axis in (1, 2, 3):
        present = has1 if axis == 1 else (hu if axis == 2 else hv)
        if not present:
            continue
        k = 0 if axis == 1 else (ku if axis == 2 else kv)
        if pa >= 0 and not crossings_equal(cell, pa, pk, axis, k, r, c):
            return False
        pa = axis
        pk = k
    return True


@njit(cache=True, inline="always")
def _reuse_blocks(cell, z, r0, c0):
    """Whether patch (r0, c0), which blocked the previous cell, certainly blocks this
    sightline too: its exact segment there runs between the face crossings (within
    delta of their floats), and certainly_blocking checks points inside it."""
    rows, cols = z.shape
    ue, ve, du, dv = cell.ue, cell.ve, cell.du, cell.dv
    ta = 0.0
    tb = 1.0
    for i in range(2):
        p0, d, k0, e_p, e_d = ((ue, du, c0, cell.e_ue, cell.e_du) if i == 0 else
                               (ve, dv, r0, cell.e_ve, cell.e_dv))
        if d == 0:
            if not (e_d == 0 and e_p == 0 and math.floor(p0) == k0):
                return False
            continue
        a = (k0 - p0) / d
        b = (k0 + 1 - p0) / d
        ta = max(ta, min(a, b))
        tb = min(tb, max(a, b))
    ra = min(max(r0, 0), rows - 1)
    rb = min(max(r0 + 1, 0), rows - 1)
    ca = min(max(c0, 0), cols - 1)
    cb = min(max(c0 + 1, 0), cols - 1)
    z00 = z[ra, ca]
    z01 = z[ra, cb]
    z10 = z[rb, ca]
    z11 = z[rb, cb]
    if math.isnan(z00) or math.isnan(z01) or math.isnan(z10) or math.isnan(z11):
        return False
    return certainly_blocking(cell, z00, z01, z10, z11, c0, r0, ta, tb)


@njit(cache=True, parallel=True)
def _decide_pass(z, ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, cells, out, pyr, offs, shapes):
    """Visibility without clearance: out = 0 visible, 1 blocked, 2 exact check.
    Walks forward over the boundaries as a stream (oracle.py's floats, order and
    de-duplication, with certify.boundaries' flags computed on the fly), skips any
    block of patches on which the exact gap is certainly > 0, and stops at the first
    certainly blocking segment."""
    nlev = shapes.shape[0]
    chunk = 512
    for q in prange((cells.shape[0] + chunk - 1) // chunk):
      have = False                          # patch that blocked the previous cell
      br0 = 0
      bc0 = 0
      failed = np.full(nlev, -1, np.int64)
      for i in range(q * chunk, min((q + 1) * chunk, cells.shape[0])):
        r = cells[i, 0]
        c = cells[i, 1]
        cell = make_cell(ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, z[r, c], r, c)
        du, dv, dz, bend, delta = cell.du, cell.dv, cell.dz, cell.bend, cell.delta
        if have and _reuse_blocks(cell, z, br0, bc0):
            out[i] = 1
            continue
        for lev in range(nlev):
            failed[lev] = -1
        uk, us, un = _axis(ue, du)
        vk, vs, vn = _axis(ve, dv)
        ui = 0
        vi = 0
        one = False
        lim = 16 * delta
        # line and curvature errors for the block bounds
        e_ze_ = cell.e_ze
        e_dz = cell.e_dz
        bend_lo = bend - cell.e_bend
        # t = 0 and its flag (no crossing is ever at 0 exactly, see certify)
        ta = 0.0
        nxt = _peek(uk, us, un, ui, ue, du, vk, vs, vn, vi, ve, dv, one)
        fa = near_observer_flag(cell) or nxt - ta <= lim or not math.isfinite(delta)
        state = 0
        while True:
            b = _peek(uk, us, un, ui, ue, du, vk, vs, vn, vi, ve, dv, one)
            if b == np.inf:
                break
            # consume the group equal to b
            has1 = False
            hu = False
            hv = False
            ku = 0
            kv = 0
            if not one and b == 1.0:
                one = True
                has1 = True
            if ui < un and (uk + ui * us - ue) / du == b:
                hu = True
                ku = uk + ui * us
                ui += 1
            if vi < vn and (vk + vi * vs - ve) / dv == b:
                hv = True
                kv = vk + vi * vs
                vi += 1
            nxt = _peek(uk, us, un, ui, ue, du, vk, vs, vn, vi, ve, dv, one)
            grouped = (1 if has1 else 0) + (1 if hu else 0) + (1 if hv else 0)
            fb = ((grouped > 1 and not _group_ok(cell, r, c, has1, hu, ku, hv, kv))
                  or b - ta <= lim or nxt - b <= lim or not math.isfinite(delta))
            last = nxt == np.inf
            tb = b
            mid = (ta + tb) / 2
            pr = math.floor(ve + mid * dv) + 1
            pc = math.floor(ue + mid * du) + 1
            skip_to = -1.0
            if math.isfinite(delta):
                for lev in range(nlev - 1, 0, -1):
                    brr = pr >> lev
                    bcc = pc >> lev
                    h = shapes[lev, 0]
                    w = shapes[lev, 1]
                    if brr < 0 or bcc < 0 or brr >= h or bcc >= w:
                        continue
                    key = brr * w + bcc
                    if failed[lev] == key:
                        continue
                    M = pyr[offs[lev] + key]
                    tx = np.inf
                    if du > 0:
                        tx = min(tx, (((bcc + 1) << lev) - 1 - ue) / du)
                    elif du < 0:
                        tx = min(tx, ((bcc << lev) - 1 - ue) / du)
                    if dv > 0:
                        tx = min(tx, (((brr + 1) << lev) - 1 - ve) / dv)
                    elif dv < 0:
                        tx = min(tx, ((brr << lev) - 1 - ve) / dv)
                    if tx <= tb:
                        continue
                    # exact gap on [ta - delta, tx + delta] >= line - M + curvature
                    t1 = ta - delta
                    t2 = tx + delta
                    l1 = ze + t1 * dz
                    l2 = ze + t2 * dz
                    l1 -= e_ze_ + abs(t1) * e_dz + 3 * U * (abs(ze) + abs(t1 * dz))
                    l2 -= e_ze_ + abs(t2) * e_dz + 3 * U * (abs(ze) + abs(t2 * dz))
                    lo = min(l1, l2)
                    if bend_lo >= 0:
                        t0 = max(t1, 0.0)
                        bl = bend_lo * t0 * t0 * (1 - 4 * U)
                    else:
                        bl = bend_lo * max(t1 * t1, t2 * t2) * (1 + 4 * U)
                    lb = lo - M + bl
                    if lb - 16 * U * (abs(lo) + abs(M) + abs(bl)) > 0:
                        skip_to = tx
                        break
                    failed[lev] = key
            if skip_to >= 0:
                # drop every segment ending at or before skip_to. The new ta is the
                # largest raw boundary <= skip_to; its predecessor (for the 16 delta
                # rule) is the largest raw value below it, which is tb or one of the
                # last two consumed on an axis, or 1; its group is whatever equals it.
                ninf = -np.inf
                j = _first_above(uk, us, un, ui, ue, du, skip_to)
                u1 = (uk + (j - 1) * us - ue) / du if j > ui else ninf
                u2 = (uk + (j - 2) * us - ue) / du if j - 2 >= ui else ninf
                ku = uk + (j - 1) * us
                jv = _first_above(vk, vs, vn, vi, ve, dv, skip_to)
                v1 = (vk + (jv - 1) * vs - ve) / dv if jv > vi else ninf
                v2 = (vk + (jv - 2) * vs - ve) / dv if jv - 2 >= vi else ninf
                kv = vk + (jv - 1) * vs
                one_now = (not one) and 1.0 <= skip_to
                o1 = 1.0 if one_now else ninf
                nta = max(tb, max(u1, max(v1, o1)))
                ui = j
                vi = jv
                if one_now:
                    one = True
                if nta == tb:
                    fa = fb
                else:
                    prev = tb
                    for val in (u1, u2, v1, v2, o1):
                        if val < nta and val > prev:
                            prev = val
                    hu = u1 == nta
                    hv = v1 == nta
                    has1 = o1 == nta
                    nxt = _peek(uk, us, un, ui, ue, du, vk, vs, vn, vi, ve, dv, one)
                    grouped = (1 if has1 else 0) + (1 if hu else 0) + (1 if hv else 0)
                    fa = ((grouped > 1 and not _group_ok(cell, r, c, has1, hu, ku, hv, kv))
                          or nta - prev <= lim or nxt - nta <= lim)
                ta = nta
                continue
            if fa or fb:
                state = 2
            else:
                st, r0, c0 = segment(cell, z, ta, tb, last)
                if st == BLOCKING:
                    state = 1
                    have = True
                    br0 = r0
                    bc0 = c0
                    break
                if st == UNDECIDED:
                    state = 2
            ta = tb
            fa = fb
        out[i] = state



def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None, curvature=False, refraction=0.13, earth_radius=6371008.8,
             clearance=True):
    """Same signature and result as oracle.viewshed: (visible, clearance, n_exact).
    With clearance=False the clearance comes back as None, visibility is unchanged,
    and n_exact (cells this run decided exactly) can be lower."""
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
    du, dv = cc - ue, rr - ve
    # math.hypot, not np.hypot (they can differ in the last bit). Only the full pass
    # (t_far) and max_distance need it.
    L = None
    if clearance or max_distance is not None:
        L = np.array([math.hypot(a, b) for a, b in zip(du.tolist(), dv.tolist())])
    if max_distance is not None:
        keep = ~(L * res > max_distance)
        rr, cc, L = rr[keep], cc[keep], L[keep]
    cells = np.stack([rr, cc], axis=1).astype(np.int64)
    state = np.empty(len(rr), np.int8)
    far = None
    if clearance:
        far = np.empty(len(rr))
        _full_pass(z, ue, ve, ze, rx.e_ue, rx.e_ve, rx.e_ze, th, drop, res, cells, L, state, far)
    else:
        pyr, offs, shapes = _pyramid(z)
        _decide_pass(z, ue, ve, ze, rx.e_ue, rx.e_ve, rx.e_ze, th, drop, res, cells, state,
                     pyr, offs, shapes)

    blocked = state == 1
    unsure = np.nonzero(state == 2)[0]
    buf = np.empty((rows + cols + 8, 2), np.int64)
    for k in unsure:
        r, c = int(rr[k]), int(cc[k])
        n = _undecided_patches(z, ue, ve, ze, rx.e_ue, rx.e_ve, rx.e_ze, th, drop, res, r, c, buf)
        patches = sorted({(int(a) + i, int(b) + j) for a, b in buf[:n]
                          for i in (-1, 0, 1) for j in (-1, 0, 1)})
        blocked[k] = blocked_exact_patches(z, transform, x, y, ground, observer_height,
                                           r, c, target_height, drop, patches)
    visible = np.zeros(z.shape, dtype=bool)
    visible[rr, cc] = ~blocked
    clr = None
    if clearance:
        clr = np.full(z.shape, np.nan)
        clr[rr, cc] = far
    return visible, clr, len(unsure)
