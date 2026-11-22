"""The bilinear oracle (oracle.py) compiled with numba: the same loops, the same
float operations in the same order, so the result is bit-identical to
oracle.viewshed (tests/test_oracle_fast.py holds it to that).

numba compiles without fastmath, so there is no reassociation and no fused
multiply-add: every operation rounds exactly as CPython's float does. Two things
stay in Python because numba's versions could round differently: math.hypot
(CPython has its own algorithm) and the rational-arithmetic tie check.

Python's min(a, b) keeps a unless b < a; the `if v < m` updates below copy that.
"""

import math

import numpy as np
from numba import njit, prange

from . import oracle
from .oracle_fast import WINDOW, blocked_exact_patches

VERSION = oracle.VERSION


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
def _merge(ts, U, nu, V, nv):
    """sorted(set([0, 1, *U, *V])) for ascending U and V; returns the length."""
    ts[0] = 0.0
    k = 1
    i = 0
    j = 0
    one = False
    while i < nu or j < nv or not one:
        best = np.inf
        src = 3
        if i < nu and U[i] < best:
            best = U[i]; src = 0
        if j < nv and V[j] < best:
            best = V[j]; src = 1
        if not one and 1.0 < best:
            best = 1.0; src = 2
        if src == 0:
            i += 1
        elif src == 1:
            j += 1
        else:
            one = True
        if best != ts[k - 1]:
            ts[k] = best
            k += 1
    return k


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


@njit(cache=True, parallel=True)
def _float_pass(z, ue, ve, ze, th, drop, res, max_d, cells, L, mn_out, far_out):
    """oracle.viewshed's per-cell float loop for the listed cells (in parallel:
    each cell is independent, so the result doesn't depend on the split)."""
    rows, cols = z.shape
    for i in prange(cells.shape[0]):
        U = np.empty(cols + 4)
        V = np.empty(rows + 4)
        ts = np.empty(rows + cols + 8)
        r = cells[i, 0]
        c = cells[i, 1]
        du = c - ue
        dv = r - ve
        bend = drop * (du * du + dv * dv) * res * res
        dz = z[r, c] - bend + th - ze
        k = _merge(ts, U, _crossings_sorted(U, ue, du), V, _crossings_sorted(V, ve, dv))
        Li = L[i]
        t_far = 1 - 0.5 / Li if Li > 0.5 else 0.0
        mn = np.inf
        far = np.inf
        for j in range(k - 1):
            ta = ts[j]
            tb = ts[j + 1]
            m = _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, j == k - 2, bend)
            if math.isnan(m):
                continue
            if m < mn:
                mn = m
            if ta < t_far:
                if tb <= t_far:
                    mf = m
                else:
                    mf = _segment_min(z, ue, ve, ze, du, dv, dz, ta, t_far, False, bend)
                if mf < far:
                    far = mf
        mn_out[i] = mn
        far_out[i] = far


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


@njit(cache=True, parallel=True)
def _float_pass_pruned(z, ue, ve, ze, th, drop, res, cells, L, mn_out, far_out,
                       pyr, offs, shapes, margin):
    """_float_pass with the same results, walking each sightline from the target
    back to the observer and skipping blocks of patches whose lower bound on the gap
    is more than `margin` above both running minima. A skipped segment's computed
    value is then strictly above them, so the minima (and their bits) don't change.
    Walking backwards, ties keep the later-walked value (<=): that is the earlier
    one in t, which is what Python's min keeps in oracle.py."""
    rows, cols = z.shape
    nlev = shapes.shape[0]
    for i in prange(cells.shape[0]):
        U = np.empty(cols + 4)
        V = np.empty(rows + 4)
        ts = np.empty(rows + cols + 8)
        r = cells[i, 0]
        c = cells[i, 1]
        du = c - ue
        dv = r - ve
        bend = drop * (du * du + dv * dv) * res * res
        dz = z[r, c] - bend + th - ze
        k = _merge(ts, U, _crossings_sorted(U, ue, du), V, _crossings_sorted(V, ve, dv))
        Li = L[i]
        t_far = 1 - 0.5 / Li if Li > 0.5 else 0.0
        mn = np.inf
        far = np.inf
        j = k - 2
        while j >= 0:
            ta = ts[j]
            tb = ts[j + 1]
            if ta < t_far and far < np.inf:
                thr = max(mn, far) + margin
                mid = (ta + tb) / 2
                pr = math.floor(ve + mid * dv) + 1
                pc = math.floor(ue + mid * du) + 1
                skip_to = -1.0
                for lev in range(nlev - 1, 0, -1):
                    br = pr >> lev
                    bc = pc >> lev
                    h = shapes[lev, 0]
                    w = shapes[lev, 1]
                    if br < 0 or bc < 0 or br >= h or bc >= w:
                        continue
                    M = pyr[offs[lev] + br * w + bc]
                    # where the sightline enters this block (forward in t); the same
                    # (k - p0) / d formula as the crossings, so it lands on a ts value
                    te = 0.0
                    if du > 0:
                        te = max(te, ((bc << lev) - 1 - ue) / du)
                    elif du < 0:
                        te = max(te, (((bc + 1) << lev) - 1 - ue) / du)
                    if dv > 0:
                        te = max(te, ((br << lev) - 1 - ve) / dv)
                    elif dv < 0:
                        te = max(te, (((br + 1) << lev) - 1 - ve) / dv)
                    if te >= ta:
                        continue                   # the block holds no more than this segment
                    lo_line = min(ze + te * dz, ze + tb * dz)
                    lo_bend = bend * te * te if bend >= 0 else bend * tb * tb
                    if lo_line - M + lo_bend > thr:
                        skip_to = te
                        break
                if skip_to >= 0:
                    # drop every segment with ta >= skip_to
                    while j >= 0 and ts[j] >= skip_to:
                        j -= 1
                    continue
            m = _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, j == k - 2, bend)
            if not math.isnan(m):
                if m <= mn:
                    mn = m
                if ta < t_far:
                    if tb <= t_far:
                        mf = m
                    else:
                        mf = _segment_min(z, ue, ve, ze, du, dv, dz, ta, t_far, False, bend)
                    if mf <= far:
                        far = mf
            j -= 1
        mn_out[i] = mn
        far_out[i] = far


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


@njit(cache=True, inline="always")
def _patch_segment(ue, ve, du, dv, r0, c0):
    """(ta, tb) of the sightline's segment inside patch (r0, c0), exactly as in
    oracle.py's boundary list, or (-1, -1) when that can't be guaranteed. Inside a
    patch there are no other boundaries, so the segment runs from the entry to the
    exit crossing; both come from the crossings' (k - p0) / d. Patches near either
    end are left out: there 0, 1 or a crossing rounded past 1 can take part."""
    if du > 0:
        tu0 = (c0 - ue) / du
        tu1 = (c0 + 1 - ue) / du
    elif du < 0:
        tu0 = (c0 + 1 - ue) / du
        tu1 = (c0 - ue) / du
    else:
        if not (c0 < ue < c0 + 1):
            return -1.0, -1.0
        tu0 = -np.inf
        tu1 = np.inf
    if dv > 0:
        tv0 = (r0 - ve) / dv
        tv1 = (r0 + 1 - ve) / dv
    elif dv < 0:
        tv0 = (r0 + 1 - ve) / dv
        tv1 = (r0 - ve) / dv
    else:
        if not (r0 < ve < r0 + 1):
            return -1.0, -1.0
        tv0 = -np.inf
        tv1 = np.inf
    ta = max(tu0, tv0)
    tb = min(tu1, tv1)
    if not (1e-6 < ta < tb < 1 - 1e-6):
        return -1.0, -1.0
    mid = (ta + tb) / 2
    if math.floor(ue + mid * du) != c0 or math.floor(ve + mid * dv) != r0:
        return -1.0, -1.0
    return ta, tb


@njit(cache=True, parallel=True)
def _decide_pass(z, ue, ve, ze, th, drop, res, cells, out, pyr, offs, shapes, tol, margin):
    """Visibility without clearance: out = 0 visible (every segment > tol), 1 blocked
    (some segment < -tol), 2 for the exact check. That is oracle.py's decision on its
    minimum, so it agrees cell for cell. Walks forward, generating the boundaries as
    a stream (the same floats, order and de-duplication as sorted(set(...))), skips
    blocks whose lower bound on the gap is above tol + margin (their segments can't
    change the decision), and stops at the first segment below -tol."""
    nlev = shapes.shape[0]
    chunk = 512
    for q in prange((cells.shape[0] + chunk - 1) // chunk):
      br0 = -(1 << 40)                     # patch that blocked the previous cell
      bc0 = -(1 << 40)
      for i in range(q * chunk, min((q + 1) * chunk, cells.shape[0])):
        r = cells[i, 0]
        c = cells[i, 1]
        du = c - ue
        dv = r - ve
        bend = drop * (du * du + dv * dv) * res * res
        dz = z[r, c] - bend + th - ze
        # blocker reuse: the previous cell's blocker often hides this one too
        if br0 > -(1 << 40):
            sa, sb = _patch_segment(ue, ve, du, dv, br0, bc0)
            if sa >= 0:
                m = _segment_min(z, ue, ve, ze, du, dv, dz, sa, sb, False, bend)
                if m < -tol:
                    out[i] = 1
                    continue
        uk, us, un = _axis(ue, du)
        vk, vs, vn = _axis(ve, dv)
        ui = 0
        vi = 0
        one = False                      # 1.0 emitted yet
        thr = tol + margin
        failed = np.full(nlev, -1, np.int64)   # last block tried in vain, per level
        state = 0
        ta = 0.0
        # next boundary after ta
        while True:
            best = np.inf
            if ui < un:
                best = min(best, (uk + ui * us - ue) / du)
            if vi < vn:
                best = min(best, (vk + vi * vs - ve) / dv)
            if not one:
                best = min(best, 1.0)
            if best == np.inf:
                break
            # consume everything equal to best (the set() drops repeats)
            if ui < un and (uk + ui * us - ue) / du == best:
                ui += 1
            if vi < vn and (vk + vi * vs - ve) / dv == best:
                vi += 1
            if not one and best == 1.0:
                one = True
            if best == ta:
                continue
            tb = best
            last = ui >= un and vi >= vn and one
            mid = (ta + tb) / 2
            pr = math.floor(ve + mid * dv) + 1
            pc = math.floor(ue + mid * du) + 1
            skip_to = -1.0
            for lev in range(nlev - 1, 0, -1):
                br = pr >> lev
                bc = pc >> lev
                h = shapes[lev, 0]
                w = shapes[lev, 1]
                if br < 0 or bc < 0 or br >= h or bc >= w:
                    continue
                key = br * w + bc
                if failed[lev] == key:
                    continue
                M = pyr[offs[lev] + key]
                tx = np.inf
                if du > 0:
                    tx = min(tx, (((bc + 1) << lev) - 1 - ue) / du)
                elif du < 0:
                    tx = min(tx, ((bc << lev) - 1 - ue) / du)
                if dv > 0:
                    tx = min(tx, (((br + 1) << lev) - 1 - ve) / dv)
                elif dv < 0:
                    tx = min(tx, ((br << lev) - 1 - ve) / dv)
                if tx <= tb:
                    continue
                lo_line = min(ze + ta * dz, ze + tx * dz)
                lo_bend = bend * ta * ta if bend >= 0 else bend * tx * tx
                if lo_line - M + lo_bend > thr:
                    skip_to = tx
                    break
                failed[lev] = key
            if skip_to >= 0:
                # drop every segment with tb <= skip_to: the new ta is the last
                # boundary <= skip_to, and the stream resumes after it
                nta = tb
                j = _first_above(uk, us, un, ui, ue, du, skip_to)
                if j > ui:
                    nta = max(nta, (uk + (j - 1) * us - ue) / du)
                    ui = j
                j = _first_above(vk, vs, vn, vi, ve, dv, skip_to)
                if j > vi:
                    nta = max(nta, (vk + (j - 1) * vs - ve) / dv)
                    vi = j
                if not one and 1.0 <= skip_to:
                    one = True
                    nta = max(nta, 1.0)
                ta = nta
                continue
            m = _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, last, bend)
            if not math.isnan(m):
                if m < -tol:
                    state = 1
                    br0 = math.floor(ve + mid * dv)
                    bc0 = math.floor(ue + mid * du)
                    break
                if not (m > tol):
                    state = 2
            ta = tb
        out[i] = state


@njit(cache=True)
def _near_zero(z, ue, ve, ze, th, drop, res, r, c, window, out):
    """oracle_fast.near_zero_patches without the neighbours: (r0, c0) of every float
    segment of cell (r, c) whose minimum is within `window` of 0. Returns the count."""
    rows, cols = z.shape
    U = np.empty(cols + 4)
    V = np.empty(rows + 4)
    ts = np.empty(rows + cols + 8)
    du = c - ue
    dv = r - ve
    bend = drop * (du * du + dv * dv) * res * res
    dz = z[r, c] - bend + th - ze
    k = _merge(ts, U, _crossings_sorted(U, ue, du), V, _crossings_sorted(V, ve, dv))
    n = 0
    for j in range(k - 1):
        ta = ts[j]
        tb = ts[j + 1]
        m = _segment_min(z, ue, ve, ze, du, dv, dz, ta, tb, j == k - 2, bend)
        if not math.isnan(m) and m <= window:
            mid = (ta + tb) / 2
            out[n, 0] = math.floor(ve + mid * dv)
            out[n, 1] = math.floor(ue + mid * du)
            n += 1
    return n


def _exact(z, transform, x, y, ground, oh, ue, ve, ze, th, drop, res, tol, r, c):
    """oracle_fast's candidate-patch exact check, candidates found in compiled code."""
    buf = np.empty((z.shape[0] + z.shape[1] + 8, 2), np.int64)
    n = _near_zero(z, ue, ve, ze, th, drop, res, r, c, WINDOW * tol, buf)
    patches = sorted({(int(a) + i, int(b) + j) for a, b in buf[:n] for i in (-1, 0, 1)
                      for j in (-1, 0, 1)})
    return blocked_exact_patches(z, transform, x, y, ground, oh, r, c, th, drop, patches)


def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None, curvature=False, refraction=0.13, earth_radius=6371008.8,
             prune=True, clearance=True):
    """Same signature and result as oracle.viewshed: (visible, clearance, n_exact).
    `prune` only changes the speed. With clearance=False the clearance comes back as
    None and visibility and n_exact (unchanged) are computed much faster."""
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
    tol = 1e-9 * (1.0 + np.nanmax(np.abs(z)) + abs(ze))
    res = abs(transform[1])

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
    if not clearance:
        pyr, offs, shapes = _pyramid(z)
        state = np.empty(len(rr), np.int8)
        _decide_pass(z, ue, ve, ze, float(target_height), drop, res, cells, state,
                     pyr, offs, shapes, tol, tol)
        blocked = state == 1
        unsure = np.nonzero(state == 2)[0]
        for k in unsure:
            blocked[k] = _exact(z, transform, x, y, ground, observer_height, ue, ve, ze,
                                float(target_height), drop, res, tol, int(rr[k]), int(cc[k]))
        visible = np.zeros(z.shape, dtype=bool)
        visible[rr, cc] = ~blocked
        return visible, None, len(unsure)
    mn = np.empty(len(rr))
    far = np.empty(len(rr))
    if prune:
        pyr, offs, shapes = _pyramid(z)
        _float_pass_pruned(z, ue, ve, ze, float(target_height), drop, res, cells, L, mn, far,
                           pyr, offs, shapes, tol)
    else:
        _float_pass(z, ue, ve, ze, float(target_height), drop, res, 0.0, cells, L, mn, far)

    visible = np.zeros(z.shape, dtype=bool)
    clearance = np.full(z.shape, np.nan)
    blocked = mn < -tol
    unsure = np.nonzero(~(mn < -tol) & ~(mn > tol))[0]
    for k in unsure:
        blocked[k] = _exact(z, transform, x, y, ground, observer_height, ue, ve, ze,
                                float(target_height), drop, res, tol, int(rr[k]), int(cc[k]))
    visible[rr, cc] = ~blocked
    clearance[rr, cc] = far
    return visible, clearance, len(unsure)
