"""Rigorous error bounds for the bilinear oracle's float pass.

oracle.py used to decide a cell in floats when its minimum gap was beyond +-tol,
with tol = 1e-9 of scale: an engineering margin, not a proof. This module bounds,
for one float segment, how far the float numbers can be from the exact ones, and
so certifies a segment as certainly blocking (its exact minimum < 0) or certainly
clear. Anything it can't certify goes to exact arithmetic, which makes the oracle
exact by construction.

"Exact" is oracle.py's exact model (_blocked_exact): every input (z, the
transform, the observer, heights, the curvature drop factor) taken as the exact
value of its float, and everything derived from them in rational arithmetic.

Bounds used (U = 2**-53, every IEEE operation is exact times (1 + e), |e| <= U):
- an expression with at most n operations on any path differs from its exact
  value by at most gamma(n) times the same expression with every value replaced
  by its absolute value (Higham, Accuracy and Stability, ch. 3);
- the bilinear surface is exactly its Taylor expansion of order (1, 1), so the
  effect of an error in the patch coordinates is exact, not first order;
- each input-derived float carries an absolute error bound e_*, propagated
  through the formulas below; second-order terms are kept, and every bound is
  inflated by (1 + 64 U) to cover rounding in computing the bound itself;
- per-raster errors (ue, ve, ze) are measured exactly once (Raster), per-cell
  ones (du = c - ue) exactly with TwoSum, and whether two crossings coincide
  exactly with TwoSum and Dekker's TwoProduct.
Generous constants throughout: a bound only has to be safe, and the gap between
the bound and the true error (orders of magnitude) costs nothing but a few more
exact checks.

Everything below Raster is plain float code: oracle.py runs it as Python and
oracle_numba compiles the same functions, so the two agree bit for bit.
"""

import math
from collections import namedtuple
from fractions import Fraction

U = 2.0 ** -53
INFL = 1 + 64 * U
G3 = 3 * U / (1 - 3 * U)
G4 = 4 * U / (1 - 4 * U)
G12 = 12 * U / (1 - 12 * U)
SPLIT = 134217729.0          # 2**27 + 1, Dekker's splitter

# segment states
CLEAR, BLOCKING, UNDECIDED, NODATA = 0, 1, 2, 3


def _up(f):
    """A float >= the non-negative Fraction f."""
    v = float(f)
    return v if v >= f else math.nextafter(v, math.inf)


class Raster:
    """Per-raster errors, measured exactly (Fractions, once): the float ue, ve, ze
    oracle.viewshed computes against oracle._blocked_exact's exact ones."""

    def __init__(self, transform, x, y, ground, oh, ue, ve, ze):
        F = Fraction
        ue_x = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
        ve_x = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
        ze_x = ground + F(oh)
        self.e_ue = _up(abs(F(ue) - ue_x))
        self.e_ve = _up(abs(F(ve) - ve_x))
        self.e_ze = _up(abs(F(ze) - ze_x))


def raster_slope(z):
    """Largest difference between edge-adjacent heights (no-data left out)."""
    import numpy as np
    s = 0.0
    for a, b in ((z[:, 1:], z[:, :-1]), (z[1:, :], z[:-1, :])):
        d = np.abs(a - b)
        if np.isfinite(d).any():
            s = max(s, float(np.nanmax(d)))
    return s * (1 + 4 * U)


# --- plain float code from here on (compiled by oracle_numba) --------------------

Cell = namedtuple("Cell", "ue ve ze du dv bend dz th e_ue e_ve e_ze e_du e_dv e_bend e_dz delta")


def two_sum_err(a, b, s):
    """The exact rounding error of s = fl(a + b) (Knuth's TwoSum): a + b = s + err."""
    bb = s - a
    return (a - (s - bb)) + (b - bb)


def two_prod(a, b):
    """(p, e) with p = fl(a * b) and a * b = p + e exactly (Dekker; no overflow)."""
    p = a * b
    t = SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def make_cell(ue, ve, ze, e_ue, e_ve, e_ze, th, drop, res, z_rc, r, c):
    """The error state of the sightline to cell (r, c): oracle.viewshed's floats for
    it, each with a bound on its distance from the exact value."""
    du = c - ue
    dv = r - ve
    e_du = INFL * (abs(two_sum_err(c, -ue, du)) + e_ue)
    e_dv = INFL * (abs(two_sum_err(r, -ve, dv)) + e_ve)
    bend = drop * (du * du + dv * dv) * res * res
    e_s = 2 * abs(du) * e_du + e_du * e_du + 2 * abs(dv) * e_dv + e_dv * e_dv
    e_bend = INFL * (6 * U * abs(bend) + abs(drop) * res * res * e_s * (1 + 8 * U))
    dz = z_rc - bend + th - ze
    e_dz = INFL * (3 * U * (abs(z_rc) + abs(bend) + abs(th) + abs(ze)) * (1 + 4 * U)
                   + e_bend + e_ze)
    # a float crossing t = fl(fl(k - p0) / d) against (k - p0*) / d*
    delta = 0.0
    for d, e_p, e_d in ((du, e_ue, e_du), (dv, e_ve, e_dv)):
        if d == 0:
            continue
        if e_d > 1e-3 * abs(d):
            delta = math.inf                     # degenerate: certify nothing
            break
        delta = max(delta, INFL * 1.01 * (3 * U + (e_p + e_d) / abs(d)) * (1 + 2 * U))
    return Cell(ue, ve, ze, du, dv, bend, dz, th, e_ue, e_ve, e_ze, e_du, e_dv, e_bend, e_dz,
                delta)


def crossings_equal(cell, a_axis, a_k, b_axis, b_k, r, c):
    """True only if two crossings are certainly equal in exact arithmetic. Axes:
    0 = t 0, 1 = t 1, 2 = u (k is the column), 3 = v (k is the row). False means
    unequal or not provable, which only costs an exact check."""
    if a_axis > b_axis:
        a_axis, a_k, b_axis, b_k = b_axis, b_k, a_axis, a_k
    if a_axis == b_axis:
        return a_axis < 2                        # 0 = 0, 1 = 1; distinct k never meet
    if a_axis == 0:
        return False                             # an exact crossing is never at 0
    if a_axis == 1:                              # (k - p0*) / (target - p0*) = 1 iff k = target
        return b_k == (c if b_axis == 2 else r)
    # a u crossing and a v crossing: (k - ue*) dv* == (m - ve*) du*, provable when
    # every one of those differences is exact in floats
    if cell.e_ue != 0 or cell.e_ve != 0 or cell.e_du != 0 or cell.e_dv != 0:
        return False
    a = a_k - cell.ue
    b = b_k - cell.ve
    if two_sum_err(a_k, -cell.ue, a) != 0 or two_sum_err(b_k, -cell.ve, b) != 0:
        return False
    p1, e1 = two_prod(a, cell.dv)
    p2, e2 = two_prod(b, cell.du)
    return p1 == p2 and e1 == e2


def near_observer_flag(cell):
    """True when an integer lies within an observer coordinate's error: a crossing
    right at the observer could then be in one list (float or exact) and not the
    other, so the segment starting at t = 0 must not be certified. (At the target
    end the exact coordinate is the integer itself, so a stray crossing shows up as
    a float boundary near 1, which the 16 delta rule catches.)"""
    for p0, d, e in ((cell.ue, cell.du, cell.e_ue), (cell.ve, cell.dv, cell.e_ve)):
        if d != 0 and e > 0:
            k = math.floor(p0 + 0.5)
            if abs(p0 - k) <= e * (1 + 4 * U) + 2 * U * abs(p0):
                return True
    return False


def coord_errors(cell, c0, r0, t):
    """Bounds on the errors of the patch coordinates tc = ue + t du - c0 and tr at t."""
    at = abs(t)
    e_tc = (3 * U * (abs(cell.ue) + abs(t * cell.du) + abs(c0)) * (1 + 4 * U)
            + cell.e_ue + at * cell.e_du)
    e_tr = (3 * U * (abs(cell.ve) + abs(t * cell.dv) + abs(r0)) * (1 + 4 * U)
            + cell.e_ve + at * cell.e_dv)
    return e_tc, e_tr


def gap_and_error(cell, z00, z01, z10, z11, c0, r0, t):
    """(float gap at t, bound on |float gap - exact gap of this patch's polynomial at
    t|, coordinate error bounds). The float gap is oracle._segment_min's gap(t)."""
    ue, ve, du, dv = cell.ue, cell.ve, cell.du, cell.dv
    tc = ue + t * du - c0
    tr = ve + t * dv - r0
    bil = (z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr
    g = cell.ze + t * cell.dz - bil + cell.bend * t * t
    at = abs(t)
    e_tc, e_tr = coord_errors(cell, c0, r0, t)
    b_abs = ((abs(z00) * abs(1 - tc) + abs(z01) * abs(tc)) * abs(1 - tr)
             + (abs(z10) * abs(1 - tc) + abs(z11) * abs(tc)) * abs(tr))
    dbc = abs(z01 - z00) * abs(1 - tr) + abs(z11 - z10) * abs(tr)
    dbr = abs(z10 - z00) * abs(1 - tc) + abs(z11 - z01) * abs(tc)
    e3a = abs(z00 - z01 - z10 + z11)
    pert = (dbc * e_tc + dbr * e_tr + e3a * e_tc * e_tr) * (1 + 8 * U)
    err = INFL * (G12 * (abs(cell.ze) + at * abs(cell.dz) + b_abs + abs(cell.bend) * t * t)
                  + pert + cell.e_ze + at * cell.e_dz + t * t * cell.e_bend)
    return g, err, e_tc, e_tr


def derivatives(cell, z00, z01, z10, z11, c0, r0, ta, e_tc, e_tr):
    """(g1, e_g1, g2, e_g2): oracle._segment_min's slope at ta and curvature, with
    bounds against the exact patch polynomial's (e_tc, e_tr: coord_errors at ta)."""
    ue, ve, du, dv, bend = cell.ue, cell.ve, cell.du, cell.dv, cell.bend
    e3 = z00 - z01 - z10 + z11
    g2 = -e3 * du * dv + bend
    tca = ue + ta * du - c0
    trb = ve + ta * dv - r0
    g1 = cell.dz - ((z01 - z00) * du + (z10 - z00) * dv + e3 * (tca * dv + trb * du)) + 2 * bend * ta
    e3a = abs(e3)
    e_e3 = G3 * (abs(z00) + abs(z01) + abs(z10) + abs(z11))
    e3m = e3a + e_e3
    a01 = abs(z01 - z00)
    a10 = abs(z10 - z00)
    e_g1 = INFL * (G12 * (abs(cell.dz) + a01 * abs(du) + a10 * abs(dv)
                          + e3a * (abs(tca * dv) + abs(trb * du)) + 2 * abs(bend) * abs(ta))
                   + cell.e_dz + a01 * cell.e_du + a10 * cell.e_dv
                   + e_e3 * (abs(tca * dv) + abs(trb * du))
                   + e3m * (e_tc * abs(dv) + abs(tca) * cell.e_dv + e_tc * cell.e_dv
                            + e_tr * abs(du) + abs(trb) * cell.e_du + e_tr * cell.e_du)
                   + 2 * abs(ta) * cell.e_bend) * (1 + 8 * U)
    e_g2 = INFL * (G4 * (e3a * abs(du * dv) + abs(bend)) + e_e3 * abs(du * dv)
                   + e3m * (abs(du) * cell.e_dv + abs(dv) * cell.e_du + cell.e_du * cell.e_dv)
                   + cell.e_bend) * (1 + 8 * U)
    return g1, e_g1, g2, e_g2


def certainly_blocking(cell, z00, z01, z10, z11, c0, r0, ta, tb):
    """True if the exact gap is < 0 somewhere in this patch's exact segment, whose
    ends are within delta of ta and tb: checked at points certainly inside it."""
    if not (tb - ta > 8 * cell.delta):
        return False
    e_tc, e_tr = coord_errors(cell, c0, r0, ta)
    g1, _, g2, _ = derivatives(cell, z00, z01, z10, z11, c0, r0, ta, e_tc, e_tr)
    return _blocking_at(cell, z00, z01, z10, z11, c0, r0, ta, tb, g1, g2)


def _blocking_at(cell, z00, z01, z10, z11, c0, r0, ta, tb, g1, g2):
    """certainly_blocking with the float slope at ta and curvature already known."""
    delta = cell.delta
    if not (tb - ta > 8 * delta):
        return False
    lo_p = ta + 3 * delta
    hi_p = tb - 3 * delta
    n = 2
    v = lo_p
    if g2 > 0:
        v = min(max(ta + (-g1 / (2 * g2)), lo_p), hi_p)
        n = 3
    for i in range(n):
        p = lo_p if i == 0 else (hi_p if i == 1 else v)
        g, err, _, _ = gap_and_error(cell, z00, z01, z10, z11, c0, r0, p)
        if g + err * (1 + 16 * U) + 16 * U * abs(g) < 0:
            return True
    return False


def segment(cell, z, ta, tb, last):
    """Certify one segment of oracle.py's float list (patch from its midpoint, as
    there). Returns (state, r0, c0): BLOCKING (exact minimum < 0), CLEAR (certainly
    not blocking), UNDECIDED (check exactly), NODATA (skipped, as oracle.py skips
    it). Slivers and near-coincident boundaries are the caller's to flag."""
    rows, cols = z.shape
    du, dv, ue, ve = cell.du, cell.dv, cell.ue, cell.ve
    mid = (ta + tb) / 2
    c0 = math.floor(ue + mid * du)
    r0 = math.floor(ve + mid * dv)
    delta = cell.delta
    if not math.isfinite(delta):
        return UNDECIDED, r0, c0
    # the patch must be certain: the exact segment's midpoint (within delta of this
    # one in t) well inside it on both axes
    e_tc, e_tr = coord_errors(cell, c0, r0, mid)
    fu = ue + mid * du - c0
    fv = ve + mid * dv - r0
    for i in range(2):
        f, d, e_d, p0, e_p0, e_t = ((fu, du, cell.e_du, ue, cell.e_ue, e_tc) if i == 0 else
                                    (fv, dv, cell.e_dv, ve, cell.e_ve, e_tr))
        if d == 0 and e_d == 0:
            # the exact sightline runs along this axis at p0*, within e_p0 of p0: its
            # patch is floor(p0) unless an integer lies within that error (with no
            # error, even a sightline exactly on a grid line has a certain patch)
            if e_p0 > 0 and abs(p0 - math.floor(p0 + 0.5)) <= e_p0 * (1 + 4 * U) + 2 * U * abs(p0):
                return UNDECIDED, r0, c0
        elif min(f, 1 - f) <= 2 * (e_t + abs(d) * delta):
            return UNDECIDED, r0, c0
    ra = min(max(r0, 0), rows - 1)
    rb = min(max(r0 + 1, 0), rows - 1)
    ca = min(max(c0, 0), cols - 1)
    cb = min(max(c0 + 1, 0), cols - 1)
    z00 = z[ra, ca]
    z01 = z[ra, cb]
    z10 = z[rb, ca]
    z11 = z[rb, cb]
    if math.isnan(z00) or math.isnan(z01) or math.isnan(z10) or math.isnan(z11):
        return NODATA, r0, c0
    # Clear is tested first: it is the common case, and since both certificates are
    # rigorous a segment can't pass both, so the order never changes the answer.
    g0, e0, e_tc, e_tr = gap_and_error(cell, z00, z01, z10, z11, c0, r0, ta)
    g1, e_g1, g2, e_g2 = derivatives(cell, z00, z01, z10, z11, c0, r0, ta, e_tc, e_tr)
    g1_lo = g1 - e_g1
    g1_hi = g1 + e_g1
    g2_lo = g2 - e_g2
    g2_hi = g2 + e_g2
    g1m = max(abs(g1_lo), abs(g1_hi))
    g2m = max(abs(g2_lo), abs(g2_hi))
    if last and cell.th == 0:
        # exact gap(1) = 0, so gap = (1 - t) K(t) with K linear: not blocking iff
        # K(ta) >= 0 and K(1) = -gap'(1) >= 0, with gap'(1) = g1(ta) + 2 g2 (1 - ta)
        shift = (g1m + 2 * g2m * delta) * delta
        k_ta = g0 - e0 - shift
        if g2_hi > 0:
            slope1_hi = g1_hi + 2 * g2_hi * ((1 - ta) * (1 + U) + delta)
        else:
            slope1_hi = g1_hi + 2 * g2_hi * max(1 - ta - delta, 0.0) * (1 - U)
        slop = 16 * U * (abs(g0) + e0 + shift)
        if k_ta - slop > 0 and slope1_hi * (1 + 16 * U) + 16 * U * abs(g1) < 0:
            return CLEAR, r0, c0
        if _blocking_at(cell, z00, z01, z10, z11, c0, r0, ta, tb, g1, g2):
            return BLOCKING, r0, c0
        return UNDECIDED, r0, c0
    # minimum of the exact polynomial over the exact segment, inside
    # [ta - delta, tb + delta]: gap(ta + s) = gap(ta) + g1 s + g2 s^2
    lh = ((tb - ta) + delta) * (1 + 2 * U)
    q = min(0.0, g1_lo * lh + g2_lo * lh * lh)
    if g2_lo > 0:
        s0 = -g1_lo / (2 * g2_lo)
        if 0 < s0 < lh:
            q = min(q, -(g1_lo * g1_lo) / (4 * g2_lo) * (1 + 4 * U))
    q = min(q, -g1m * delta + min(0.0, g2_lo) * delta * delta)
    slop = 16 * U * (abs(g0) + e0 + abs(q) + g1m * lh + g2m * lh * lh)
    if g0 - e0 + q - slop > 0:
        return CLEAR, r0, c0
    if _blocking_at(cell, z00, z01, z10, z11, c0, r0, ta, tb, g1, g2):
        return BLOCKING, r0, c0
    return UNDECIDED, r0, c0


def boundaries(cell, r, c):
    """oracle.py's sorted boundary list for this sightline, and the set of its values
    whose segments must not be certified because the float and exact lists might not
    pair up one to one there: two boundaries within 16 delta of each other, unless
    they are equal in floats and certainly equal exactly (a sightline through a
    lattice point: the same de-duplication on both sides); t = 0 when
    near_observer_flag; everything when delta is unbounded. Python only (oracle.py);
    oracle_numba has the same rule in compiled form."""
    ue, ve, du, dv = cell.ue, cell.ve, cell.du, cell.dv
    lab = [(0.0, 0, 0), (1.0, 1, 0)]
    for p0, d, axis in ((ue, du, 2), (ve, dv, 3)):
        if d == 0:
            continue
        p1 = p0 + d
        lo, hi = min(p0, p1), max(p0, p1)
        for k in range(math.floor(lo) + 1, math.ceil(hi)):
            lab.append(((k - p0) / d, axis, k))
    lab.sort(key=lambda e: e[0])
    flagged = set()
    lim = 16 * cell.delta
    for a, b in zip(lab, lab[1:]):
        if b[0] - a[0] <= lim and not (a[0] == b[0]
                                        and crossings_equal(cell, a[1], a[2], b[1], b[2], r, c)):
            flagged.update((a[0], b[0]))
    if near_observer_flag(cell):
        flagged.add(0.0)
    if not math.isfinite(cell.delta):
        flagged.update(t for t, _, _ in lab)
    return sorted({t for t, _, _ in lab}), flagged
