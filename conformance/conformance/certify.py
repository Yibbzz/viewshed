"""Rigorous error bounds for the bilinear oracle's float pass.

oracle.py decides a cell in floats when its minimum gap is beyond +-tol, with tol
= 1e-9 of scale: an engineering margin, not a proof. This module bounds, for one
float segment, how far the float numbers can be from the exact ones, and so
certifies a segment as certainly not blocking (its exact minimum is >= 0 where it
matters) or certainly blocking (< 0). Anything it can't certify goes to exact
arithmetic, which makes the oracle exact by construction.

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
  inflated by (1 + 64 U) to cover rounding in computing the bound itself.
Generous constants throughout: a bound only has to be safe, and the gap between
the bound and the true error (orders of magnitude) costs nothing but a few more
exact checks.
"""

import math
from fractions import Fraction

U = 2.0 ** -53
INFL = 1 + 64 * U


def gamma(n):
    return n * U / (1 - n * U)


G4, G6, G12 = gamma(4), gamma(6), gamma(12)


def two_sum_err(a, b, s):
    """The exact rounding error of s = fl(a + b), as a float (Knuth's TwoSum: exact
    without overflow): a + b = s + err exactly."""
    bb = s - a
    return (a - (s - bb)) + (b - bb)


def _up(f):
    """A float >= the non-negative Fraction f."""
    v = float(f)
    return v if v >= f else math.nextafter(v, math.inf)


class Raster:
    """Per-raster errors, measured exactly (Fractions, once): the float ue, ve, ze
    oracle.viewshed computes against oracle._blocked_exact's exact ones."""

    __slots__ = ("ue", "ve", "ze", "e_ue", "e_ve", "e_ze", "ue_x", "ve_x")

    def __init__(self, transform, x, y, ground, oh, ue, ve, ze):
        F = Fraction
        ue_x = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
        ve_x = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
        ze_x = ground + F(oh)
        self.ue, self.ve, self.ze = ue, ve, ze
        self.ue_x, self.ve_x = ue_x, ve_x
        self.e_ue = _up(abs(F(ue) - ue_x))
        self.e_ve = _up(abs(F(ve) - ve_x))
        self.e_ze = _up(abs(F(ze) - ze_x))


class Cell:
    """Error state for one sightline (observer to the centre of cell (r, c)).
    Takes the same float values oracle.viewshed computes, plus the inputs."""

    __slots__ = ("ue", "ve", "ze", "du", "dv", "bend", "dz", "th", "e_ue", "e_ve", "e_ze",
                 "e_du", "e_dv", "e_bend", "e_dz", "delta", "gp", "S", "raster")

    def __init__(self, raster, th, drop, res, z_rc, r, c, S):
        ue, ve, ze = raster.ue, raster.ve, raster.ze
        self.raster = raster
        self.ue, self.ve, self.ze, self.th, self.S = ue, ve, ze, th, S
        self.e_ue, self.e_ve, self.e_ze = raster.e_ue, raster.e_ve, raster.e_ze
        # du = fl(c - ue): its rounding error exactly (TwoSum), plus ue's own error
        du, dv = c - ue, r - ve
        self.du, self.dv = du, dv
        self.e_du = INFL * (abs(two_sum_err(c, -ue, du)) + self.e_ue)
        self.e_dv = INFL * (abs(two_sum_err(r, -ve, dv)) + self.e_ve)
        bend = drop * (du * du + dv * dv) * res * res
        self.bend = bend
        e_s = (2 * abs(du) * self.e_du + self.e_du ** 2 + 2 * abs(dv) * self.e_dv + self.e_dv ** 2)
        self.e_bend = INFL * (6 * U * abs(bend) + abs(drop) * res * res * e_s * (1 + 8 * U))
        dz = z_rc - bend + th - ze
        self.dz = dz
        self.e_dz = INFL * (3 * U * (abs(z_rc) + abs(bend) + abs(th) + abs(ze)) * (1 + 4 * U)
                            + self.e_bend + self.e_ze)
        # boundary error: a crossing t = fl(fl(k - p0) / d) against (k - p0*) / d*
        delta = 0.0
        for d, e_p, e_d in ((du, self.e_ue, self.e_du), (dv, self.e_ve, self.e_dv)):
            if d == 0:
                continue
            if e_d > 1e-3 * abs(d):
                delta = math.inf                 # degenerate: certify nothing
                break
            delta = max(delta, INFL * 1.01 * (3 * U + (e_p + e_d) / abs(d)) * (1 + 2 * U))
        self.delta = delta
        # |G'(t)| on any patch along this line, for t within delta of [0, 1]
        self.gp = INFL * ((abs(dz) + self.e_dz) + S * (abs(du) + self.e_du + abs(dv) + self.e_dv)
                          + 2 * (abs(bend) + self.e_bend) * (1 + delta)) * (1 + 8 * U)


def boundaries(cell, r, c):
    """oracle.py's sorted boundary list for this sightline, and the set of its values
    whose segments must not be certified: where the float and exact lists might not
    pair up one to one. That is (a) two boundaries within 16 delta of each other,
    unless they are exactly equal both in floats and in exact arithmetic (a line
    through a lattice point: the same de-duplication on both sides), checked here
    exactly; (b) t = 0 when an integer lies within the observer coordinate's error
    (a crossing right at the observer could be in one list and not the other). At
    the target end the exact coordinate is the integer column or row itself, so a
    stray crossing there shows up as a float boundary near 1 and is caught by (a)."""
    ue, ve, du, dv = cell.ue, cell.ve, cell.du, cell.dv
    rx = cell.raster
    lab = [(0.0, "0", 0), (1.0, "1", 0)]
    for p0, d, axis in ((ue, du, "u"), (ve, dv, "v")):
        if d == 0:
            continue
        p1 = p0 + d
        lo, hi = min(p0, p1), max(p0, p1)
        for k in range(math.floor(lo) + 1, math.ceil(hi)):
            lab.append(((k - p0) / d, axis, k))
    lab.sort(key=lambda e: e[0])
    flagged = set()
    exact = {}

    def ex(e):
        if e not in exact:
            t, axis, k = e
            if axis == "0":
                exact[e] = Fraction(0)
            elif axis == "1":
                exact[e] = Fraction(1)
            elif axis == "u":
                exact[e] = (k - rx.ue_x) / (c - rx.ue_x)
            else:
                exact[e] = (k - rx.ve_x) / (r - rx.ve_x)
        return exact[e]

    lim = 16 * cell.delta
    for a, b in zip(lab, lab[1:]):
        if b[0] - a[0] <= lim:
            if not (a[0] == b[0] and ex(a) == ex(b)):
                flagged.update((a[0], b[0]))
    for p0, d, e in ((ue, du, cell.e_ue), (ve, dv, cell.e_ve)):
        if d != 0 and e > 0:
            k = round(p0)
            if abs(p0 - k) <= e * (1 + 4 * U) + 2 * U * abs(p0):
                flagged.add(0.0)
    if not math.isfinite(cell.delta):
        flagged.update(t for t, _, _ in lab)
    ts = sorted({t for t, _, _ in lab})
    return ts, flagged


def _gap_and_error(cell, z00, z01, z10, z11, c0, r0, t):
    """(float gap at t, bound on |float gap - exact gap of this patch's polynomial at t|).
    The float gap is oracle._segment_min's gap(t), operation for operation."""
    ue, ve, du, dv = cell.ue, cell.ve, cell.du, cell.dv
    tc = ue + t * du - c0
    tr = ve + t * dv - r0
    bil = (z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr
    g = cell.ze + t * cell.dz - bil + cell.bend * t * t
    at = abs(t)
    e_tc = 3 * U * (abs(ue) + abs(t * du) + abs(c0)) * (1 + 4 * U) + cell.e_ue + at * cell.e_du
    e_tr = 3 * U * (abs(ve) + abs(t * dv) + abs(r0)) * (1 + 4 * U) + cell.e_ve + at * cell.e_dv
    b_abs = ((abs(z00) * abs(1 - tc) + abs(z01) * abs(tc)) * abs(1 - tr)
             + (abs(z10) * abs(1 - tc) + abs(z11) * abs(tc)) * abs(tr))
    dbc = abs(z01 - z00) * abs(1 - tr) + abs(z11 - z10) * abs(tr)
    dbr = abs(z10 - z00) * abs(1 - tc) + abs(z11 - z01) * abs(tc)
    e3a = abs(z00 - z01 - z10 + z11)
    pert = (dbc * e_tc + dbr * e_tr + e3a * e_tc * e_tr) * (1 + 8 * U)
    err = INFL * (G12 * (abs(cell.ze) + at * abs(cell.dz) + b_abs + abs(cell.bend) * t * t)
                  + pert + cell.e_ze + at * cell.e_dz + t * t * cell.e_bend)
    return g, err, e_tc, e_tr


def _derivatives(cell, z00, z01, z10, z11, c0, r0, ta):
    """(g1, e_g1, g2, e_g2): oracle._segment_min's slope at ta and curvature, with
    bounds against the exact patch polynomial's."""
    ue, ve, du, dv, bend = cell.ue, cell.ve, cell.du, cell.dv, cell.bend
    e3 = z00 - z01 - z10 + z11
    g2 = -e3 * du * dv + bend
    tca, trb = ue + ta * du - c0, ve + ta * dv - r0
    g1 = cell.dz - ((z01 - z00) * du + (z10 - z00) * dv + e3 * (tca * dv + trb * du)) + 2 * bend * ta
    e3a = abs(e3)
    e_e3 = gamma(3) * (abs(z00) + abs(z01) + abs(z10) + abs(z11))
    e3m = e3a + e_e3
    _, _, e_tc, e_tr = _gap_and_error(cell, z00, z01, z10, z11, c0, r0, ta)
    a01, a10 = abs(z01 - z00), abs(z10 - z00)
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


def segment(cell, z, ta, tb, last):
    """Certify one float segment of oracle.py's list (patch from its midpoint, as
    there). Returns (state, r0, c0): state 1 = certainly blocking (exact minimum < 0),
    0 = certainly not blocking, 2 = undecided (check exactly), 3 = no data (skipped,
    as oracle.py skips it). The caller flags slivers (boundaries within 16 delta) as
    undecided before calling this."""
    rows, cols = z.shape
    du, dv, ue, ve = cell.du, cell.dv, cell.ue, cell.ve
    mid = (ta + tb) / 2
    c0, r0 = math.floor(ue + mid * du), math.floor(ve + mid * dv)
    ra, rb = min(max(r0, 0), rows - 1), min(max(r0 + 1, 0), rows - 1)
    ca, cb = min(max(c0, 0), cols - 1), min(max(c0 + 1, 0), cols - 1)
    z00, z01, z10, z11 = z[ra, ca], z[ra, cb], z[rb, ca], z[rb, cb]
    delta = cell.delta
    if not math.isfinite(delta):
        return 2, r0, c0
    # the patch must be certain: the exact segment's midpoint (within delta of this
    # one in t) well inside it on both axes
    _, _, e_tc, e_tr = _gap_and_error(cell, 0.0, 0.0, 0.0, 0.0, c0, r0, mid)
    fu, fv = ue + mid * du - c0, ve + mid * dv - r0
    for f, d, e_d, p0, e_p0, e_t in ((fu, du, cell.e_du, ue, cell.e_ue, e_tc),
                                    (fv, dv, cell.e_dv, ve, cell.e_ve, e_tr)):
        if d == 0 and e_d == 0:
            # the exact sightline runs along this axis at p0*, within e_p0 of p0: its
            # patch is floor(p0) unless an integer lies within that error (with no
            # error, even a sightline exactly on a grid line has a certain patch)
            if e_p0 > 0 and abs(p0 - round(p0)) <= e_p0 * (1 + 4 * U) + 2 * U * abs(p0):
                return 2, r0, c0
        elif min(f, 1 - f) <= 2 * (e_t + abs(d) * delta):
            return 2, r0, c0
    if math.isnan(z00) or math.isnan(z01) or math.isnan(z10) or math.isnan(z11):
        return 3, r0, c0
    corners = (z00, z01, z10, z11, c0, r0)

    # certainly blocking: the gap at a point certainly inside the exact segment is < 0
    if tb - ta > 8 * delta:
        lo_p, hi_p = ta + 3 * delta, tb - 3 * delta
        g1, e_g1, g2, e_g2 = _derivatives(cell, *corners, ta)
        pts = [lo_p, hi_p]
        if g2 > 0:
            v = ta + (-g1 / (2 * g2))
            pts.append(min(max(v, lo_p), hi_p))
        for p in pts:
            g, err, _, _ = _gap_and_error(cell, *corners, p)
            if g + err * (1 + 16 * U) + 16 * U * abs(g) < 0:
                return 1, r0, c0

    # certainly not blocking
    g0, e0, _, _ = _gap_and_error(cell, *corners, ta)
    g1, e_g1, g2, e_g2 = _derivatives(cell, *corners, ta)
    g1_lo, g1_hi, g2_lo, g2_hi = g1 - e_g1, g1 + e_g1, g2 - e_g2, g2 + e_g2
    g1m = max(abs(g1_lo), abs(g1_hi))
    if last and cell.th == 0:
        # exact gap(1) = 0, so gap = (1 - t) K(t) with K linear: not blocking iff
        # K(ta) >= 0 and K(1) = -gap'(1) >= 0. gap'(1) = g1(ta) + 2 g2 (1 - ta).
        shift = (g1m + 2 * max(abs(g2_lo), abs(g2_hi)) * delta) * delta
        k_ta = g0 - e0 - shift
        one = (1 - ta) * (1 + U) + delta
        slope1_hi = g1_hi + 2 * (g2_hi * one if g2_hi > 0 else g2_hi * max(1 - ta - delta, 0) * (1 - U))
        slop = 16 * U * (abs(g0) + e0 + shift)
        if k_ta - slop > 0 and slope1_hi * (1 + 16 * U) + 16 * U * abs(g1) < 0:
            return 0, r0, c0
        return 2, r0, c0
    # minimum of the exact polynomial over [ta* , tb*] within [ta - delta, tb + delta]:
    # gap(ta + s) = gap(ta) + g1 s + g2 s^2
    lh = ((tb - ta) + delta) * (1 + 2 * U)
    q = min(0.0, g1_lo * lh + g2_lo * lh * lh)
    if g2_lo > 0:
        s0 = -g1_lo / (2 * g2_lo)
        if 0 < s0 < lh:
            q = min(q, -(g1_lo * g1_lo) / (4 * g2_lo) * (1 + 4 * U))
    q = min(q, -g1m * delta + min(0.0, g2_lo) * delta * delta)
    slop = 16 * U * (abs(g0) + e0 + abs(q) + g1m * lh + max(abs(g2_lo), abs(g2_hi)) * lh * lh)
    if g0 - e0 + q - slop > 0:
        return 0, r0, c0
    return 2, r0, c0


def raster_slope(z):
    """Largest difference between edge-adjacent heights (no-data left out), inflated:
    bounds the bilinear surface's partial derivatives in every patch."""
    import numpy as np
    s = 0.0
    for a, b in ((z[:, 1:], z[:, :-1]), (z[1:, :], z[:-1, :])):
        d = np.abs(a - b)
        if np.isfinite(d).any():
            s = max(s, float(np.nanmax(d)))
    return s * (1 + 4 * U)
