"""Exact horizon oracle on the bilinear surface.

For a float direction (dx, dy) and cell T = (r, c) (height z_T + th), the horizon rise
    Hs(T) = sup over s > 0 on positive-length stretches of data patches of
            (B(X(s)) - z_T - th - k res^2 L^2 s^2) / s,       X(s) = T + s (dx, dy)
in map units per unit ray parameter (tan of the horizon elevation = Hs / (L res)). Exactly:
T is shadowed by a sun (dx, dy, dz) iff Hs(T) > dz (shadow_oracle's semantics).
Per patch the quotient is A s + Bq + C / s (A = e3 dx dy - k res^2 L^2, Bq = slope terms, C = B(T) -
z_T - th), so Hs is a rational or p + r sqrt(m) with rationals: exact via Alg below.
Returns: an exact Alg (or None for -inf: nothing ahead) and its correctly rounded float.
Optional distance cut (max_dist R, map units): only ray points at horizontal distance s L res <= R count,
i.e. s <= s* = R / (res L), L = sqrt(dx^2 + dy^2). s* is irrational in general, but f(s*) = B +
(A R / (res L^2) + C res / R) sqrt(L^2) is again p + r sqrt(m), and s <= s* iff s^2 L^2 res^2 <= R^2.
"""

import math
import struct
from decimal import Decimal, getcontext
from fractions import Fraction as Q

import numpy as np

from .oracle import drop_factor
from .shadow_oracle import _candidates, _corners, _interval

getcontext().prec = 80


def _sgn(x):
    return (x > 0) - (x < 0)


def _sign_a_b_sqrt(a, b, u):
    """sign(a + b sqrt(u)), rationals, u >= 0."""
    if b == 0 or u == 0:
        return _sgn(a)
    sb = _sgn(b)
    if _sgn(a) == sb or a == 0:
        return sb
    # opposite signs: compare a^2 with b^2 u
    d = a * a - b * b * u
    return _sgn(a) * _sgn(d) if d != 0 else 0


class Alg:
    """p + r sqrt(m), rationals, m >= 0."""
    __slots__ = ("p", "r", "m")

    def __init__(self, p, r=Q(0), m=Q(0)):
        self.p, self.r, self.m = Q(p), Q(r), Q(m)
        if self.r == 0 or self.m == 0:
            self.r, self.m = Q(0), Q(0)

    def sign_minus(self, o):
        """sign(self - o) exactly."""
        a = self.p - o.p
        b1, u = self.r, self.m
        b2, v = -o.r, o.m
        if b2 == 0 or v == 0:
            return _sign_a_b_sqrt(a, b1, u)
        if b1 == 0 or u == 0:
            return _sign_a_b_sqrt(a, b2, v)
        if u == v:
            return _sign_a_b_sqrt(a, b1 + b2, u)
        # sign(X - Y), X = a + b1 sqrt u, Y = -b2 sqrt v
        sx = _sign_a_b_sqrt(a, b1, u)
        sy = _sgn(-b2)
        if sx != sy:
            return _sgn(sx - sy)
        if sx == 0:
            return 0
        # same sign s: X - Y has sign s * sign(X^2 - Y^2)
        d = _sign_a_b_sqrt(a * a + b1 * b1 * u - b2 * b2 * v, 2 * a * b1, u)
        return sx * d

    def gt(self, o):
        return self.sign_minus(o) > 0

    def approx(self):
        return float(Decimal(self.p.numerator) / Decimal(self.p.denominator)
                     + Decimal(self.r.numerator) / Decimal(self.r.denominator)
                     * (Decimal(self.m.numerator) / Decimal(self.m.denominator)).sqrt())

    def rounded(self):
        """The float nearest to the exact value (ties to even)."""
        f = self.approx()
        for _ in range(8):
            lo = Q(f) + (Q(math.nextafter(f, -math.inf)) - Q(f)) / 2
            hi = Q(f) + (Q(math.nextafter(f, math.inf)) - Q(f)) / 2
            s_lo, s_hi = self.sign_minus(Alg(lo)), self.sign_minus(Alg(hi))
            if s_lo > 0 and s_hi < 0:
                return f
            if s_lo == 0 or s_hi == 0:          # exactly halfway: ties to even
                g = math.nextafter(f, -math.inf) if s_lo == 0 else math.nextafter(f, math.inf)
                even = lambda x: struct.unpack("<q", struct.pack("<d", x))[0] % 2 == 0
                return f if even(f) else g
            f = math.nextafter(f, -math.inf) if s_lo <= 0 else math.nextafter(f, math.inf)
        raise AssertionError("rounding did not converge")


def _patch_sup(A, B, C, lo, hi):
    """Exact sup of A s + B + C / s over s in [lo, hi] (lo > 0), or over (0, hi] if lo == 0 (then C <= 0)."""
    cands = [Alg(A * hi + B + C / hi)]
    if lo > 0:
        cands.append(Alg(A * lo + B + C / lo))
    elif C == 0:
        cands.append(Alg(B))                       # the limit at s -> 0
    if C < 0 and A < 0:                            # concave: interior max at s* = sqrt(C / A)
        m = C / A
        if (lo == 0 or lo * lo < m) and m < hi * hi:
            cands.append(Alg(B, -2, A * C))        # A s* + C / s* = -2 sqrt(A C)
    best = cands[0]
    for x in cands[1:]:
        if x.gt(best):
            best = x
    return best


def _patch_sup_cut(A, B, C, lo, L2, res, R):
    """Exact sup of A s + B + C / s over s in [lo, s*] (or (0, s*] if lo == 0), s* = R / (res sqrt(L2))."""
    s2 = R * R / (res * res * L2)                    # s*^2, rational
    cands = [Alg(B, A * R / (res * L2) + C * res / R, L2)]
    if lo > 0:
        cands.append(Alg(A * lo + B + C / lo))
    elif C == 0:
        cands.append(Alg(B))
    if C < 0 and A < 0:
        m = C / A
        if (lo == 0 or lo * lo < m) and m < s2:
            cands.append(Alg(B, -2, A * C))
    best = cands[0]
    for x in cands[1:]:
        if x.gt(best):
            best = x
    return best


def horizon_exact(z, r, c, dx, dy, drop, res, th=0.0, max_dist=None):
    """Exact Hs(T) as an Alg, or None if no data patch lies ahead (within max_dist, if given)."""
    rows, cols = z.shape
    zT = Q(float(z[r, c])) + Q(th)
    dxq, dyq = Q(dx), Q(dy)
    L2 = dxq * dxq + dyq * dyq
    curv = Q(drop) * Q(res) * Q(res) * L2
    resq = Q(res)
    R = None if max_dist is None else Q(max_dist)
    best = None
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
            continue
        cut = False
        if R is not None:
            # beyond s*: lo >= s* (lo > 0) -> nothing of positive length within the cut
            if lo > 0 and lo * lo * L2 * resq * resq >= R * R:
                continue
            cut = hi * hi * L2 * resq * resq > R * R
        z00, z01, z10, z11 = (Q(float(v)) for v in zc)
        e3 = z00 - z01 - z10 + z11
        u0, v0 = Q(c) - c0, Q(r) - r0
        A = e3 * dxq * dyq - curv
        B = (z01 - z00) * dxq + (z10 - z00) * dyq + e3 * (u0 * dyq + v0 * dxq)
        C = z00 + (z01 - z00) * u0 + (z10 - z00) * v0 + e3 * u0 * v0 - zT
        v = _patch_sup_cut(A, B, C, lo, L2, resq, R) if cut else _patch_sup(A, B, C, lo, hi)
        if best is None or v.gt(best):
            best = v
    return best


def horizon_map(z, transform, vec, target_height=0.0, curvature=False, refraction=0.13, earth_radius=6371008.8, max_dist=None):
    """(exact Alg-or-None array, correctly rounded float array; -inf if nothing ahead, NaN no-data)."""
    z = np.ascontiguousarray(z, dtype=float)
    dx, dy = float(vec[0]), float(vec[1])
    drop = drop_factor(curvature, refraction, earth_radius)
    res = abs(transform[1])
    ex = np.empty(z.shape, dtype=object)
    fl = np.full(z.shape, np.nan)
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            if z[r, c] != z[r, c]:
                continue
            h = horizon_exact(z, r, c, dx, dy, drop, res, target_height, max_dist)
            ex[r, c] = h
            fl[r, c] = -math.inf if h is None else h.rounded()
    return ex, fl
