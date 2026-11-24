"""Exact Fresnel clearance of links on the bilinear surface (the reference for `fresnel_clearance`).

A link joins two points a, b = (x, y, height above the surface), heights > 0. With D the
horizontal length (map units, metres), t in [0, 1] the position along it, and the same
sightline, pieces and no-data rule as `oracle.line_of_sight`:

    gap(t) = sightline(t) - ground(t) - drop * D^2 * t (1 - t)     (curvature: chord drop)
    r1(t)  = sqrt(lambda * D * t (1 - t)),  lambda = c / frequency  (first Fresnel zone radius)
    rho(t) = gap(t) / r1(t)

The link's worst clearance is the infimum of rho over t in (0, 1) on pieces whose patch has
data: 1 = first zone clear, 0.6 = the usual planning target, 0 = grazing, < 0 = obstructed.

Exact structure: on each piece gap is a quadratic q(t) with rational coefficients, and
rho = q / sqrt(w) / sqrt(lambda D) with w = t (1 - t). Extrema of q / sqrt(w) are where
N = q' w - q w' / 2 (a cubic over Q) changes sign. Its real roots in the piece are isolated
with a Sturm sequence and bisected in rationals to width 2^-120; q / sqrt(w) is evaluated at
the piece's ends and at both ends of every root bracket with a 60-digit square root. The
result is within ~1e-30 of the exact infimum, far below the crate's tolerance.
"""

import math
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

from .oracle import _bilinear, _corners, _crossings, _ground_exact, drop_factor

SPEED_OF_LIGHT = 299_792_458.0
getcontext().prec = 60


# --- polynomials over Q: coefficient lists, lowest degree first --------------------------

def _trim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def _mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1) if p and q else []
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return _trim(out)


def _sub(p, q):
    n = max(len(p), len(q))
    return _trim([(p[i] if i < len(p) else 0) - (q[i] if i < len(q) else 0) for i in range(n)])


def _deriv(p):
    return _trim([i * p[i] for i in range(1, len(p))])


def _ev(p, t):
    v = Fraction(0)
    for a in reversed(p):
        v = v * t + a
    return v


def _rem(p, q):
    p = list(p)
    while len(p) >= len(q) and p:
        f = p[-1] / q[-1]
        k = len(p) - len(q)
        for i, b in enumerate(q):
            p[k + i] -= f * b
        p = _trim(p)
    return p


def _sturm(p):
    seq = [p, _deriv(p)]
    while seq[-1]:
        r = _rem(seq[-2], seq[-1])
        if not r:
            break
        seq.append([-a for a in r])
    return seq


def _changes(seq, t):
    signs = [s for s in (_ev(p, t) for p in seq) if s != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if (a > 0) != (b > 0))


def _roots(p, lo, hi, width=Fraction(1, 2 ** 120)):
    """Brackets [u, v] (lo <= u <= v <= hi), each holding one distinct root of p in (lo, hi]."""
    if len(p) < 2:
        return []
    seq = _sturm(p)
    out = []
    stack = [(lo, hi)]
    while stack:
        u, v = stack.pop()
        n = _changes(seq, u) - _changes(seq, v)       # distinct roots in (u, v]
        if n == 0:
            continue
        if n == 1 and v - u <= width:
            out.append((u, v))
            continue
        m = (u + v) / 2
        stack += [(u, m), (m, v)]
    return out


def _dsqrt(x):
    return Decimal(x.numerator).__truediv__(Decimal(x.denominator)).sqrt()


# --- the link -----------------------------------------------------------------------------

def fresnel_clearance(z, transform, a, b, frequency, curvature=False, refraction=0.13,
                      earth_radius=6371008.8):
    """(worst rho as a 60-digit Decimal, the t where it is attained (a Fraction, approximate
    to 2^-120 at an interior optimum), worst rho as a float); rho = +inf (Decimal('Infinity'))
    when no piece has data. Raises ValueError as `oracle.line_of_sight`, and for heights <= 0,
    frequency <= 0 or a zero-length link."""
    F = Fraction
    z = np.asarray(z, dtype=float)
    rows, cols = z.shape
    if not (frequency > 0 and math.isfinite(frequency)):
        raise ValueError("frequency must be > 0")
    ends = []
    for name, p in (("a", a), ("b", b)):
        x, y, h = (float(v) for v in p)
        u = (F(x) - F(transform[0])) / F(transform[1]) - F(1, 2)
        v = (F(y) - F(transform[3])) / F(transform[5]) - F(1, 2)
        if not (-F(1, 2) <= u <= cols - F(1, 2) and -F(1, 2) <= v <= rows - F(1, 2)):
            raise ValueError(f"point {name} is outside the raster")
        if not h > 0:
            raise ValueError("heights must be > 0")
        ground = _ground_exact(z, transform, x, y)
        if ground is None:
            raise ValueError(f"there is no height data under point {name}")
        ends.append((u, v, ground + F(h)))
    (ue, ve, ze), (ub, vb, zb) = ends
    du, dv = ub - ue, vb - ve
    res = abs(F(transform[1]))
    d2 = (du * du + dv * dv) * res * res              # D^2, exact
    if d2 == 0:
        raise ValueError("the two points coincide")
    bend = F(drop_factor(curvature, refraction, earth_radius)) * d2
    lam = F(SPEED_OF_LIGHT) / F(frequency)
    # 1 / sqrt(lambda D) = 1 / (lambda^(1/2) d2^(1/4))
    scale = 1 / (_dsqrt(lam) * _dsqrt(d2).sqrt())
    w = [F(0), F(1), F(-1)]                           # t (1 - t)
    best, best_t = None, None
    ts = sorted({F(0), F(1), *_crossings(ue, du), *_crossings(ve, dv)})
    for ta, tb in zip(ts, ts[1:]):
        mid = (ta + tb) / 2
        c0, r0 = math.floor(ue + mid * du), math.floor(ve + mid * dv)
        cs = _corners(z, r0, c0)
        if any(math.isnan(q) for q in cs):
            continue
        cs = tuple(F(q) for q in cs)

        def gap(t):
            return ze + t * (zb - ze) - _bilinear(cs, ue + t * du - c0, ve + t * dv - r0) - bend * t * (1 - t)

        # gap is exactly quadratic in t: interpolate through three points
        g0, g1, g2 = gap(ta), gap(mid), gap(tb)
        hh = (tb - ta) / 2
        a2 = (g0 - 2 * g1 + g2) / (2 * hh * hh)
        a1 = (g2 - g0) / (2 * hh) - 2 * a2 * mid
        a0 = g1 - a2 * mid * mid - a1 * mid
        q = _trim([a0, a1, a2])
        assert all(_ev(q, t) == g for t, g in ((ta, g0), (mid, g1), (tb, g2)))
        n = _sub(_mul(_deriv(q), w), _mul(q, [F(1, 2) * c for c in _deriv(w)]))
        cand = [t for t in (ta, tb) if 0 < t < 1]
        for u, v in _roots(n, ta, tb):
            cand += [t for t in (u, v) if 0 < t < 1]
        for t in cand:
            phi = Decimal(_ev(q, t).numerator) / Decimal(_ev(q, t).denominator) / _dsqrt(t * (1 - t))
            if best is None or phi < best:
                best, best_t = phi, t
    if best is None:
        return Decimal("Infinity"), None, math.inf
    rho = best * scale
    return rho, best_t, float(rho)
