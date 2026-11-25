"""Sky view factor and openness oracle: exact horizons per azimuth (horizon_oracle, correctly rounded
rises) aggregated with the crate's formulas (viewshed::sky). The only difference to the crate is the
float evaluation of the per-direction terms and the sum."""

import math

import numpy as np

from . import horizon_oracle as ho


def azimuth_vector(transform, az):
    """As shadow.rs azimuth_vector: exact sin/cos at multiples of 90 deg."""
    a = az % 360.0
    sc = {0.0: (0.0, 1.0), 90.0: (1.0, 0.0), 180.0: (0.0, -1.0), 270.0: (-1.0, 0.0)}.get(a)
    if sc is None:
        r = math.radians(a)
        sc = (math.sin(r), math.cos(r))
    res = abs(transform[1])
    return sc[0] * (res / transform[1]), sc[1] * (res / transform[5])


def _sincos(rise, scale):
    if rise != rise:
        return (math.nan, math.nan)
    if rise == -math.inf:
        return (-1.0, 0.0)
    d = math.hypot(rise, scale)
    return (rise / d, scale / d)


def gradient(z, res):
    rows, cols = z.shape
    def at(r, c):
        if 0 <= r < rows and 0 <= c < cols and z[r, c] == z[r, c]:
            return float(z[r, c])
        return None
    def d(lo, mid, hi):
        if lo is not None and hi is not None:
            return (hi - lo) / (2.0 * res)
        if hi is not None:
            return (hi - mid) / res
        if lo is not None:
            return (mid - lo) / res
        return 0.0
    g = np.zeros((rows, cols, 2))
    for r in range(rows):
        for c in range(cols):
            m = at(r, c)
            if m is None:
                continue
            g[r, c] = (d(at(r, c - 1), m, at(r, c + 1)), d(at(r + 1, c), m, at(r - 1, c)))
    return g


def _rises(z, transform, K, th, curvature, refraction, earth_radius, max_dist):
    res = abs(transform[1])
    for i in range(K):
        az = 360.0 * i / K
        v = azimuth_vector(transform, az)
        _, rise = ho.horizon_map(z, transform, v, th, curvature, refraction, earth_radius, max_dist)
        yield az, rise, math.hypot(v[0], v[1]) * res


def sky_view_factor(z, transform, kind, K, th=0.0, curvature=False, refraction=0.13, earth_radius=6371008.8, max_dist=None):
    z = np.ascontiguousarray(z, dtype=float)
    rows, cols = z.shape
    res = abs(transform[1])
    grad = gradient(z, res) if kind == "iso" else None
    acc = np.zeros((rows, cols))
    for az, rise, scale in _rises(z, transform, K, th, curvature, refraction, earth_radius, max_dist):
        ue, un = math.sin(math.radians(az)), math.cos(math.radians(az))
        for r in range(rows):
            for c in range(cols):
                se, ce = _sincos(rise[r, c], scale)
                if se != se:
                    continue
                if kind == "vs":
                    acc[r, c] += max(se, 0.0)
                else:
                    gx, gy = grad[r, c]
                    g = gx * ue + gy * un
                    cos_s = 1.0 / math.sqrt(1.0 + gx * gx + gy * gy)
                    dp = math.hypot(g, 1.0)
                    if se * dp < g * ce:
                        se, ce = g / dp, 1.0 / dp
                    e = math.atan2(se, ce)
                    t = cos_s * (ce * ce - g * ((math.pi / 2 - e) - se * ce))
                    acc[r, c] += max(t, 0.0)
    out = (1.0 - acc / K) if kind == "vs" else acc / K
    out[np.isnan(z)] = np.nan
    return out


def openness(z, transform, kind, K, max_dist=None):
    z = np.ascontiguousarray(z, dtype=float)
    zz = z if kind == "opos" else -z
    acc = np.zeros(z.shape)
    for az, rise, scale in _rises(zz, transform, K, 0.0, False, 0.0, 6371008.8, max_dist):
        for r in range(z.shape[0]):
            for c in range(z.shape[1]):
                se, ce = _sincos(rise[r, c], scale)
                if se == se:
                    acc[r, c] += 90.0 - math.degrees(math.atan2(se, ce))
    out = acc / K
    out[np.isnan(z)] = np.nan
    return out
