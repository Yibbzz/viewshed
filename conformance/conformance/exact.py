"""Order-independent reductions for scene generation.

numpy's float sums depend on its summation order, which changed between numpy 1
and 2 and can differ by platform, so z.mean() and z.std() can differ in the last
bit. math.fsum is correctly rounded whatever the order, so the scenes (and the
golden input hashes built from them) come out bit-identical everywhere.
"""

import math

import numpy as np


def mean(a):
    v = np.asarray(a, np.float64).ravel().tolist()
    return math.fsum(v) / len(v)


def std(a):
    """Population standard deviation, like a.std()."""
    a = np.asarray(a, np.float64)
    d = (a - mean(a)).ravel()
    return math.sqrt(math.fsum((d * d).tolist()) / d.size)


# Smooth periodic functions from +, -, *, / and floor only. Those are correctly
# rounded IEEE operations, so the result is bit-identical on every platform and
# numpy version, unlike np.sin/np.cos/np.exp (libm and numpy's SIMD code differ
# in the last bit between Linux, macOS, Windows and CPUs). They only need to be
# smooth hill shapes, not accurate approximations.

_PI = 3.141592653589793
_TWO_PI = 2 * _PI
_PI2 = _PI * _PI
_PEAK = 2 * _PI2 * _PI / (3 * math.sqrt(3))  # max of r (pi^2 - r^2), at r = pi / sqrt 3


def wave(t):
    """A sine-like wave with period 2 pi and peak +-1: r (pi^2 - r^2) on the
    reduced angle r in [-pi, pi). Odd with zeros at +-pi, so it and its slope
    are continuous across periods."""
    t = np.asarray(t, np.float64)
    r = t - np.floor(t / _TWO_PI + 0.5) * _TWO_PI
    return r * (_PI2 - r * r) / _PEAK


def cowave(t):
    """wave shifted a quarter period: the cosine-like partner."""
    return wave(np.asarray(t, np.float64) + _PI / 2)


def bump(u):
    """A smooth bump, 1 at u = 0 and falling off like a Gaussian ridge: 1 / (1 + u^2)^2."""
    u = np.asarray(u, np.float64)
    s = 1 + u * u
    return 1 / (s * s)


def noise(rng, shape):
    """Uniform noise in [-1, 1) from rng.random(), which is integer arithmetic
    plus an exact conversion; rng.normal calls libm exp/log on some draws."""
    return rng.random(shape) * 2 - 1
