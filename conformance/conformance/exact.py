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
