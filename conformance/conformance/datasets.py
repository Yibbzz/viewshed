"""Small synthetic rasters for oracle comparison. Deterministic (fixed seeds).

All are 1 m cells with the observer at the centre of the middle cell.
"""

import numpy as np

from . import exact

N = 61
GT = (0.0, 1.0, 0.0, float(N), 0.0, -1.0)
OBSERVER = (N / 2, N / 2)


def _grid():
    c = np.arange(N) + 0.5
    return np.meshgrid(c, N - c)


def _smooth_noise(seed, passes, amp):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (N, N))
    for _ in range(passes):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    return amp * z / exact.std(z)


def flat_wall():
    z = np.zeros((N, N))
    z[18, 10:50] = 4.0
    z[40:52, 44] = 3.0
    return z


def hills():
    x, y = _grid()
    rng = np.random.default_rng(2)
    z = np.zeros((N, N))
    for _ in range(8):
        cx, cy = rng.uniform(0, N, 2)
        z += rng.uniform(2, 8) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * rng.uniform(4, 10) ** 2))
    return z


def rough():
    return _smooth_noise(3, 3, 2.0)


def urban():
    x, y = _grid()
    z = 0.05 * x + 0.02 * y
    rng = np.random.default_rng(4)
    for _ in range(14):
        r0, c0 = rng.integers(0, N - 8, 2)
        if abs(r0 + 4 - N / 2) < 6 and abs(c0 + 4 - N / 2) < 6:
            continue        # keep the observer in the street
        h, w = rng.integers(3, 8, 2)
        z[r0:r0 + h, c0:c0 + w] += rng.uniform(3, 12)
    return z


def nodata_holes():
    z = hills()
    z[5:12, 40:55] = np.nan
    z[45:50, 3:20] = np.nan
    return z


DATASETS = {"flat_wall": flat_wall, "hills": hills, "rough": rough, "urban": urban,
            "nodata_holes": nodata_holes}
TARGET_HEIGHTS = (0.0, 1.5)
OBSERVER_HEIGHT = 1.6
