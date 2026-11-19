"""Fast, exact line-of-sight viewsheds.

Version 0.0.x is the reference implementation in NumPy: a straightforward,
correct line-of-sight check from the observer to every cell. It is slow and
meant for small rasters and testing. Later versions keep this API and these
results and move the work to a compiled Rust core.

Surface model: heights at cell centres, bilinearly interpolated between them
(clamped at the raster border). The observer's eye is `observer_height` above
the surface at the observer point. A cell is visible when the straight line
from the eye to the point `target_height` above the cell centre stays on or
above the surface, sampled every quarter cell. NaN cells never block a view
and are never visible. This matches the Rust crate `viewshed`.
"""

import math

import numpy as np

__version__ = "0.0.1"
__all__ = ["viewshed", "line_of_sight", "surface_height"]

_STEPS_PER_CELL = 4.0
_EPS = 1e-9


def _check(dem, transform):
    dem = np.asarray(dem, dtype=float)
    if dem.ndim != 2:
        raise ValueError("dem must be a 2-D array")
    if transform[2] != 0 or transform[4] != 0:
        raise ValueError("rotated rasters are not supported")
    w, h = abs(transform[1]), abs(transform[5])
    if w == 0 or abs(w - h) > 1e-9 * w:
        raise ValueError("cells must be square with a non-zero size")
    return dem


def surface_height(dem, transform, x, y):
    """Bilinear surface height at map position(s) (x, y); NaN outside the raster."""
    dem = np.asarray(dem, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rows, cols = dem.shape
    fc = (x - transform[0]) / transform[1] - 0.5
    fr = (y - transform[3]) / transform[5] - 0.5
    outside = (fc < -0.5) | (fr < -0.5) | (fc > cols - 0.5) | (fr > rows - 0.5)
    c0, r0 = np.floor(fc), np.floor(fr)
    tc, tr = fc - c0, fr - r0
    ca = np.clip(c0, 0, cols - 1).astype(int)
    cb = np.clip(c0 + 1, 0, cols - 1).astype(int)
    ra = np.clip(r0, 0, rows - 1).astype(int)
    rb = np.clip(r0 + 1, 0, rows - 1).astype(int)
    top = dem[ra, ca] * (1 - tc) + dem[ra, cb] * tc
    bottom = dem[rb, ca] * (1 - tc) + dem[rb, cb] * tc
    return np.where(outside, np.nan, top * (1 - tr) + bottom * tr)


def line_of_sight(dem, transform, eye, target):
    """True if nothing on the surface rises above the line from `eye` to `target` (x, y, z)."""
    dem = _check(dem, transform)
    return _los(dem, transform, np.asarray(eye, float), np.asarray(target, float))


def _los(dem, transform, eye, target):
    horiz = math.hypot(target[0] - eye[0], target[1] - eye[1])
    n = int(math.ceil(horiz / abs(transform[1]) * _STEPS_PER_CELL))
    if n < 2:
        return True
    t = np.arange(1, n) / n
    pts = eye + t[:, None] * (target - eye)
    surface = surface_height(dem, transform, pts[:, 0], pts[:, 1])
    line = pts[:, 2]
    with np.errstate(invalid="ignore"):
        blocked = surface > line + _EPS * (1 + np.abs(line))   # NaN never blocks
    return not bool(np.any(blocked))


def viewshed(dem, transform, observer, observer_height=1.6, target_height=0.0, max_distance=None):
    """Visibility of every cell from `observer` = (x, y).

    dem: 2-D array of surface heights (first row = top), NaN for no data.
    transform: GDAL-style geotransform (x0, cell width, 0, y0, 0, -cell height).
    Returns a boolean array the same shape as `dem`.
    """
    dem = _check(dem, transform)
    x, y = float(observer[0]), float(observer[1])
    ground = float(surface_height(dem, transform, x, y))
    rows, cols = dem.shape
    fc = (x - transform[0]) / transform[1]
    fr = (y - transform[3]) / transform[5]
    if not (0 <= fc <= cols and 0 <= fr <= rows):
        raise ValueError("the observer is outside the raster")
    if math.isnan(ground):
        raise ValueError("there is no height data under the observer")
    eye = np.array([x, y, ground + observer_height])
    out = np.zeros(dem.shape, dtype=bool)
    xs = transform[0] + (np.arange(cols) + 0.5) * transform[1]
    ys = transform[3] + (np.arange(rows) + 0.5) * transform[5]
    for r in range(rows):
        for c in range(cols):
            z = dem[r, c]
            if math.isnan(z):
                continue
            if max_distance is not None and math.hypot(xs[c] - x, ys[r] - y) > max_distance:
                continue
            out[r, c] = _los(dem, transform, eye, np.array([xs[c], ys[r], z + target_height]))
    return out
