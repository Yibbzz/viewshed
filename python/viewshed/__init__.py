"""Fast, exact line-of-sight viewsheds.

    import viewshed
    seen = viewshed.viewshed(dem, transform, observer=(x, y))

`dem` is a 2-D NumPy array of surface heights (first row = top, NaN = no
data) and `transform` a GDAL-style geotransform (x0, cell width, 0, y0, 0,
-cell height). Cells must be square and the raster north-up. Heights and
distances are in the raster's units, which must be metres when `curvature`
is on.

The default surface model is the one GRASS `r.viewshed` uses, and results
match it cell for cell. The observer stands at the centre of its cell. Every
cell closer than a target is reduced to three height samples (its centre and
the two corners that bound it as seen from the observer, each corner
averaged from the four cells meeting there), and a target is hidden when a
closer cell's gradient, interpolated by angle in the target's direction, is
steeper. `surface="bilinear"` is exact line of sight on the bilinear surface instead.

Earth curvature and refraction are on by default: every point is lowered by
d^2 / (2 R) * (1 - refraction) at horizontal distance d, as in GRASS
`r.viewshed -c -r` and GDAL's curvature coefficient.

The work is done in Rust on all CPU cores, with the GIL released.
"""

import numpy as np

from . import _viewshed, reference

__version__ = "0.0.1"
__all__ = ["viewshed", "ztv", "MEAN_EARTH_RADIUS", "reference"]

MEAN_EARTH_RADIUS = _viewshed.MEAN_EARTH_RADIUS
"""IUGG mean earth radius in metres, the default `earth_radius`. GRASS uses the
semi-major axis of its location's ellipsoid instead (6377563.396 for EPSG:27700)."""


def _dem(dem):
    dem = np.ascontiguousarray(dem, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError(f"dem must be a 2-D array, got {dem.ndim}-D")
    return dem


def _transform(transform):
    t = tuple(float(v) for v in transform)
    if len(t) != 6:
        raise ValueError("transform must have 6 numbers (a GDAL geotransform)")
    return t


def _distance(max_distance):
    if max_distance is None:
        return None
    if not max_distance > 0:
        raise ValueError("max_distance must be positive or None")
    return float(max_distance)


def _heights(*pairs):
    for name, value in pairs:
        if not (value >= 0):
            raise ValueError(f"{name} must be >= 0")


def viewshed(dem, transform, observer, observer_height=1.6, target_height=0.0,
             max_distance=None, curvature=True, refraction=0.13,
             earth_radius=MEAN_EARTH_RADIUS, surface="rviewshed"):
    """Which cells can be seen from `observer` = (x, y)? A bool array shaped like `dem`.

    observer_height: eye height above the surface at the observer.
    target_height: height above each cell's surface that has to be visible.
    max_distance: cells farther than this are not visible (None = no limit).
    curvature, refraction, earth_radius: earth curvature correction (see module docs).
    surface: "rviewshed" (default, matches GRASS) or "bilinear" (exact bilinear line of sight).
    """
    _heights(("observer_height", observer_height), ("target_height", target_height))
    x, y = (float(v) for v in observer)
    return _viewshed.viewshed(_dem(dem), _transform(transform), x, y, float(observer_height),
                              float(target_height), _distance(max_distance), bool(curvature),
                              float(refraction), float(earth_radius), str(surface))


def ztv(dem, transform, targets, observer_height=2.0, max_distance=None, method="forward",
        curvature=True, refraction=0.13, earth_radius=MEAN_EARTH_RADIUS):
    """Zone of theoretical visibility: for every cell, how many targets it can see.

    targets: sequence of (x, y, height above the surface), e.g. blade tips.
    observer_height: eye height above every cell.
    method: "forward" (default; every cell is an observer looking at every target,
        which is the definition) or "reciprocal" (one viewshed from each target down
        to every cell: about 5 times faster on a wind farm, and agrees with "forward"
        on all but a few cells at shadow edges).
    Returns a uint32 array shaped like `dem`; no-data cells are 0.
    """
    _heights(("observer_height", observer_height))
    tg = []
    for t in targets:
        x, y, h = (float(v) for v in t)
        _heights(("target height", h))
        tg.append((x, y, h))
    return _viewshed.ztv(_dem(dem), _transform(transform), tg, float(observer_height),
                         _distance(max_distance), str(method), bool(curvature),
                         float(refraction), float(earth_radius))
