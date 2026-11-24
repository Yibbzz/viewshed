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

`intervisible` answers exactly, for many pairs of points at once, whether each
pair can see each other (radio links).

`reverse_viewshed` turns the question round: from which cells can a target
(a mast top, a roof) be seen?

`shadow` gives exact cast shadows on the same bilinear surface for a sun at a
given azimuth and elevation, and `sun_hours` adds them up over a time range: hours
of direct sun per cell. `horizon_map` gives, for every cell, the exact elevation angle of
the skyline in one direction.

Earth curvature and refraction are on by default: every point is lowered by
d^2 / (2 R) * (1 - refraction) at horizontal distance d, as in GRASS
`r.viewshed -c -r` and GDAL's curvature coefficient.

The work is done in Rust on all CPU cores, with the GIL released.
"""

import math
from collections import namedtuple
from datetime import datetime, timedelta

import numpy as np

from . import _viewshed, reference

__version__ = "0.3.0"
__all__ = ["viewshed", "reverse_viewshed", "intervisible", "ztv", "shadow", "horizon_map", "sun_hours", "sun_position", "MEAN_EARTH_RADIUS", "reference"]

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


def reverse_viewshed(dem, transform, target, target_height=0.0, observer_height=1.6,
                     max_distance=None, curvature=True, refraction=0.13,
                     earth_radius=MEAN_EARTH_RADIUS, surface="rviewshed"):
    """From which cells can `target` = (x, y) be seen? A bool array shaped like `dem`.

    target_height: the target's height above the surface at (x, y) (a mast, a roof).
    observer_height: eye height above the surface at every cell.
    max_distance, curvature, refraction, earth_radius, surface: as for `viewshed`.

    A cell is True when an observer standing there sees the target: for every cell,
    the same answer as `viewshed` from that cell, read at the target's cell.
    With surface="bilinear" this costs one exact viewshed (sightlines, the blocking
    rule and the curvature drop are the same in both directions, so it is a viewshed
    from the target with the two heights swapped); the target stays at its exact
    point. The default "rviewshed" model is not symmetric, so every cell looks at the
    target's cell centre in turn (as `ztv` does): exact for the model, but slower.
    No-data cells are False.
    """
    _heights(("target_height", target_height), ("observer_height", observer_height))
    x, y = (float(v) for v in target)
    try:
        return _viewshed.reverse_viewshed(_dem(dem), _transform(transform), x, y,
                                          float(observer_height), float(target_height),
                                          _distance(max_distance), bool(curvature),
                                          float(refraction), float(earth_radius), str(surface))
    except ValueError as e:
        msg = str(e).replace("the observer", "the target")
        raise ValueError(msg) from None


def intervisible(dem, transform, a, b, curvature=True, refraction=0.13,
                 earth_radius=MEAN_EARTH_RADIUS):
    """Can each point in `a` see the matching point in `b`? Exact line of sight.

    a, b: points (x, y, height above the surface), one point or an array of shape (n, 3)
        each (radio masts, sensors, ...). Points can be anywhere in the raster.
    curvature, refraction, earth_radius: as for `viewshed`.
    Returns a bool, or a bool array of length n.

    Exact on the bilinear surface (as `surface="bilinear"`): visible when the segment
    between the two points is nowhere strictly below the surface along a stretch of
    positive length. The answer doesn't depend on which point is `a`. Pairs are decided
    in parallel; a point outside the raster or over no-data raises ValueError naming its
    pair.
    """
    pa, pb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    single = pa.ndim == 1 and pb.ndim == 1
    pa, pb = np.atleast_2d(pa), np.atleast_2d(pb)
    if pa.shape[1:] != (3,) or pb.shape[1:] != (3,):
        raise ValueError("points must be (x, y, height) or arrays of shape (n, 3)")
    if len(pa) != len(pb):
        if len(pa) == 1:
            pa = np.repeat(pa, len(pb), axis=0)
        elif len(pb) == 1:
            pb = np.repeat(pb, len(pa), axis=0)
        else:
            raise ValueError(f"a has {len(pa)} points and b has {len(pb)}")
    pairs = [(tuple(p), tuple(q)) for p, q in zip(pa.tolist(), pb.tolist())]
    seen = _viewshed.intervisible(_dem(dem), _transform(transform), pairs, bool(curvature),
                                  float(refraction), float(earth_radius))
    return bool(seen[0]) if single else seen


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


def shadow(dem, transform, azimuth, elevation, target_height=0.0, curvature=True,
           refraction=0.13, earth_radius=MEAN_EARTH_RADIUS):
    """Which cells are lit by a sun at (azimuth, elevation)? A bool array shaped like `dem`.

    azimuth: degrees clockwise from north (90 = east). elevation: degrees above the
    horizon, strictly between 0 and 90.
    target_height: height above each cell's surface that has to be lit.
    curvature, refraction, earth_radius: as for `viewshed`; refraction above 1 with
    curvature on is not supported.

    Exact on the bilinear surface (heights at cell centres, bilinear between them): a
    cell is shadowed when its ray towards the sun passes strictly below the surface
    along a stretch of positive length. Exact for the direction vector computed from
    the two angles (cardinal azimuths are exact). Rays leaving the raster are lit; no-data
    cells are False and never cast shadow.
    """
    _heights(("target_height", target_height))
    return _viewshed.shadow(_dem(dem), _transform(transform), float(azimuth), float(elevation),
                            float(target_height), bool(curvature), float(refraction),
                            float(earth_radius))


def horizon_map(dem, transform, azimuth, target_height=0.0, curvature=True, refraction=0.13,
                earth_radius=MEAN_EARTH_RADIUS):
    """Elevation angle of the horizon, in degrees, seen from every cell towards `azimuth`.
    A float64 array shaped like `dem`.

    azimuth: degrees clockwise from grid north (90 = east).
    target_height: height above each cell's surface that the horizon is seen from.
    curvature, refraction, earth_radius: as for `viewshed` (terrestrial refraction).

    On the bilinear surface, the same model as `shadow`: the highest elevation angle of
    any point of the terrain ahead (on stretches of positive length; no-data never
    counts; the surface ends half a cell beyond the outer cell centres, as for viewsheds).
    -90 where only no-data lies ahead, NaN on no-data cells. Not clamped at 0: a
    horizon below the horizontal is negative (GRASS r.horizon clamps at 0; use
    np.maximum(h, 0) to compare). Consistent with `shadow`: a cell is shadowed by a sun
    at that azimuth exactly when the sun is lower than its horizon.

    Exact to within 1e-9 of a cell size in height per cell size of distance (certified
    per cell), up to the final arctangent.
    """
    _heights(("target_height", target_height))
    return _viewshed.horizon_map(_dem(dem), _transform(transform), float(azimuth),
                                 float(target_height), bool(curvature), float(refraction),
                                 float(earth_radius))


SunPosition = namedtuple("SunPosition", "azimuth elevation elevation_true")
SunPosition.__doc__ = """Degrees: azimuth clockwise from true north, apparent (refracted)
elevation and true (geometric) elevation above the horizon."""


def _unix(t, name):
    if isinstance(t, datetime):
        if t.tzinfo is None or t.utcoffset() is None:
            raise ValueError(f"{name} needs a time zone (e.g. datetime(..., tzinfo=timezone.utc))")
        return t.timestamp()
    if isinstance(t, (int, float, np.integer, np.floating)) and not isinstance(t, bool):
        return float(t)
    raise TypeError(f"{name} must be a timezone-aware datetime or seconds since 1970-01-01 UTC")


def _seconds(step):
    if isinstance(step, timedelta):
        return step.total_seconds()
    return float(step)


def sun_position(lat, lon, time):
    """Where is the sun at latitude/longitude (degrees, east positive) and `time`?

    time: a timezone-aware datetime, or seconds since 1970-01-01 UTC.
    Returns SunPosition(azimuth, elevation, elevation_true) in degrees. NOAA solar
    calculator equations (after Meeus): within about 0.02 degrees of NREL's SPA for
    1950-2050. The apparent elevation includes astronomical refraction (sunlight bending
    near the horizon), which is separate from the terrestrial `refraction` of sightlines.
    """
    return SunPosition(*_viewshed.sun_position(float(lat), float(lon), _unix(time, "time")))


def _centre_and_convergence(dem, transform, crs):
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        raise ImportError("crs= needs pyproj (pip install pyproj); or pass lat, lon and "
                          "grid_convergence yourself") from None
    rows, cols = np.shape(dem)
    gt = _transform(transform)
    x, y = gt[0] + gt[1] * cols / 2, gt[3] + gt[5] * rows / 2
    crs = CRS.from_user_input(crs)
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    from_geo = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon, lat = to_geo.transform(x, y)
    d = 1e-4 if lat < 89.0 else -1e-4
    x2, y2 = from_geo.transform(lon, lat + d)
    if d < 0:
        x2, y2 = 2 * x - x2, 2 * y - y2
    # grid bearing of true north, clockwise from grid north; grid azimuth = true azimuth -
    # convergence, so the convergence is minus that bearing
    north = math.degrees(math.atan2(x2 - x, y2 - y))
    return lat, lon, -north


def sun_hours(dem, transform, start, end, step=timedelta(minutes=15), lat=None, lon=None,
              crs=None, grid_convergence=None, target_height=0.0, curvature=True,
              refraction=0.13, earth_radius=MEAN_EARTH_RADIUS):
    """Hours of direct sun per cell between `start` and `end`. A float64 array shaped
    like `dem`; no-data cells are NaN.

    start, end: timezone-aware datetimes (or seconds since 1970-01-01 UTC).
    step: sampling interval (timedelta or seconds, default 15 minutes). Each sample is
        taken at the middle of its interval and counts for the whole interval; samples
        with the sun below the horizon count for nothing.
    lat, lon: where the sun's position is computed (degrees); one position per sample
        for the whole raster (across 50 km the azimuth varies by about 0.3 degrees).
    crs: the raster's coordinate reference system (anything pyproj accepts, e.g.
        "EPSG:32630"). With it, lat/lon default to the raster's centre and the grid
        convergence (the angle between grid north and true north, a few degrees in UTM
        away from the central meridian) is computed there. Needs pyproj.
    grid_convergence: degrees, grid azimuth = true azimuth - grid_convergence. Overrides
        the value from `crs`; without `crs` it defaults to 0 (grid north = true north).
    target_height, curvature, refraction, earth_radius: as for `shadow`.

    Each sample's shadow map is exact (see `shadow`) for the sun position computed by
    the package; the sum over time is a midpoint-rule estimate, as fine as `step`.
    """
    if crs is not None:
        c_lat, c_lon, c_conv = _centre_and_convergence(dem, transform, crs)
        lat = c_lat if lat is None else lat
        lon = c_lon if lon is None else lon
        if grid_convergence is None:
            grid_convergence = c_conv
    if lat is None or lon is None:
        raise ValueError("pass lat and lon, or crs= to take them from the raster's centre")
    _heights(("target_height", target_height))
    samples = _viewshed.sun_samples(float(lat), float(lon), _unix(start, "start"),
                                    _unix(end, "end"), _seconds(step),
                                    float(grid_convergence or 0.0))
    return _viewshed.sun_hours(_dem(dem), _transform(transform), samples, float(target_height),
                               bool(curvature), float(refraction), float(earth_radius))
