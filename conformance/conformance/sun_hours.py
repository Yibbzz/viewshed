"""Sun-hours oracle: hours of direct sun per cell over a time range, from exact shadow maps
(shadow_oracle) at midpoint time samples. One sun position per time for the whole raster.

Samples: t_i = start + (i + 1/2) step, i < n = floor((end - start) / step); each adds step / 3600
hours where the cell is lit; a partial last step (remainder r > 0) adds one sample at its midpoint
weighted r / 3600. Sun below the horizon (apparent elevation <= 0): nothing lit. Apparent
elevation >= 90: every cell with data lit. Grid azimuth = true azimuth - grid_convergence (degrees;
meridian convergence of the raster's projection at the area). No-data: NaN.
"""

import math

import numpy as np

from . import shadow_oracle as so
from .sunpos import sun_position


def samples(lat, lon, start, end, step, grid_convergence=0.0):
    """[(grid azimuth, apparent elevation, hours)] for the sun above the horizon: midpoint samples
    of whole steps, plus one at the midpoint of a partial last step (weighted by its length)."""
    if not (-90.0 <= lat <= 90.0 and math.isfinite(lon) and math.isfinite(grid_convergence)):
        raise ValueError("latitude must be in [-90, 90] degrees and longitude and grid convergence finite")
    if not (step > 0 and math.isfinite(step) and math.isfinite(start) and math.isfinite(end) and end >= start):
        raise ValueError("times must be finite, the end not before the start, and the time step positive")
    n = int(math.floor((end - start) / step))
    times = [(start + (i + 0.5) * step, step) for i in range(n)]
    done = start + n * step
    rem = end - done
    if rem > 0.0:
        times.append((done + 0.5 * rem, rem))
    out = []
    for t, w in times:
        az, el, _ = sun_position(lat, lon, t)
        if el > 0.0:
            out.append((math.fmod(az - grid_convergence + 720.0, 360.0), el, w / 3600.0))
    return out


def sun_hours_at(z, transform, positions, target_height=0.0, curvature=False, refraction=0.13,
                 earth_radius=6371008.8):
    """Hours lit per cell for [(grid azimuth, elevation, hours)] (the core, any weights)."""
    z = np.ascontiguousarray(z, dtype=float)
    hours = np.zeros(z.shape)
    for az, el, w in positions:
        if el >= 90.0:
            lit = ~np.isnan(z)
        else:
            lit, _ = so.shadow(z, transform, az, el, target_height, curvature, refraction, earth_radius)
        hours += np.where(lit, w, 0.0)
    hours[np.isnan(z)] = np.nan
    return hours


def sun_hours(z, transform, lat, lon, start, end, step, grid_convergence=0.0, **kw):
    """Hours of direct sun per cell between unix times start and end (seconds, UTC)."""
    return sun_hours_at(z, transform, samples(lat, lon, start, end, step, grid_convergence), **kw)
