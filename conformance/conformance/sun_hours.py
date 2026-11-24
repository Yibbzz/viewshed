"""Sun-hours oracle: hours of direct sun per cell over a time range, from exact shadow maps
(shadow_oracle) at midpoint time samples. One sun position per time for the whole raster.

Samples: t_i = start + (i + 1/2) step, i < n = floor((end - start) / step); each adds step / 3600
hours where the cell is lit. Sun below the horizon (apparent elevation <= 0): nothing lit. Apparent
elevation >= 90: every cell with data lit. Grid azimuth = true azimuth - grid_convergence (degrees;
meridian convergence of the raster's projection at the area). No-data: NaN.
"""

import math

import numpy as np

from . import shadow_oracle as so
from .sunpos import sun_position


def samples(lat, lon, start, end, step, grid_convergence=0.0):
    """[(grid azimuth, apparent elevation, hours)] for the sun above the horizon."""
    if not (step > 0 and end >= start):
        raise ValueError("need step > 0 and end >= start")
    n = int(math.floor((end - start) / step))
    out = []
    for i in range(n):
        az, el, _ = sun_position(lat, lon, start + (i + 0.5) * step)
        if el > 0.0:
            out.append((math.fmod(az - grid_convergence + 720.0, 360.0), el, step / 3600.0))
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
