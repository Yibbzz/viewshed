"""Solar position (NOAA solar calculator equations, after Meeus), for sun-hours maps.

`sun_position(lat, lon, unix_seconds)` -> (azimuth_deg, apparent_elevation_deg, true_elevation_deg):
azimuth clockwise from true north; apparent elevation includes the NOAA approximation of
astronomical refraction. Accuracy ~0.01 deg for 1950-2050 (validated against NREL SPA in
tests/test_sunpos.py). The crate's sun_position performs the same float operations in the same
order (same libm), so both give the same floats.
"""

import math


def _refraction_deg(e):
    """NOAA's approximation of atmospheric refraction (degrees) at true elevation e (degrees)."""
    if e > 85.0:
        return 0.0
    if e > 5.0:
        t = math.tan(math.radians(e))
        return (58.1 / t - 0.07 / (t * t * t) + 0.000086 / (t * t * t * t * t)) / 3600.0
    if e > -0.575:
        return (1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))) / 3600.0
    return (-20.774 / math.tan(math.radians(e))) / 3600.0


def sun_position(lat, lon, unix_seconds):
    """(azimuth, apparent elevation, true elevation) in degrees; lat/lon in degrees (east +)."""
    if not (-90.0 <= lat <= 90.0) or not math.isfinite(lon) or not math.isfinite(unix_seconds):
        raise ValueError("latitude must be in [-90, 90] and longitude, time finite")
    jd = unix_seconds / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    l0 = math.fmod(280.46646 + t * (36000.76983 + t * 0.0003032), 360.0)
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2.0 * mr) * (0.019993 - 0.000101 * t) + math.sin(3.0 * mr) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * t)
    app_long = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(eps) * math.sin(app_long))
    y = math.tan(eps / 2.0)
    y = y * y
    l0r = math.radians(l0)
    eot = 4.0 * math.degrees(y * math.sin(2.0 * l0r) - 2.0 * ecc * math.sin(mr)
                             + 4.0 * ecc * y * math.sin(mr) * math.cos(2.0 * l0r)
                             - 0.5 * y * y * math.sin(4.0 * l0r) - 1.25 * ecc * ecc * math.sin(2.0 * mr))
    minutes = math.fmod(unix_seconds, 86400.0) / 60.0
    if minutes < 0.0:
        minutes += 1440.0
    tst = math.fmod(minutes + eot + 4.0 * lon, 1440.0)
    if tst < 0.0:
        tst += 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    cz = min(1.0, max(-1.0, cz))
    elev = 90.0 - math.degrees(math.acos(cz))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la))) + 180.0
    az = math.fmod(az, 360.0)
    return az, elev + _refraction_deg(elev), elev
