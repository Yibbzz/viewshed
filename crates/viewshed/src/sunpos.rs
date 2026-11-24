//! Solar position and sun-hours maps (hours of direct sun per cell over a time range).
//!
//! Sun position: the NOAA solar calculator equations (after Meeus), ~0.01 deg for 1950-2050 (within
//! 0.018 deg of NREL SPA over 20,000 random times and places in the conformance tests). The
//! apparent elevation includes NOAA's approximation of *astronomical* refraction (the bending of
//! sunlight near the horizon); the line of sight to the sun uses the *terrestrial* refraction of
//! [`ShadowOptions`] (curvature), as the viewshed does. Two different things.
//!
//! Sun hours: exact shadow maps ([`shadow`](crate::shadow)) at sampled times, one sun position per
//! time for the whole raster (across 50 km the sun's azimuth varies by ~0.3 deg). Time sampling is
//! the caller's choice (midpoint rule): shadows are exact at each sample, the time integral is not.

use rayon::prelude::*;

use crate::shadow::{shadow, ShadowOptions, Sun};
use crate::{Error, Grid};

/// Position of the sun in the sky (degrees): azimuth clockwise from true north, apparent
/// (refracted) and true elevation above the horizon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunPosition {
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub elevation_true_deg: f64,
}

fn refraction_deg(e: f64) -> f64 {
    if e > 85.0 {
        0.0
    } else if e > 5.0 {
        let t = e.to_radians().tan();
        (58.1 / t - 0.07 / (t * t * t) + 0.000086 / (t * t * t * t * t)) / 3600.0
    } else if e > -0.575 {
        (1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))) / 3600.0
    } else {
        (-20.774 / e.to_radians().tan()) / 3600.0
    }
}

/// Solar position at latitude/longitude (degrees, east positive) and a UTC time in seconds since
/// 1970-01-01 (fractional allowed). The same float operations as the conformance oracle
/// (conformance/sunpos.py).
pub fn sun_position(lat_deg: f64, lon_deg: f64, unix_seconds: f64) -> Result<SunPosition, Error> {
    if !(-90.0..=90.0).contains(&lat_deg) || !lon_deg.is_finite() || !unix_seconds.is_finite() {
        return Err(Error::InvalidSunHours);
    }
    let jd = unix_seconds / 86400.0 + 2440587.5;
    let t = (jd - 2451545.0) / 36525.0;
    let l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0;
    let m = 357.52911 + t * (35999.05029 - 0.0001537 * t);
    let ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
    let mr = m.to_radians();
    let c = mr.sin() * (1.914602 - t * (0.004817 + 0.000014 * t)) + (2.0 * mr).sin() * (0.019993 - 0.000101 * t) + (3.0 * mr).sin() * 0.000289;
    let omega = (125.04 - 1934.136 * t).to_radians();
    let app_long = (l0 + c - 0.00569 - 0.00478 * omega.sin()).to_radians();
    let eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0;
    let eps = (eps0 + 0.00256 * omega.cos()).to_radians();
    let decl = (eps.sin() * app_long.sin()).asin();
    let mut y = (eps / 2.0).tan();
    y *= y;
    let l0r = l0.to_radians();
    let eot = 4.0
        * (y * (2.0 * l0r).sin() - 2.0 * ecc * mr.sin() + 4.0 * ecc * y * mr.sin() * (2.0 * l0r).cos()
            - 0.5 * y * y * (4.0 * l0r).sin()
            - 1.25 * ecc * ecc * (2.0 * mr).sin())
        .to_degrees();
    let mut minutes = (unix_seconds % 86400.0) / 60.0;
    if minutes < 0.0 {
        minutes += 1440.0;
    }
    let mut tst = (minutes + eot + 4.0 * lon_deg) % 1440.0;
    if tst < 0.0 {
        tst += 1440.0;
    }
    let ha = (tst / 4.0 - 180.0).to_radians();
    let la = lat_deg.to_radians();
    let cz = (la.sin() * decl.sin() + la.cos() * decl.cos() * ha.cos()).clamp(-1.0, 1.0);
    let elev = 90.0 - cz.acos().to_degrees();
    let az = (ha.sin().atan2(ha.cos() * la.sin() - decl.tan() * la.cos()).to_degrees() + 180.0) % 360.0;
    Ok(SunPosition { azimuth_deg: az, elevation_deg: elev + refraction_deg(elev), elevation_true_deg: elev })
}

/// Hours of direct sun per cell for sun positions with weights in hours: the sum of the weights
/// of the positions at which the cell is lit (exact shadows). Azimuths here are relative to the
/// raster's grid north. Elevation <= 0: the sun is down (adds nothing); elevation >= 90: every
/// cell with data is lit. No-data cells: NaN.
pub fn sun_hours(grid: &Grid, positions: &[(Sun, f64)], opts: &ShadowOptions) -> Result<Vec<f64>, Error> {
    if positions.iter().any(|(s, w)| !(w.is_finite() && *w >= 0.0) || !s.azimuth_deg.is_finite() || s.elevation_deg.is_nan()) {
        return Err(Error::InvalidSunHours);
    }
    let n = grid.rows * grid.cols;
    let mut hours = vec![0.0f64; n];
    for &(sun, w) in positions {
        if sun.elevation_deg <= 0.0 || w == 0.0 {
            continue;
        }
        if sun.elevation_deg >= 90.0 {
            hours.par_iter_mut().zip(grid.z.par_iter()).for_each(|(h, z)| {
                if !z.is_nan() {
                    *h += w;
                }
            });
            continue;
        }
        let lit = shadow(grid, sun, opts)?;
        hours.par_iter_mut().zip(lit.par_iter()).for_each(|(h, &l)| {
            if l {
                *h += w;
            }
        });
    }
    hours.par_iter_mut().zip(grid.z.par_iter()).for_each(|(h, z)| {
        if z.is_nan() {
            *h = f64::NAN;
        }
    });
    Ok(hours)
}

/// Grid-relative sun positions at midpoint samples start + (i + 1/2) step (i < floor((end - start) /
/// step)), each weighted step / 3600 hours; samples with the sun down are dropped. Grid azimuth =
/// true azimuth - `grid_convergence_deg` (the meridian convergence of the raster's projection).
pub fn sun_samples(lat_deg: f64, lon_deg: f64, start_unix: f64, end_unix: f64, step_s: f64, grid_convergence_deg: f64) -> Result<Vec<(Sun, f64)>, Error> {
    if !(step_s > 0.0 && step_s.is_finite() && end_unix >= start_unix && grid_convergence_deg.is_finite()) {
        return Err(Error::InvalidSunHours);
    }
    let n = ((end_unix - start_unix) / step_s).floor() as u64;
    let mut out = Vec::new();
    for i in 0..n {
        let p = sun_position(lat_deg, lon_deg, start_unix + (i as f64 + 0.5) * step_s)?;
        if p.elevation_deg > 0.0 {
            out.push((Sun { azimuth_deg: (p.azimuth_deg - grid_convergence_deg + 720.0) % 360.0, elevation_deg: p.elevation_deg }, step_s / 3600.0));
        }
    }
    Ok(out)
}

/// Hours of direct sun per cell between two UTC times (seconds since 1970), sampled every
/// `step_s` seconds (midpoint rule), with the sun's position at `lat_deg`, `lon_deg` for the whole
/// raster. See [`sun_samples`] and [`sun_hours`].
#[allow(clippy::too_many_arguments)]
pub fn sun_hours_between(grid: &Grid, lat_deg: f64, lon_deg: f64, start_unix: f64, end_unix: f64, step_s: f64, grid_convergence_deg: f64, opts: &ShadowOptions) -> Result<Vec<f64>, Error> {
    sun_hours(grid, &sun_samples(lat_deg, lon_deg, start_unix, end_unix, step_s, grid_convergence_deg)?, opts)
}
