//! Sky view factor and openness from exact horizons.
//!
//! Every per-direction horizon comes from [`horizon_rise_vec`](crate::horizon_rise_vec): exact on the
//! bilinear surface (certified within 1e-9 res in the tangent), along each cell's own ray. What
//! remains approximate is the azimuth sampling: `azimuths` directions evenly spaced clockwise from
//! grid north (0°, 360°/K, ...). With K = 32 the mean azimuth-sampling error of the sky view factor
//! is about 5e-4 on rough mountains and 4e-5 on plains (research note svf-2026-11-24.md).

use rayon::prelude::*;

use crate::horizon::{horizon_rise_vec, HorizonOptions};
use crate::shadow::azimuth_vector;
use crate::{Error, Grid, MEAN_EARTH_RADIUS};

/// Which sky view factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyViewFactor {
    /// Fraction of the sky hemisphere above the horizontal that is not hidden by terrain
    /// (Zakšek et al. 2011; RVT): `1 - mean_k sin(max(H_k, 0))`, H_k the horizon elevation in
    /// direction k. Horizons below the horizontal count as 0; so do directions with nothing ahead.
    VisibleSky,
    /// Isotropic diffuse sky view factor of the sloped surface (Dozier & Frew 1990, eq. 7b; IPW and
    /// topocalc `viewf`): `mean_k max(0, cos S (cos² e_k - g_k ((π/2 - e_k) - sin e_k cos e_k)))`,
    /// S the slope, g_k the directional slope towards k, e_k the horizon elevation raised to at least
    /// the cell's own tangent plane (e_k = max(H_k, atan g_k)). An unobstructed plane gives 1 (its
    /// whole sky hemisphere is open; (1 + cos S) / 2 is a tilted element in horizontal surroundings,
    /// and topocalc, which clamps horizons at the horizontal, gives about cos S). Slope from central
    /// differences of the cell-centre heights (the mean of the
    /// bilinear surface's one-sided slopes at the centre; one-sided next to an edge or no-data, 0 on
    /// an axis with no neighbour). Not clamped: with few directions the mean can exceed 1 slightly
    /// (the per-direction terms are above 1 downslope, below upslope): up to ~0.02 at K = 8, 1e-4 at
    /// K = 32 on test terrain.
    Isotropic,
}

/// Sky view factor settings. Curvature and refraction as for horizons.
#[derive(Debug, Clone, PartialEq)]
pub struct SkyOptions {
    /// Number of directions K (evenly spaced from grid north, clockwise). Default 32.
    pub azimuths: usize,
    /// Only terrain within this horizontal distance counts (raster units); `None`: all of it.
    pub max_distance: Option<f64>,
    /// Height above the surface of the point in each cell the sky is seen from.
    pub target_height: f64,
    pub curvature: bool,
    pub refraction: f64,
    pub earth_radius: f64,
}

impl Default for SkyOptions {
    fn default() -> Self {
        SkyOptions { azimuths: 32, max_distance: None, target_height: 0.0, curvature: true, refraction: 0.13, earth_radius: MEAN_EARTH_RADIUS }
    }
}

/// Which openness (Yokoyama et al. 2002).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Openness {
    /// Mean over directions of the zenith angle of the horizon, 90° - H_k (degrees, unclamped:
    /// above 90 where the terrain falls away). A direction with nothing within reach counts as
    /// H_k = -90 (180°), as in RVT.
    Positive,
    /// The same for the surface turned upside down (-z): mean of 90° - D_k, D_k the largest
    /// depression angle (nadir side).
    Negative,
}

/// Openness settings. Openness is measured at the surface (no target height) and without earth
/// curvature (turning the surface upside down would make curvature raise distant terrain).
#[derive(Debug, Clone, PartialEq)]
pub struct OpennessOptions {
    /// Number of directions K (evenly spaced from grid north, clockwise). Default 32.
    pub azimuths: usize,
    /// The search radius L (raster units); `None`: unlimited.
    pub max_distance: Option<f64>,
}

impl Default for OpennessOptions {
    fn default() -> Self {
        OpennessOptions { azimuths: 32, max_distance: None }
    }
}

/// The K azimuths, degrees clockwise from grid north: 360 k / K.
fn azimuths(k: usize) -> Vec<f64> {
    (0..k).map(|i| 360.0 * i as f64 / k as f64).collect()
}

/// For each direction: (azimuth, per cell the horizon as (sin e, cos e), e the elevation angle; the
/// "nothing ahead" value is (-1, 0)). The tangent is rise / scale, scale = |v| res.
fn horizons(grid: &Grid, k: usize, opts: &HorizonOptions) -> Result<Vec<(f64, Vec<(f64, f64)>)>, Error> {
    // The directions are independent. Computing them in parallel is much faster than
    // computing 32 exact horizon maps one after another; Rayon handles the nested
    // parallelism inside each horizon map as well.
    azimuths(k)
        .into_par_iter()
        .map(|az| {
            let v = azimuth_vector(grid, az)?;
            let rise = horizon_rise_vec(grid, v, opts)?;
            let scale = (v[0] * v[0] + v[1] * v[1]).sqrt() * grid.res();
            let sc = rise
                .into_iter()
                .map(|h| {
                    if h.is_nan() {
                        (f64::NAN, f64::NAN)
                    } else if h == f64::NEG_INFINITY {
                        (-1.0, 0.0)
                    } else {
                        let d = h.hypot(scale);
                        (h / d, scale / d)
                    }
                })
                .collect();
            Ok((az, sc))
        })
        .collect::<Result<Vec<_>, Error>>()
}

/// Sky view factor per cell (0..1; NaN on no-data). See [`SkyViewFactor`] for the definitions.
pub fn sky_view_factor(grid: &Grid, kind: SkyViewFactor, opts: &SkyOptions) -> Result<Vec<f64>, Error> {
    if opts.azimuths == 0 {
        return Err(Error::InvalidAzimuthCount);
    }
    let hopts = HorizonOptions {
        target_height: opts.target_height,
        curvature: opts.curvature,
        refraction: opts.refraction,
        earth_radius: opts.earth_radius,
        max_distance: opts.max_distance,
    };
    let n = grid.z.len();
    let grad = if kind == SkyViewFactor::Isotropic { gradient(grid) } else { Vec::new() };
    let mut acc = vec![0.0f64; n];
    for dir in horizons(grid, opts.azimuths, &hopts)? {
        let (az, sc) = dir;
        let (ue, un) = (az.to_radians().sin(), az.to_radians().cos());
        for i in 0..n {
            let (se, ce) = sc[i];
            if se.is_nan() {
                continue;
            }
            match kind {
                SkyViewFactor::VisibleSky => acc[i] += se.max(0.0),
                SkyViewFactor::Isotropic => {
                    let (gx, gy) = grad[i];
                    let g = gx * ue + gy * un; // slope towards the direction (rise per run)
                    let cos_s = 1.0 / (1.0 + gx * gx + gy * gy).sqrt();
                    // raise the horizon to the tangent plane: e >= atan g
                    let (dp, gp) = (g.hypot(1.0), g);
                    let (se, ce) = if se * dp < gp * ce { (gp / dp, 1.0 / dp) } else { (se, ce) };
                    let e = se.atan2(ce);
                    let t = cos_s * (ce * ce - g * ((std::f64::consts::FRAC_PI_2 - e) - se * ce));
                    acc[i] += t.max(0.0);
                }
            }
        }
    }
    let k = opts.azimuths as f64;
    Ok(grid
        .z
        .iter()
        .zip(acc)
        .map(|(z, a)| {
            if z.is_nan() {
                f64::NAN
            } else {
                match kind {
                    SkyViewFactor::VisibleSky => 1.0 - a / k,
                    SkyViewFactor::Isotropic => a / k,
                }
            }
        })
        .collect())
}

/// Openness per cell, degrees (NaN on no-data). See [`Openness`].
pub fn openness(grid: &Grid, kind: Openness, opts: &OpennessOptions) -> Result<Vec<f64>, Error> {
    if opts.azimuths == 0 {
        return Err(Error::InvalidAzimuthCount);
    }
    let hopts = HorizonOptions { target_height: 0.0, curvature: false, refraction: 0.0, earth_radius: MEAN_EARTH_RADIUS, max_distance: opts.max_distance };
    let neg: Vec<f64>;
    let flipped;
    let g = match kind {
        Openness::Positive => grid,
        Openness::Negative => {
            neg = grid.z.iter().map(|v| -v).collect();
            flipped = Grid::new(&neg, grid.rows, grid.cols, grid.transform)?;
            &flipped
        }
    };
    let n = grid.z.len();
    let mut acc = vec![0.0f64; n];
    for dir in horizons(g, opts.azimuths, &hopts)? {
        let (_, sc) = dir;
        for i in 0..n {
            let (se, ce) = sc[i];
            if !se.is_nan() {
                acc[i] += 90.0 - se.atan2(ce).to_degrees();
            }
        }
    }
    let k = opts.azimuths as f64;
    Ok(grid.z.iter().zip(acc).map(|(z, a)| if z.is_nan() { f64::NAN } else { a / k }).collect())
}

/// Per cell (dz/dEast, dz/dNorth): central differences of the cell-centre heights, one-sided next to
/// an edge or no-data, 0 on an axis with no neighbour (NaN cells: unused).
fn gradient(grid: &Grid) -> Vec<(f64, f64)> {
    let (rows, cols) = (grid.rows, grid.cols);
    let res = grid.res();
    let at = |r: isize, c: isize| -> Option<f64> {
        if r < 0 || c < 0 || r as usize >= rows || c as usize >= cols {
            return None;
        }
        let v = grid.z[r as usize * cols + c as usize];
        if v.is_nan() { None } else { Some(v) }
    };
    let d = |lo: Option<f64>, mid: f64, hi: Option<f64>| -> f64 {
        match (lo, hi) {
            (Some(a), Some(b)) => (b - a) / (2.0 * res),
            (None, Some(b)) => (b - mid) / res,
            (Some(a), None) => (mid - a) / res,
            (None, None) => 0.0,
        }
    };
    let mut out = vec![(0.0, 0.0); rows * cols];
    for r in 0..rows as isize {
        for c in 0..cols as isize {
            let Some(z) = at(r, c) else { continue };
            let gx = d(at(r, c - 1), z, at(r, c + 1));
            // north = up = decreasing row index
            let gy = d(at(r + 1, c), z, at(r - 1, c));
            out[r as usize * cols + c as usize] = (gx, gy);
        }
    }
    out
}
