//! Fast, exact line-of-sight viewsheds.
//!
//! # Surface models
//!
//! [`Surface::RViewshed`] (the default) is the model GRASS `r.viewshed` uses,
//! computed by a radial sweep in O(n log n) and spread over all cores (Rayon).
//! The observer is snapped to the centre of its cell, and the eye is that
//! cell's height plus `observer_height`. Every closer cell is reduced to three
//! samples: its centre, and the two corners that bound it as seen from the
//! observer, with corner heights averaged from the four cells meeting there.
//! A target cell (centre + `target_height`) is hidden when a strictly closer
//! cell's gradient, interpolated in angle at the target's direction, is
//! strictly steeper. The result matches GRASS 8.4 cell for cell.
//!
//! [`Surface::Bilinear`] is exact line of sight on the bilinear surface between
//! cell centres, observer at its exact point: every float segment certified by
//! rigorous rounding-error bounds, ties settled in exact rational arithmetic.
//!
//! No-data cells (NaN) never block a view and are never visible.
//!
//! ```
//! use viewshed::{viewshed, Grid, Options};
//!
//! // 5 x 5 flat raster, 1 m cells, top-left corner at (0, 5)
//! let z = vec![10.0; 25];
//! let grid = Grid::new(&z, 5, 5, [0.0, 1.0, 0.0, 5.0, 0.0, -1.0]).unwrap();
//! let seen = viewshed(&grid, 2.5, 2.5, &Options::default()).unwrap();
//! assert!(seen.iter().all(|&v| v));
//! ```

use std::fmt;

mod bilinear;
mod los;
mod shadow;
mod sunpos;
mod sweep;
mod ztv;

pub use los::{intervisible, LosOptions, Point};
pub use shadow::{shadow, ShadowOptions, Sun};
#[doc(hidden)]
pub use shadow::shadow_vec;
pub use sunpos::{sun_hours, sun_hours_between, sun_position, sun_samples, SunPosition};
pub use ztv::{ztv, ZtvMethod, ZtvOptions, ZtvTarget};

/// Something went wrong before any visibility was computed.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The height array length doesn't equal rows × cols.
    ShapeMismatch { len: usize, rows: usize, cols: usize },
    /// Rotated rasters (non-zero rotation terms) aren't supported.
    Rotated,
    /// Cells must be square and the cell size non-zero.
    NonSquareCells,
    /// The observer is outside the raster.
    ObserverOutside,
    /// There is no height data under the observer.
    ObserverOnNoData,
    /// The area searched is over 1,000,000 cells across (set `max_distance`).
    TooLarge,
    /// The sun's elevation isn't in (0, 90) degrees or its azimuth isn't finite (or, for a
    /// direction vector, it has no horizontal part or points down).
    InvalidSun,
    /// A target height below zero (or NaN).
    InvalidTargetHeight,
    /// Refraction above 1 with curvature on (the earth would bulge upwards): not supported
    /// for shadows.
    UnsupportedRefraction,
    /// A latitude outside [-90, 90], or a non-finite longitude or grid convergence.
    InvalidLocation,
    /// A non-finite time, an end before the start, or a time step that isn't a positive number.
    InvalidTime,
    /// A sun-hours weight that isn't a finite number >= 0.
    InvalidWeight,
    /// A point of pair `pair` is outside the raster (or not finite).
    PointOutside { pair: usize },
    /// There is no height data under a point of pair `pair`.
    PointOnNoData { pair: usize },
    /// A point of pair `pair` has a height below zero (or not finite).
    InvalidPointHeight { pair: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch { len, rows, cols } => {
                write!(f, "height array has {len} values, expected {rows} x {cols}")
            }
            Error::Rotated => write!(f, "rotated rasters are not supported"),
            Error::NonSquareCells => write!(f, "cells must be square with a non-zero size"),
            Error::ObserverOutside => write!(f, "the observer is outside the raster"),
            Error::ObserverOnNoData => write!(f, "there is no height data under the observer"),
            Error::TooLarge => write!(f, "the area searched is over 1,000,000 cells across; set max_distance"),
            Error::InvalidSun => write!(f, "the sun's elevation must be in (0, 90) degrees and its azimuth finite"),
            Error::InvalidLocation => write!(f, "latitude must be in [-90, 90] degrees and longitude and grid convergence finite"),
            Error::InvalidTime => write!(f, "times must be finite, the end not before the start, and the time step positive"),
            Error::InvalidWeight => write!(f, "weights must be finite and >= 0"),
            Error::InvalidTargetHeight => write!(f, "target_height must be >= 0"),
            Error::UnsupportedRefraction => write!(f, "refraction above 1 (an earth bulging upwards) is not supported for shadows"),
            Error::PointOutside { pair } => write!(f, "pair {pair}: a point is outside the raster"),
            Error::PointOnNoData { pair } => write!(f, "pair {pair}: there is no height data under a point"),
            Error::InvalidPointHeight { pair } => write!(f, "pair {pair}: heights must be finite and >= 0"),
        }
    }
}

impl std::error::Error for Error {}

/// A raster of surface heights in row-major order (first row = top).
#[derive(Debug, Clone, Copy)]
pub struct Grid<'a> {
    z: &'a [f64],
    rows: usize,
    cols: usize,
    /// GDAL-style geotransform: (x of left edge, cell width, 0, y of top edge, 0, -cell height).
    transform: [f64; 6],
}

impl<'a> Grid<'a> {
    pub fn new(z: &'a [f64], rows: usize, cols: usize, transform: [f64; 6]) -> Result<Self, Error> {
        if z.len() != rows * cols {
            return Err(Error::ShapeMismatch { len: z.len(), rows, cols });
        }
        if transform[2] != 0.0 || transform[4] != 0.0 {
            return Err(Error::Rotated);
        }
        let (w, h) = (transform[1].abs(), transform[5].abs());
        if w == 0.0 || (w - h).abs() > 1e-9 * w {
            return Err(Error::NonSquareCells);
        }
        Ok(Grid { z, rows, cols, transform })
    }

    fn res(&self) -> f64 {
        self.transform[1].abs()
    }

    fn at(&self, r: usize, c: usize) -> f64 {
        self.z[r * self.cols + c]
    }

    /// Fractional (column, row) of a map position, in cell-centre units.
    fn frac(&self, x: f64, y: f64) -> (f64, f64) {
        (
            (x - self.transform[0]) / self.transform[1] - 0.5,
            (y - self.transform[3]) / self.transform[5] - 0.5,
        )
    }

    fn inside(&self, x: f64, y: f64) -> bool {
        let (fc, fr) = self.frac(x, y);
        fc >= -0.5 && fr >= -0.5 && fc <= self.cols as f64 - 0.5 && fr <= self.rows as f64 - 0.5
    }

    /// Bilinear surface height; NaN outside the raster or next to no-data.
    pub fn height(&self, x: f64, y: f64) -> f64 {
        if !self.inside(x, y) {
            return f64::NAN;
        }
        let (fc, fr) = self.frac(x, y);
        let (c0, r0) = (fc.floor(), fr.floor());
        let (tc, tr) = (fc - c0, fr - r0);
        let clamp = |v: f64, n: usize| (v.max(0.0) as usize).min(n - 1);
        let (ca, cb) = (clamp(c0, self.cols), clamp(c0 + 1.0, self.cols));
        let (ra, rb) = (clamp(r0, self.rows), clamp(r0 + 1.0, self.rows));
        let top = self.at(ra, ca) * (1.0 - tc) + self.at(ra, cb) * tc;
        let bottom = self.at(rb, ca) * (1.0 - tc) + self.at(rb, cb) * tc;
        top * (1.0 - tr) + bottom * tr
    }

    /// Map position of the centre of cell (row, col).
    pub fn centre(&self, r: usize, c: usize) -> (f64, f64) {
        (
            self.transform[0] + (c as f64 + 0.5) * self.transform[1],
            self.transform[3] + (r as f64 + 0.5) * self.transform[5],
        )
    }
}

/// Which terrain model decides visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Surface {
    /// GRASS `r.viewshed`'s model (the default): each closer cell is reduced to
    /// three gradient samples (its centre and the two corners bounding it as seen
    /// from the observer), interpolated in angle. The observer is snapped to the
    /// centre of its cell. Computed by a radial sweep in O(n log n).
    #[default]
    RViewshed,
    /// Bilinear surface between cell centres, observer at its exact point. Exact:
    /// each sightline is split at the patch boundaries, every segment's minimum gap
    /// is certified from rigorous rounding-error bounds, and anything undecided is
    /// settled in exact rational arithmetic. `max_distance` is compared exactly too.
    Bilinear,
}

/// Viewshed settings. Heights and distances are in the raster's units (metres
/// when `curvature` is on).
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub observer_height: f64,
    pub target_height: f64,
    /// Horizontal distance limit; cells beyond it are not visible.
    pub max_distance: Option<f64>,
    pub surface: Surface,
    /// Lower every point by the earth's curvature as seen from the observer:
    /// `d^2 / (2 R) * (1 - refraction)` at horizontal distance `d`, the usual
    /// surveying correction (as in GRASS `r.viewshed -c -r` and GDAL's `-cc`).
    pub curvature: bool,
    /// Refraction coefficient: light bends with the earth, which raises
    /// distant points by this fraction of the curvature drop. 0.13 is a common
    /// value for visible light; GRASS defaults to 1/7 (0.14286).
    pub refraction: f64,
    /// Earth radius in metres. The default is the IUGG mean radius; GRASS uses
    /// the semi-major axis of its location's ellipsoid.
    pub earth_radius: f64,
}

/// The IUGG mean earth radius, metres.
pub const MEAN_EARTH_RADIUS: f64 = 6_371_008.8;

impl Default for Options {
    fn default() -> Self {
        Options {
            observer_height: 1.6,
            target_height: 0.0,
            max_distance: None,
            surface: Surface::default(),
            curvature: true,
            refraction: 0.13,
            earth_radius: MEAN_EARTH_RADIUS,
        }
    }
}

impl Options {
    /// The curvature drop at horizontal distance d is `d * d * drop_factor()`.
    pub(crate) fn drop_factor(&self) -> f64 {
        if self.curvature {
            (1.0 - self.refraction) / (2.0 * self.earth_radius)
        } else {
            0.0
        }
    }
}

const STEPS_PER_CELL: f64 = 4.0;
const EPS: f64 = 1e-9;

/// True if nothing on the surface rises above the line from `eye` to `target` (flat earth).
///
/// Sampled (4 points per cell, with a tolerance), so it can miss a thin blocker or
/// report a grazing one; [`intervisible`] is exact, takes heights above the ground,
/// handles curvature and many pairs at once.
#[deprecated(since = "0.4.0", note = "sampled, not exact: use `intervisible`")]
pub fn line_of_sight(grid: &Grid, eye: [f64; 3], target: [f64; 3]) -> bool {
    line_of_sight_curved(grid, eye, target, 0.0)
}

/// As `line_of_sight`, with the surface lowered by `d * d * drop` at horizontal
/// distance d from the eye (`target[2]` must already include its own drop).
fn line_of_sight_curved(grid: &Grid, eye: [f64; 3], target: [f64; 3], drop: f64) -> bool {
    let (dx, dy) = (target[0] - eye[0], target[1] - eye[1]);
    let horiz = (dx * dx + dy * dy).sqrt();
    let n = (horiz / grid.res() * STEPS_PER_CELL).ceil() as usize;
    for i in 1..n {
        let t = i as f64 / n as f64;
        let line_z = eye[2] + t * (target[2] - eye[2]);
        let d = t * horiz;
        let surface = grid.height(eye[0] + t * dx, eye[1] + t * dy) - d * d * drop;
        if surface > line_z + EPS * (1.0 + line_z.abs()) {
            return false; // NaN surface compares false: no-data never blocks
        }
    }
    true
}

/// Visibility of every cell from the observer at map position (x, y).
///
/// Returns one `bool` per cell, row-major, same shape as the grid.
pub fn viewshed(grid: &Grid, x: f64, y: f64, opts: &Options) -> Result<Vec<bool>, Error> {
    match opts.surface {
        Surface::RViewshed => sweep::viewshed(grid, x, y, opts),
        Surface::Bilinear => bilinear::viewshed(grid, x, y, opts),
    }
}

/// Reverse viewshed: from which cells can the target at map position (x, y) be seen?
///
/// `opts.target_height` is the target's height above the surface at (x, y) (a mast
/// top, a roof); `opts.observer_height` is the eye height above every cell. Cell
/// (r, c) is `true` when an observer standing there sees the target: the same
/// answer as running [`viewshed`] from that cell and reading the target's cell, for
/// every cell, but in one call.
///
/// - [`Surface::Bilinear`]: a sightline is a straight segment, the blocking rule is
///   the same in both directions, and the curvature drop (`d_a * d_b` times a
///   constant below the chord, for distances `d_a`, `d_b` to the two ends) is
///   symmetric too, so this is one exact viewshed from the target with the two
///   heights swapped: O(n log n). The target stays at its exact point; observers
///   stand at cell centres.
/// - [`Surface::RViewshed`]: that model is not symmetric (only cells nearer the
///   observer block), so every cell is an observer looking at the target's cell
///   centre, as [`ztv`] does with [`ZtvMethod::Forward`]: exact for the model, at
///   most one line of sight per cell.
///
/// No-data cells are `false`. Errors as for [`viewshed`], with the target in the
/// observer's place (outside the raster, or no height data under it).
pub fn reverse_viewshed(grid: &Grid, x: f64, y: f64, opts: &Options) -> Result<Vec<bool>, Error> {
    match opts.surface {
        Surface::Bilinear => {
            let swapped = Options {
                observer_height: opts.target_height,
                target_height: opts.observer_height,
                ..opts.clone()
            };
            bilinear::viewshed(grid, x, y, &swapped)
        }
        Surface::RViewshed => {
            let zo = ZtvOptions {
                observer_height: opts.observer_height,
                max_distance: opts.max_distance,
                method: ZtvMethod::Forward,
                curvature: opts.curvature,
                refraction: opts.refraction,
                earth_radius: opts.earth_radius,
            };
            let counts = ztv(grid, &[ZtvTarget { x, y, height: opts.target_height }], &zo)?;
            Ok(counts.into_iter().map(|n| n > 0).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GT: [f64; 6] = [0.0, 1.0, 0.0, 41.0, 0.0, -1.0]; // 41 x 41, 1 m cells

    fn flat(h: f64) -> Vec<f64> {
        vec![h; 41 * 41]
    }

    #[test]
    fn flat_ground_is_all_visible() {
        let z = flat(50.0);
        let g = Grid::new(&z, 41, 41, GT).unwrap();
        assert!(viewshed(&g, 20.5, 20.5, &Options::default()).unwrap().iter().all(|&v| v));
    }

    #[test]
    fn wall_hides_ground_behind_it() {
        let mut z = flat(0.0);
        for c in 0..41 {
            z[15 * 41 + c] = 5.0; // a 5 m wall along row 15, north of the observer
        }
        let g = Grid::new(&z, 41, 41, GT).unwrap();
        let seen = viewshed(&g, 20.5, 20.5, &Options::default()).unwrap();
        assert!(seen[16 * 41 + 20]); // just in front of the wall (the wall's south side)
        assert!(seen[15 * 41 + 20]); // the wall top
        assert!(!seen[5 * 41 + 20]); // well behind it
        assert!(seen[30 * 41 + 20]); // the other way
    }

    #[test]
    fn taller_observer_sees_over_the_wall() {
        let mut z = flat(0.0);
        for c in 0..41 {
            z[15 * 41 + c] = 5.0;
        }
        let g = Grid::new(&z, 41, 41, GT).unwrap();
        let opts = Options { observer_height: 50.0, ..Options::default() };
        assert!(viewshed(&g, 20.5, 20.5, &opts).unwrap()[5 * 41 + 20]);
    }

    #[test]
    fn max_distance_and_nodata() {
        let mut z = flat(0.0);
        z[0] = f64::NAN;
        let g = Grid::new(&z, 41, 41, GT).unwrap();
        let opts = Options { max_distance: Some(10.0), ..Options::default() };
        let seen = viewshed(&g, 20.5, 20.5, &opts).unwrap();
        assert!(seen[20 * 41 + 25]);
        assert!(!seen[20 * 41 + 35]);
        assert!(!seen[0]);
    }

    #[test]
    fn curvature_sets_the_horizon_on_flat_ground() {
        // horizon at sqrt(h / f) = 10.5 cells, f = (1 - k) / (2 R)
        let z = flat(0.0);
        let g = Grid::new(&z, 41, 41, GT).unwrap();
        let (h, k) = (1.6, 0.13);
        let r = (1.0 - k) * 10.5 * 10.5 / (2.0 * h);
        for surface in [Surface::RViewshed, Surface::Bilinear] {
            let opts = Options { observer_height: h, refraction: k, earth_radius: r, surface, ..Options::default() };
            let seen = viewshed(&g, 20.5, 20.5, &opts).unwrap();
            assert!(seen[20 * 41 + 20 + 9], "{surface:?}: within the horizon");
            assert!(!seen[20 * 41 + 20 + 12], "{surface:?}: beyond the horizon");
            let flat_earth = Options { curvature: false, ..opts };
            assert!(viewshed(&g, 20.5, 20.5, &flat_earth).unwrap().iter().all(|&v| v));
        }
    }

    #[test]
    fn bad_input_is_an_error() {
        let z = flat(0.0);
        assert_eq!(Grid::new(&z, 40, 41, GT).unwrap_err(), Error::ShapeMismatch { len: 1681, rows: 40, cols: 41 });
        assert_eq!(Grid::new(&z, 41, 41, [0.0, 1.0, 0.1, 41.0, 0.0, -1.0]).unwrap_err(), Error::Rotated);
        let g = Grid::new(&z, 41, 41, GT).unwrap();
        assert_eq!(viewshed(&g, 100.0, 0.0, &Options::default()).unwrap_err(), Error::ObserverOutside);
    }

    #[test]
    fn reverse_viewshed_equals_a_viewshed_from_every_cell() {
        // rough terrain with no-data holes; a small earth so curvature matters
        let n = 25;
        let gt = [0.0, 10.0, 0.0, 250.0, 0.0, -10.0];
        let mut z: Vec<f64> = (0..n * n)
            .map(|i| {
                let (r, c) = ((i / n) as f64, (i % n) as f64);
                20.0 * (r * 0.7).sin() * (c * 0.45).cos() + 7.0 * ((r * c) * 0.13).sin()
            })
            .collect();
        for &(r, c) in &[(3, 4), (17, 9), (20, 20), (8, 18)] {
            z[r * n + c] = f64::NAN;
        }
        let g = Grid::new(&z, n, n, gt).unwrap();
        let (tr, tc) = (12, 6);
        let (tx, ty) = (tc as f64 * 10.0 + 5.0, 250.0 - tr as f64 * 10.0 - 5.0);
        for surface in [Surface::Bilinear, Surface::RViewshed] {
            for (curvature, max_distance) in [(false, None), (true, Some(170.0))] {
                let opts = Options {
                    observer_height: 1.6,
                    target_height: 12.0,
                    surface,
                    curvature,
                    earth_radius: 20_000.0,
                    max_distance,
                    ..Options::default()
                };
                let rev = reverse_viewshed(&g, tx, ty, &opts).unwrap();
                let fwd_opts = Options { observer_height: 1.6, target_height: 12.0, ..opts.clone() };
                let (mut seen, mut checked) = (0, 0);
                for r in 0..n {
                    for c in 0..n {
                        let (x, y) = (c as f64 * 10.0 + 5.0, 250.0 - r as f64 * 10.0 - 5.0);
                        let want = match viewshed(&g, x, y, &fwd_opts) {
                            Ok(v) => v[tr * n + tc],
                            Err(Error::ObserverOnNoData) => {
                                // only a no-data cell has no ground under its centre
                                assert!(z[r * n + c].is_nan() && !rev[r * n + c]);
                                continue;
                            }
                            Err(e) => panic!("{e}"),
                        };
                        assert_eq!(rev[r * n + c], want, "{surface:?} curvature {curvature} cell ({r}, {c})");
                        checked += 1;
                        seen += want as usize;
                    }
                }
                assert!(checked > 500 && seen > 50 && seen < checked, "{surface:?}: {seen} of {checked}");
            }
        }
    }
}
