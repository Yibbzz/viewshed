//! Fast, exact line-of-sight viewsheds.
//!
//! **Version 0.0.x is the reference implementation:** a straightforward,
//! correct line-of-sight check from the observer to every cell. It is slow
//! (each cell is checked separately) and meant for small rasters and for
//! testing. The fast sweep-line algorithm will keep the same API and results.
//!
//! # Surface model
//!
//! Heights are given at cell centres and the surface between them is
//! bilinearly interpolated (heights are clamped to the edge cells at the
//! raster border). The observer's eye is `observer_height` above the surface
//! at the observer point. A cell is visible when the straight line from the
//! eye to the point `target_height` above the cell centre stays on or above
//! the surface. The line is sampled every quarter cell along the way. No-data
//! cells (NaN) never block a view and are never visible.
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

/// Something went wrong before any visibility was computed.
#[derive(Debug, Clone, PartialEq)]
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

/// Viewshed settings. Heights and distances are in the raster's units.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub observer_height: f64,
    pub target_height: f64,
    /// Horizontal distance limit; cells beyond it are not visible.
    pub max_distance: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Options { observer_height: 1.6, target_height: 0.0, max_distance: None }
    }
}

const STEPS_PER_CELL: f64 = 4.0;
const EPS: f64 = 1e-9;

/// True if nothing on the surface rises above the line from `eye` to `target`.
pub fn line_of_sight(grid: &Grid, eye: [f64; 3], target: [f64; 3]) -> bool {
    let (dx, dy) = (target[0] - eye[0], target[1] - eye[1]);
    let horiz = (dx * dx + dy * dy).sqrt();
    let n = (horiz / grid.res() * STEPS_PER_CELL).ceil() as usize;
    for i in 1..n {
        let t = i as f64 / n as f64;
        let line_z = eye[2] + t * (target[2] - eye[2]);
        let surface = grid.height(eye[0] + t * dx, eye[1] + t * dy);
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
    if !grid.inside(x, y) {
        return Err(Error::ObserverOutside);
    }
    let ground = grid.height(x, y);
    if ground.is_nan() {
        return Err(Error::ObserverOnNoData);
    }
    let eye = [x, y, ground + opts.observer_height];
    let mut out = vec![false; grid.rows * grid.cols];
    for r in 0..grid.rows {
        for c in 0..grid.cols {
            let z = grid.at(r, c);
            if z.is_nan() {
                continue;
            }
            let (cx, cy) = grid.centre(r, c);
            if let Some(max) = opts.max_distance {
                if ((cx - x).powi(2) + (cy - y).powi(2)).sqrt() > max {
                    continue;
                }
            }
            out[r * grid.cols + c] = line_of_sight(grid, eye, [cx, cy, z + opts.target_height]);
        }
    }
    Ok(out)
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
    fn bad_input_is_an_error() {
        let z = flat(0.0);
        assert_eq!(Grid::new(&z, 40, 41, GT).unwrap_err(), Error::ShapeMismatch { len: 1681, rows: 40, cols: 41 });
        assert_eq!(Grid::new(&z, 41, 41, [0.0, 1.0, 0.1, 41.0, 0.0, -1.0]).unwrap_err(), Error::Rotated);
        let g = Grid::new(&z, 41, 41, GT).unwrap();
        assert_eq!(viewshed(&g, 100.0, 0.0, &Options::default()).unwrap_err(), Error::ObserverOutside);
    }
}
