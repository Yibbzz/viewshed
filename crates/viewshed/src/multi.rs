//! Many observers: cumulative viewsheds and a visibility index.
//!
//! Both are sums of ordinary viewsheds ([`crate::viewshed`]), so they inherit its surface
//! model and its exactness: a count is exactly the number of observers whose viewshed
//! contains the cell. Observers are spread over the cores (each viewshed is parallel too;
//! Rayon balances the two levels).

use rayon::prelude::*;

use crate::{viewshed, Error, Grid, Options, Point};

/// For every cell, how many of `observers` see it. Each observer is a map position and
/// its eye height above the surface (`opts.observer_height` is not used); the other
/// settings (`target_height`, `max_distance`, `surface`, curvature) come from `opts`.
/// Row-major, the grid's shape; no-data cells are 0. A bad observer gives
/// [`Error::Observer`] with its index.
pub fn cumulative(grid: &Grid, observers: &[Point], opts: &Options) -> Result<Vec<u32>, Error> {
    let n = grid.rows * grid.cols;
    observers
        .par_iter()
        .enumerate()
        .try_fold(
            || vec![0u32; n],
            |mut acc, (i, p)| {
                let o = Options { observer_height: p.height, ..opts.clone() };
                let seen = viewshed(grid, p.x, p.y, &o).map_err(|e| Error::Observer { index: i, error: Box::new(e) })?;
                acc.iter_mut().zip(seen).for_each(|(a, v)| *a += v as u32);
                Ok(acc)
            },
        )
        .try_reduce(
            || vec![0u32; n],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                Ok(a)
            },
        )
}

/// Visibility index: every `stride`-th cell with data (rows and columns 0, stride, 2 stride,
/// ...) is an observer at its cell centre, `opts.observer_height` above the surface; for
/// every cell, the fraction of those observers that see it (0 to 1; NaN on no-data). With
/// `stride` 1 every cell is an observer (the "total viewshed"); use `max_distance` and a
/// stride to keep it affordable: the cost is one viewshed per observer. Returns the
/// fractions and the number of observers.
pub fn visibility_index(grid: &Grid, stride: usize, opts: &Options) -> Result<(Vec<f64>, usize), Error> {
    if stride == 0 {
        return Err(Error::InvalidStride);
    }
    let t = grid.transform;
    let mut observers = Vec::new();
    for r in (0..grid.rows).step_by(stride) {
        for c in (0..grid.cols).step_by(stride) {
            if grid.at(r, c).is_nan() {
                continue;
            }
            let x = t[0] + (c as f64 + 0.5) * t[1];
            let y = t[3] + (r as f64 + 0.5) * t[5];
            observers.push(Point { x, y, height: opts.observer_height });
        }
    }
    let counts = cumulative(grid, &observers, opts)?;
    let k = observers.len();
    let frac = counts
        .iter()
        .zip(grid.z.iter())
        .map(|(&n, z)| if z.is_nan() { f64::NAN } else if k == 0 { 0.0 } else { n as f64 / k as f64 })
        .collect();
    Ok((frac, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Surface;

    fn terrain() -> (Vec<f64>, usize) {
        let n = 25;
        let mut z: Vec<f64> = (0..n * n)
            .map(|i| {
                let (r, c) = ((i / n) as f64, (i % n) as f64);
                (6.0 * (r * 0.5).sin() * (c * 0.35).cos() + 2.0 * ((r + c) * 0.8).sin()).round()
            })
            .collect();
        z[7 * n + 11] = f64::NAN;
        (z, n)
    }

    #[test]
    fn cumulative_is_the_sum_of_viewsheds() {
        let (z, n) = terrain();
        let g = Grid::new(&z, n, n, [0.0, 2.0, 0.0, 50.0, 0.0, -2.0]).unwrap();
        let obs = [
            Point { x: 5.0, y: 45.0, height: 1.6 },
            Point { x: 30.3, y: 20.9, height: 10.0 },
            Point { x: 49.0, y: 1.0, height: 0.0 },
            Point { x: 25.0, y: 25.0, height: 3.0 },
        ];
        for surface in [Surface::RViewshed, Surface::Bilinear] {
            let opts = Options { target_height: 0.5, max_distance: Some(40.0), surface, ..Options::default() };
            let got = cumulative(&g, &obs, &opts).unwrap();
            let mut want = vec![0u32; n * n];
            for p in &obs {
                let o = Options { observer_height: p.height, ..opts.clone() };
                for (w, v) in want.iter_mut().zip(viewshed(&g, p.x, p.y, &o).unwrap()) {
                    *w += v as u32;
                }
            }
            assert_eq!(got, want, "{surface:?}");
            assert!(got.contains(&4) && got.iter().any(|&k| k < 4));
        }
    }

    #[test]
    fn visibility_index_samples_every_stride_th_cell() {
        let (z, n) = terrain();
        let g = Grid::new(&z, n, n, [0.0, 2.0, 0.0, 50.0, 0.0, -2.0]).unwrap();
        let opts = Options { surface: Surface::Bilinear, ..Options::default() };
        let (vi, k) = visibility_index(&g, 4, &opts).unwrap();
        assert_eq!(k, 7 * 7); // rows/cols 0, 4, ..., 24; (7, 11) isn't sampled
        let mut obs = Vec::new();
        for r in (0..n).step_by(4) {
            for c in (0..n).step_by(4) {
                obs.push(Point { x: c as f64 * 2.0 + 1.0, y: 50.0 - r as f64 * 2.0 - 1.0, height: 1.6 });
            }
        }
        let counts = cumulative(&g, &obs, &opts).unwrap();
        for i in 0..n * n {
            if z[i].is_nan() {
                assert!(vi[i].is_nan());
            } else {
                assert_eq!(vi[i], counts[i] as f64 / k as f64);
            }
        }
        assert_eq!(visibility_index(&g, 0, &opts).unwrap_err(), Error::InvalidStride);
    }

    #[test]
    fn a_bad_observer_is_named() {
        let (z, n) = terrain();
        let g = Grid::new(&z, n, n, [0.0, 2.0, 0.0, 50.0, 0.0, -2.0]).unwrap();
        let obs = [Point { x: 5.0, y: 45.0, height: 1.6 }, Point { x: 500.0, y: 45.0, height: 1.6 }];
        let e = cumulative(&g, &obs, &Options::default()).unwrap_err();
        assert_eq!(e, Error::Observer { index: 1, error: Box::new(Error::ObserverOutside) });
        assert_eq!(e.to_string(), "observer 1: the observer is outside the raster");
    }
}
