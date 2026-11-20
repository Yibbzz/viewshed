//! Zones of theoretical visibility (ZTV): from how many targets, such as
//! wind-turbine blade tips, each cell's observer can see.
//!
//! Two methods:
//!
//! - [`ZtvMethod::Forward`] is the definition: every cell is an observer
//!   (eye = cell height + `observer_height`) looking at each target (its cell
//!   centre + the target's height), evaluated with the same r.viewshed model
//!   as [`crate::viewshed`]. The answer for a cell equals running a full
//!   viewshed from that cell and reading the target's cell. Each observer
//!   walks only the cells its sightline crosses, so the cost is about one
//!   line of sight per cell and target: O(n · d).
//! - [`ZtvMethod::Reciprocal`] is the usual shortcut: one viewshed per target,
//!   *from* the target's height down to `observer_height` above every cell.
//!   A sightline is the same both ways, but the r.viewshed model isn't
//!   symmetric (only cells nearer the observer block, and samples are
//!   interpolated around the observer), so a few cells near shadow edges can
//!   differ from the forward answer. O(n log n) per target.

use rayon::prelude::*;

use crate::sweep::{cell_at, corners, margin, Target};
use crate::{Error, Grid, Options, Surface};

/// A point to be seen, such as a blade tip: map position and height above the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZtvTarget {
    pub x: f64,
    pub y: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZtvMethod {
    /// Every cell as an observer (the definition). Slower.
    #[default]
    Forward,
    /// One viewshed from each target (the common shortcut). Faster, not exactly the same.
    Reciprocal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZtvOptions {
    /// Eye height above the surface at every cell.
    pub observer_height: f64,
    /// Targets farther than this from a cell don't count for it.
    pub max_distance: Option<f64>,
    pub method: ZtvMethod,
}

impl Default for ZtvOptions {
    fn default() -> Self {
        ZtvOptions { observer_height: 2.0, max_distance: None, method: ZtvMethod::default() }
    }
}

/// For every cell, how many of `targets` are visible from it. Row-major, the
/// grid's shape. No-data cells get 0.
pub fn ztv(grid: &Grid, targets: &[ZtvTarget], opts: &ZtvOptions) -> Result<Vec<u32>, Error> {
    let t = &grid.transform;
    let mut cells = Vec::with_capacity(targets.len());
    for tg in targets {
        let c = ((tg.x - t[0]) / t[1]).floor() as i64;
        let r = ((tg.y - t[3]) / t[5]).floor() as i64;
        if c < 0 || r < 0 || c >= grid.cols as i64 || r >= grid.rows as i64 {
            return Err(Error::ObserverOutside);
        }
        if grid.at(r as usize, c as usize).is_nan() {
            return Err(Error::ObserverOnNoData);
        }
        cells.push((r, c, tg.height));
    }
    match opts.method {
        ZtvMethod::Reciprocal => {
            let mut counts = vec![0u32; grid.rows * grid.cols];
            for tg in targets {
                let o = Options {
                    observer_height: tg.height,
                    target_height: opts.observer_height,
                    max_distance: opts.max_distance,
                    surface: Surface::RViewshed,
                };
                let seen = crate::viewshed(grid, tg.x, tg.y, &o)?;
                counts.iter_mut().zip(seen).for_each(|(n, v)| *n += v as u32);
            }
            Ok(counts)
        }
        ZtvMethod::Forward => {
            let cols = grid.cols;
            let mut counts = vec![0u32; grid.rows * cols];
            counts.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
                for (c, n) in row.iter_mut().enumerate() {
                    let z = grid.at(r, c);
                    if z.is_nan() {
                        continue;
                    }
                    let eye = z + opts.observer_height;
                    for &(rt, ct, h) in &cells {
                        if sees(grid, r as i64, c as i64, eye, rt, ct, h, opts.max_distance) {
                            *n += 1;
                        }
                    }
                }
            });
            Ok(counts)
        }
    }
}

fn cross(a: (i64, i64), b: (i64, i64)) -> i128 {
    a.0 as i128 * b.1 as i128 - a.1 as i128 * b.0 as i128
}

/// Can the eye over cell (rv, cv) see `height` above the centre of cell (rt, ct)?
/// Exactly the r.viewshed-model answer a full sweep from (rv, cv) would give.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sees(grid: &Grid, rv: i64, cv: i64, eye: f64, rt: i64, ct: i64, height: f64, max_d: Option<f64>) -> bool {
    let (dr, dc) = (rt - rv, ct - cv);
    if dr == 0 && dc == 0 {
        return true;
    }
    let zt = grid.at(rt as usize, ct as usize);
    let key = dr * dr + dc * dc;
    let dist = (key as f64).sqrt() * grid.res();
    if zt.is_nan() || max_d.map_or(false, |m| dist > m) {
        return false;
    }
    let mut target = Target {
        dir: (2 * dc, -2 * dr),
        key,
        ring: 0,
        slope: (zt + height - eye) / dist,
        theta: None,
        grad: 0.0,
    };
    let m = margin(target.slope);
    let (skip_below, sure_above) = (target.slope - m, target.slope + m);
    let (rows, cols) = (grid.rows as i64, grid.cols as i64);

    // Step along the longer axis; in each step the sightline stays within one
    // cell of the minor axis, so 4 candidates per step cover every cell it
    // crosses. Each candidate is then tested exactly: strictly closer than the
    // target, and the target direction strictly inside its corner span.
    let (n, x_major) = if dc.abs() >= dr.abs() { (dc.abs(), true) } else { (dr.abs(), false) };
    for i in 0..=n {
        let (major, minor_end) = if x_major { (i * dc.signum(), dr) } else { (i * dr.signum(), dc) };
        let minor = major as f64 * minor_end as f64 / if x_major { dc as f64 } else { dr as f64 };
        let k0 = minor.floor() as i64;
        for k in k0 - 1..=k0 + 2 {
            let (br, bc) = if x_major { (k, major) } else { (major, k) };
            if (br == 0 && bc == 0) || br * br + bc * bc >= key {
                continue;
            }
            let (r, c) = (rv + br, cv + bc);
            if r < 0 || c < 0 || r >= rows || c >= cols {
                continue;
            }
            let [enter, exit] = corners(br, bc);
            let e = (2 * bc + enter.0, -(2 * br + enter.1));
            let x = (2 * bc + exit.0, -(2 * br + exit.1));
            if cross(e, target.dir) <= 0 || cross(target.dir, x) <= 0 {
                continue; // not active in the target's direction
            }
            let cell = cell_at(grid, rv, cv, eye, r, c);
            if cell.ring == u32::MAX || cell.hi() < skip_below {
                continue;
            }
            if cell.lo() > sure_above || target.blocked_by(&cell) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewshed;

    /// Small deterministic pseudo-random terrain, optionally with integer heights (many ties).
    fn terrain(n: usize, seed: u64, integer: bool, holes: bool) -> Vec<f64> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut z = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                let base = if r > 0 { z[(r - 1) * n + c] } else { 0.0 };
                let mut v = base + (next() - 0.5) * 6.0;
                if integer {
                    v = v.round();
                }
                z[r * n + c] = v;
            }
        }
        if holes {
            for _ in 0..n {
                let i = (next() * (n * n) as f64) as usize;
                z[i] = f64::NAN;
            }
        }
        z
    }

    #[test]
    fn forward_equals_a_full_viewshed_from_every_cell() {
        for (seed, integer, holes, n) in [(1, false, false, 21), (2, true, false, 19), (3, false, true, 17), (4, true, true, 23)] {
            let z = terrain(n, seed, integer, holes);
            let gt = [0.0, 1.0, 0.0, n as f64, 0.0, -1.0];
            let grid = Grid::new(&z, n, n, gt).unwrap();
            let mut targets = Vec::new();
            for (r, c, h) in [(3usize, 4usize, 5.0), (n / 2, n / 2, 0.0), (n - 2, 1, 12.5)] {
                if !z[r * n + c].is_nan() {
                    targets.push((r, c, ZtvTarget { x: c as f64 + 0.5, y: (n - r) as f64 - 0.5, height: h }));
                }
            }
            let opts = ZtvOptions { observer_height: 1.5, ..ZtvOptions::default() };
            let counts = ztv(&grid, &targets.iter().map(|t| t.2).collect::<Vec<_>>(), &opts).unwrap();
            for r in 0..n {
                for c in 0..n {
                    if z[r * n + c].is_nan() {
                        assert_eq!(counts[r * n + c], 0);
                        continue;
                    }
                    let o = Options { observer_height: 1.5, ..Options::default() };
                    let mut expect = 0;
                    for &(tr, tc, t) in &targets {
                        let v = viewshed(&grid, c as f64 + 0.5, (n - r) as f64 - 0.5, &Options { target_height: t.height, ..o.clone() }).unwrap();
                        expect += v[tr * n + tc] as u32;
                    }
                    assert_eq!(counts[r * n + c], expect, "seed {seed}, observer ({r}, {c})");
                }
            }
        }
    }

    #[test]
    fn reciprocal_is_close_to_forward() {
        let n = 41;
        let z = terrain(n, 9, false, false);
        let grid = Grid::new(&z, n, n, [0.0, 1.0, 0.0, n as f64, 0.0, -1.0]).unwrap();
        let targets = [ZtvTarget { x: 20.5, y: 20.5, height: 30.0 }];
        let f = ztv(&grid, &targets, &ZtvOptions::default()).unwrap();
        let r = ztv(&grid, &targets, &ZtvOptions { method: ZtvMethod::Reciprocal, ..ZtvOptions::default() }).unwrap();
        let differ = f.iter().zip(&r).filter(|(a, b)| a != b).count();
        assert!(differ * 20 < n * n, "{differ} of {} cells differ", n * n);
    }
}
