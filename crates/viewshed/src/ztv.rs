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
//!   walks only the cells its sightline crosses, and skips whole stretches
//!   whose highest point (from a max-height pyramid) provably can't reach
//!   the sightline, so the cost is at most one line of sight per cell and
//!   target, O(n · d), and much less over open ground.
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
    /// Earth curvature and refraction, as in [`Options`].
    pub curvature: bool,
    pub refraction: f64,
    pub earth_radius: f64,
}

impl Default for ZtvOptions {
    fn default() -> Self {
        let o = Options::default();
        ZtvOptions {
            observer_height: 2.0,
            max_distance: None,
            method: ZtvMethod::default(),
            curvature: o.curvature,
            refraction: o.refraction,
            earth_radius: o.earth_radius,
        }
    }
}

impl ZtvOptions {
    fn view(&self) -> Options {
        Options {
            max_distance: self.max_distance,
            curvature: self.curvature,
            refraction: self.refraction,
            earth_radius: self.earth_radius,
            ..Options::default()
        }
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
                    surface: Surface::RViewshed,
                    ..opts.view()
                };
                let seen = crate::viewshed(grid, tg.x, tg.y, &o)?;
                counts.iter_mut().zip(seen).for_each(|(n, v)| *n += v as u32);
            }
            Ok(counts)
        }
        ZtvMethod::Forward => {
            let cols = grid.cols;
            let drop = opts.view().drop_factor();
            let pyramid = MaxPyramid::new(grid);
            let mut counts = vec![0u32; grid.rows * cols];
            counts.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
                for (c, n) in row.iter_mut().enumerate() {
                    let z = grid.at(r, c);
                    if z.is_nan() {
                        continue;
                    }
                    let eye = z + opts.observer_height;
                    for &(rt, ct, h) in &cells {
                        if sees(grid, &pyramid, r as i64, c as i64, eye, drop, rt, ct, h, opts.max_distance) {
                            *n += 1;
                        }
                    }
                }
            });
            Ok(counts)
        }
    }
}

/// Maximum height over aligned 2^l x 2^l blocks, for every level l. No-data
/// counts as -infinity (it never blocks).
pub(crate) struct MaxPyramid {
    levels: Vec<(usize, usize, Vec<f64>)>, // (rows, cols, maxima), level 0 = the grid
}

impl MaxPyramid {
    pub(crate) fn new(grid: &Grid) -> Self {
        let base: Vec<f64> = grid.z.iter().map(|&v| if v.is_nan() { f64::NEG_INFINITY } else { v }).collect();
        let mut levels = vec![(grid.rows, grid.cols, base)];
        while {
            let (r, c, _) = levels.last().unwrap();
            *r > 1 || *c > 1
        } {
            let (r, c, prev) = levels.last().unwrap();
            let (nr, nc) = ((r + 1) / 2, (c + 1) / 2);
            let mut next = vec![f64::NEG_INFINITY; nr * nc];
            for i in 0..*r {
                for j in 0..*c {
                    let k = (i / 2) * nc + j / 2;
                    next[k] = next[k].max(prev[i * c + j]);
                }
            }
            levels.push((nr, nc, next));
        }
        MaxPyramid { levels }
    }

    /// An upper bound on the height of any cell in rows r0..=r1, cols c0..=c1
    /// (clipped to the grid): the maximum of the few blocks that cover them.
    pub(crate) fn max_in(&self, r0: i64, r1: i64, c0: i64, c1: i64) -> f64 {
        let (rows, cols, _) = self.levels[0];
        let (r0, r1) = (r0.max(0), r1.min(rows as i64 - 1));
        let (c0, c1) = (c0.max(0), c1.min(cols as i64 - 1));
        if r0 > r1 || c0 > c1 {
            return f64::NEG_INFINITY;
        }
        let span = ((r1 - r0).max(c1 - c0) + 1) as u64;
        let l = ((63 - span.leading_zeros()) as usize).min(self.levels.len() - 1);
        let (_, nc, ref m) = self.levels[l];
        let mut best = f64::NEG_INFINITY;
        for i in (r0 >> l)..=(r1 >> l) {
            for j in (c0 >> l)..=(c1 >> l) {
                best = best.max(m[i as usize * nc + j as usize]);
            }
        }
        best
    }
}

fn cross(a: (i64, i64), b: (i64, i64)) -> i128 {
    a.0 as i128 * b.1 as i128 - a.1 as i128 * b.0 as i128
}

/// Can the eye over cell (rv, cv) see `height` above the centre of cell (rt, ct)?
/// Exactly the r.viewshed-model answer a full sweep from (rv, cv) would give.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sees(
    grid: &Grid,
    pyramid: &MaxPyramid,
    rv: i64,
    cv: i64,
    eye: f64,
    drop: f64,
    rt: i64,
    ct: i64,
    height: f64,
    max_d: Option<f64>,
) -> bool {
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
        slope: (zt - dist * dist * drop + height - eye) / dist,
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
    let (major_sign, minor_end, major_end) = if x_major { (dc.signum(), dr, dc) } else { (dr.signum(), dc, dr) };
    let minor_at = |i: i64| (i * major_sign) as f64 * minor_end as f64 / major_end as f64;
    let mut visit = |i: i64| -> bool {
        let major = i * major_sign;
        let k0 = minor_at(i).floor() as i64;
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
            let cell = cell_at(grid, rv, cv, eye, drop, r, c);
            if cell.ring == u32::MAX || cell.hi() < skip_below {
                continue;
            }
            if cell.lo() > sure_above || target.blocked_by(&cell) {
                return true;
            }
        }
        false
    };

    // Bound for steps [a, b): no sample of any candidate cell there can have a
    // slope above this. Heights: the maximum over the candidates and the
    // neighbours their corners average in (corner means can't exceed it, and
    // curvature only lowers them). Distances: every sample is at least
    // a - 0.5 cells along the major axis and at most the far corner away.
    // Floating-point division and subtraction are monotone, so the bound
    // holds for the computed slopes too, and skipping a range where it is
    // below `skip_below` skips only cells the per-cell test would skip.
    let res = grid.res();
    let bound = |a: i64, b: i64| -> f64 {
        let (m0, m1) = (minor_at(a), minor_at(b - 1));
        let kmin = m0.min(m1).floor() as i64 - 2;
        let kmax = m0.max(m1).floor() as i64 + 3;
        let (j0, j1) = ((a - 1) * major_sign, b * major_sign);
        let (jmin, jmax) = (j0.min(j1), j0.max(j1));
        let (r0, r1, c0, c1) = if x_major {
            (rv + kmin, rv + kmax, cv + jmin, cv + jmax)
        } else {
            (rv + jmin, rv + jmax, cv + kmin, cv + kmax)
        };
        let h = pyramid.max_in(r0, r1, c0, c1);
        if h == f64::NEG_INFINITY {
            return f64::NEG_INFINITY; // nothing but no-data or off the grid
        }
        let rise = h - eye;
        if rise >= 0.0 {
            let dmin = (a as f64 - 0.5).max(0.5) * res * (1.0 - 1e-12);
            rise / dmin
        } else {
            let far = (b as f64 + 3.5) * res;
            let dmax = (far * far * 2.0).sqrt() * (1.0 + 1e-12);
            rise / dmax
        }
    };

    // Walk the steps near to far, skipping whole ranges the bound rules out.
    let mut stack = [(0i64, 0i64); 64];
    stack[0] = (0, n + 1);
    let mut top = 1;
    while top > 0 {
        top -= 1;
        let (a, b) = stack[top];
        if b - a <= 4 {
            for i in a..b {
                if visit(i) {
                    return false;
                }
            }
            continue;
        }
        if bound(a, b) < skip_below {
            continue;
        }
        let mid = (a + b) / 2;
        stack[top] = (mid, b);
        stack[top + 1] = (a, mid);
        top += 2;
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
        // flat earth, then a tiny earth (radius 60 m) so curvature decides many cells
        for (seed, integer, holes, n, curvature) in [
            (1, false, false, 21, false),
            (2, true, false, 19, false),
            (3, false, true, 17, false),
            (4, true, true, 23, false),
            (5, false, false, 21, true),
            (6, true, true, 19, true),
        ] {
            let z = terrain(n, seed, integer, holes);
            let gt = [0.0, 1.0, 0.0, n as f64, 0.0, -1.0];
            let grid = Grid::new(&z, n, n, gt).unwrap();
            let mut targets = Vec::new();
            for (r, c, h) in [(3usize, 4usize, 5.0), (n / 2, n / 2, 0.0), (n - 2, 1, 12.5)] {
                if !z[r * n + c].is_nan() {
                    targets.push((r, c, ZtvTarget { x: c as f64 + 0.5, y: (n - r) as f64 - 0.5, height: h }));
                }
            }
            let opts = ZtvOptions { observer_height: 1.5, curvature, earth_radius: 60.0, ..ZtvOptions::default() };
            let counts = ztv(&grid, &targets.iter().map(|t| t.2).collect::<Vec<_>>(), &opts).unwrap();
            for r in 0..n {
                for c in 0..n {
                    if z[r * n + c].is_nan() {
                        assert_eq!(counts[r * n + c], 0);
                        continue;
                    }
                    let o = Options { observer_height: 1.5, curvature, earth_radius: 60.0, ..Options::default() };
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
