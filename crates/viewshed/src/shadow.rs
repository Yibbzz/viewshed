//! Exact cast-shadow maps on the bilinear surface.
//!
//! Model (as the viewshed's bilinear surface): heights at cell centres, bilinear between them,
//! clamped at the raster border; no-data never blocks and is never lit. The sun is the float
//! vector `(dx, dy, dz)` from [`Sun::direction`] (per unit ray parameter: `dx` columns, `dy` rows,
//! `dz` up; "exact" means exact for that vector). A cell T (its centre, height `z_T +
//! target_height`) is shadowed iff some point on T's ray at parameter `s > 0`, on a stretch of
//! positive length of a patch with data, is strictly above the ray after lowering it by the
//! curvature drop `k e^2` (horizontal distance `e`). Rays leave the raster: nothing beyond the edge
//! blocks.
//!
//! Method (research notes: shadow-2026-11-23.md, shadow-rigour-2026-11-24.md). With
//! `a = u dx + v dy`, `l = -u dy + v dx`, `lambda = dz / (dx^2 + dy^2)` and `g = z - lambda a`:
//! 1. a bilinear patch lies in the convex hull of its 4 corners and `g` is affine, so the patch's
//!    max `g` is at a corner; dropping the curvature term (k >= 0) keeps "max g <= g_T" sufficient
//!    for lit;
//! 2. lateral strips (T's ray stays in T's strip): a running max of corner `g` against the sun
//!    certifies most cells lit;
//! 3. the first stretch of T's ray (patches too near for the strip bound) and every patch whose
//!    bound reaches `g_T` are decided on T's own ray (a quadratic in `s`, no square root), with
//!    certified rounding bounds; ties go to exact rational arithmetic for that cell.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;

use crate::{Error, Grid, MEAN_EARTH_RADIUS};

const U: f64 = f64::EPSILON / 2.0;
/// A generous multiple of the unit roundoff for the short sums and products below (each value
/// is a sum of at most ~8 products of at most 4 exact floats: gamma_16 would do; 64 U is safe).
const ERR: f64 = 64.0 * U;
/// Local stretch in ray-parameter units beyond the patch extent: c0 in the notes.
const C0: f64 = 1.0;

/// Outward rounding of a margin-adjusted value: the final add/sub of a margin rounds too (review
/// point 1); moving a few ulps further keeps the comparison on the safe side.
#[inline]
fn up(x: f64) -> f64 {
    x + (x.abs() * 4.0 * U + f64::MIN_POSITIVE)
}
#[inline]
fn down(x: f64) -> f64 {
    x - (x.abs() * 4.0 * U + f64::MIN_POSITIVE)
}

/// The sun's position: azimuth clockwise from north and elevation above the horizon, degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sun {
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
}

impl Sun {
    /// The direction vector `(dx, dy, dz)` the model is exact for: per unit ray parameter, `dx`
    /// columns and `dy` rows (signs from the geotransform) and `dz` map units up, for a horizontal
    /// step of one cell. Errors on elevation outside (0, 90) degrees.
    pub fn direction(&self, grid: &Grid) -> Result<[f64; 3], Error> {
        if !(self.elevation_deg > 0.0 && self.elevation_deg < 90.0) || !self.azimuth_deg.is_finite() {
            return Err(Error::InvalidSun);
        }
        // cardinal azimuths exactly (sin 180 deg is 1.2e-16 in floats: a ray a hair off the grid
        // line, while 0 deg is exactly on it); other angles through sin/cos. Exact for the
        // resulting float vector, as the oracle's sun_vector (same rule).
        let a = self.azimuth_deg.rem_euclid(360.0);
        let (sa, ca) = if a == 0.0 {
            (0.0, 1.0)
        } else if a == 90.0 {
            (1.0, 0.0)
        } else if a == 180.0 {
            (0.0, -1.0)
        } else if a == 270.0 {
            (-1.0, 0.0)
        } else {
            let r = a.to_radians();
            (r.sin(), r.cos())
        };
        let t = grid.transform;
        let res = grid.res();
        Ok([sa * (res / t[1]), ca * (res / t[5]), self.elevation_deg.to_radians().tan() * res])
    }
}

/// Shadow settings. Curvature and refraction as in [`crate::Options`].
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowOptions {
    /// Height above the surface of the point tested in each cell.
    pub target_height: f64,
    pub curvature: bool,
    pub refraction: f64,
    pub earth_radius: f64,
}

impl Default for ShadowOptions {
    fn default() -> Self {
        ShadowOptions { target_height: 0.0, curvature: true, refraction: 0.13, earth_radius: MEAN_EARTH_RADIUS }
    }
}

impl ShadowOptions {
    fn drop_factor(&self) -> f64 {
        if self.curvature {
            (1.0 - self.refraction) / (2.0 * self.earth_radius)
        } else {
            0.0
        }
    }
}

/// Lit mask, row-major (`true` = lit; no-data `false`), exact on the bilinear surface.
pub fn shadow(grid: &Grid, sun: Sun, opts: &ShadowOptions) -> Result<Vec<bool>, Error> {
    let v = sun.direction(grid)?;
    shadow_vec(grid, v, opts)
}

/// [`shadow`] for a direction vector given directly (exact for those floats). For tests.
#[doc(hidden)]
pub fn shadow_vec(grid: &Grid, v: [f64; 3], opts: &ShadowOptions) -> Result<Vec<bool>, Error> {
    let [dx, dy, dz] = v;
    if !(dx.is_finite() && dy.is_finite() && dz.is_finite()) || (dx == 0.0 && dy == 0.0) || dz < 0.0 {
        return Err(Error::InvalidSun);
    }
    if opts.target_height.is_nan() || opts.target_height < 0.0 {
        return Err(Error::InvalidTargetHeight);
    }
    let k = opts.drop_factor();
    if k < 0.0 {
        return Err(Error::UnsupportedRefraction);
    }
    let s = Setup::new(grid, dx, dy, dz, k, opts.target_height);
    Ok(s.run())
}

struct Setup<'a> {
    grid: &'a Grid<'a>,
    dx: f64,
    dy: f64,
    dz: f64,
    k: f64,
    th: f64,
    res: f64,
    l2: f64,
    lam: f64,
    /// strip width in l-units: a power of two (exact division)
    w: f64,
    /// widening of lateral ranges and a-keys (rounding of l and a at the raster's coordinates)
    e_l: f64,
    e_a: f64,
    /// the local stretch's extent in s: c0 plus a patch's along-sun extent
    s_loc: f64,
    check: bool,
}

#[derive(Clone, Copy)]
struct Ent {
    a: f64,
    g: f64,
    r0: i32,
    c0: i32,
}

/// Decision of one patch on T's ray: Above (certainly blocks), Clear, or Tie.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dec {
    Above,
    Clear,
    Tie,
}

impl<'a> Setup<'a> {
    fn new(grid: &'a Grid<'a>, dx: f64, dy: f64, dz: f64, k: f64, th: f64) -> Setup<'a> {
        let res = grid.res();
        let l2 = dx * dx + dy * dy;
        let lam = dz / l2;
        let lw = 0.25 * l2.sqrt();
        let w = 2f64.powi(lw.log2().round() as i32);
        let (rows, cols) = (grid.rows as f64, grid.cols as f64);
        let span = (rows + cols + 4.0) * (dx.abs() + dy.abs());
        Setup {
            grid,
            dx,
            dy,
            dz,
            k,
            th,
            res,
            l2,
            lam,
            w,
            e_l: 4.0 * ERR * span + 1e-300,
            e_a: 4.0 * ERR * span + 1e-300,
            s_loc: C0 + (dx.abs() + dy.abs()) / l2 * (1.0 + ERR) + ERR,
            check: std::env::var("VIEWSHED_SHADOW_CHECK").is_ok(),
        }
    }

    #[inline]
    fn at(&self, r: i64, c: i64) -> f64 {
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        self.grid.at(r.clamp(0, rows - 1) as usize, c.clamp(0, cols - 1) as usize)
    }

    #[inline]
    fn corners(&self, r0: i64, c0: i64) -> [f64; 4] {
        [self.at(r0, c0), self.at(r0, c0 + 1), self.at(r0 + 1, c0), self.at(r0 + 1, c0 + 1)]
    }

    #[inline]
    fn a_of(&self, u: f64, v: f64) -> f64 {
        u * self.dx + v * self.dy
    }

    #[inline]
    fn l_of(&self, u: f64, v: f64) -> f64 {
        -u * self.dy + v * self.dx
    }

    /// Upper bound on g over patch (r0, c0) (corner lemma + rounding), its a_min lower bound and
    /// lateral range (widened); None if the patch has no data.
    fn entry(&self, r0: i64, c0: i64) -> Option<(f64, f64, f64, f64)> {
        let zc = self.corners(r0, c0);
        if zc.iter().any(|v| v.is_nan()) {
            return None;
        }
        let (mut g, mut am, mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for (j, (dr, dc)) in [(0i64, 0i64), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            let (u, v) = ((c0 + dc) as f64, (r0 + dr) as f64);
            let a = self.a_of(u, v);
            let l = self.l_of(u, v);
            let t = self.lam * a;
            g = g.max(up(zc[j] - t + ERR * (zc[j].abs() + t.abs() + (u * self.dx).abs() * self.lam.abs() + (v * self.dy).abs() * self.lam.abs())));
            am = am.min(a);
            lo = lo.min(l);
            hi = hi.max(l);
        }
        Some((g, down(am - self.e_a), down(lo - self.e_l), up(hi + self.e_l)))
    }

    /// Patch (r0, c0) on T's ray (T = (r, c)) over s in (0, smax]: certified decision.
    fn patch_on_ray(&self, r: i64, c: i64, r0: i64, c0: i64, smax: f64) -> Dec {
        let zc = self.corners(r0, c0);
        if zc.iter().any(|v| v.is_nan()) {
            return Dec::Clear;
        }
        let (dx, dy) = (self.dx, self.dy);
        // the stretch: each bound is (integer) / float: relative error <= U
        let (mut s1, mut s2) = (0.0f64, smax);
        for (p0, d, k0) in [(c as f64, dx, c0 as f64), (r as f64, dy, r0 as f64)] {
            if d == 0.0 {
                if !(k0 <= p0 && p0 <= k0 + 1.0) {
                    return Dec::Clear;
                }
                continue;
            }
            let (x, y) = ((k0 - p0) / d, (k0 + 1.0 - p0) / d);
            s1 = s1.max(x.min(y));
            s2 = s2.min(x.max(y));
        }
        let es = 4.0 * U * s1.abs().max(s2.abs()) + 1e-300;
        if s2 < s1 - es {
            return Dec::Clear;
        }
        let [z00, z01, z10, z11] = zc;
        let own = (r0 == r || r0 == r - 1) && (c0 == c || c0 == c - 1);
        let e3 = z00 - z01 - z10 + z11;
        let zs = z00.abs() + z01.abs() + z10.abs() + z11.abs();
        let (u0, v0) = ((c - c0) as f64, (r - r0) as f64);
        let curv = self.k * self.res * self.res * self.l2;
        let qa = e3 * dx * dy - curv;
        let qb = (z01 - z00) * dx + (z10 - z00) * dy + e3 * (u0 * dy + v0 * dx) - self.dz;
        let zt = self.at(r, c) + self.th;
        let qc = if own { -self.th } else { z00 + (z01 - z00) * u0 + (z10 - z00) * v0 + e3 * u0 * v0 - zt };
        let ea = ERR * (zs * (dx * dy).abs() + curv);
        let eb = ERR * (zs * (dx.abs() + dy.abs()) * (1.0 + u0.abs() + v0.abs()) + self.dz.abs());
        let ec = if own { 0.0 } else { ERR * (zs * (1.0 + u0.abs()) * (1.0 + v0.abs()) + zt.abs()) };
        // error of the evaluated polynomial at s (coefficient errors + evaluation rounding)
        let eq = |s: f64| ea * s * s + eb * s + ec + ERR * (qa.abs() * s * s + qb.abs() * s + qc.abs());
        let q = |s: f64| (qa * s + qb) * s + qc;
        if s2 - s1 <= es {
            // near-degenerate: a point or a sliver; a tie if its value could be above
            let s = 0.5 * (s1 + s2);
            return if s > 0.0 && q(s) + eq(s) + (2.0 * qa.abs() * s + qb.abs()) * es > 0.0 { Dec::Tie } else { Dec::Clear };
        }
        // upper bound over the enlarged stretch [s1 - es, s2 + es] (a superset: still an upper bound)
        let (lo, hi) = ((s1 - es).max(0.0), s2 + es);
        let mut ub = f64::NEG_INFINITY;
        for s in [lo, hi] {
            if s > 0.0 || !own {
                ub = ub.max(q(s) + eq(s));
            }
        }
        let mut vertex = None;
        if qa < 0.0 {
            let sv = -qb / (2.0 * qa);
            if lo < sv && sv < hi {
                let m = qc - qb * qb / (4.0 * qa);
                let em = ec + sv.abs() * eb + sv * sv * ea + ERR * (qc.abs() + (qb * qb / (4.0 * qa)).abs());
                ub = ub.max(m + em);
                vertex = Some(sv);
            }
        }
        // T's own patch with th = 0: q(0) = 0 exactly and q(s) = s (qa s + qb). A slope qb within its
        // error of 0 can't be decided in floats: the true qb may be slightly positive (it is a
        // rounded sum of products), and then q > 0 on (0, qb / |qa|), a positive-length stretch.
        // Such a patch is never Clear: Above if certified below, else a tie (exact fallback).
        // (Review by the engine agent, 2026-11-24.)
        let zero_slope = own && s1 == 0.0 && self.th == 0.0 && qb.abs() <= eb;
        if own && s1 == 0.0 && self.th == 0.0 && qb > eb {
            return Dec::Above;
        }
        if ub <= 0.0 && !zero_slope {
            return Dec::Clear;
        }
        // lower bound at points certainly inside the stretch (shrunk ends, the vertex)
        let (ilo, ihi) = (s1 + es, s2 - es);
        if ihi > ilo {
            let mut lb = f64::NEG_INFINITY;
            for s in [ilo, ihi] {
                if s > 0.0 {
                    lb = lb.max(q(s) - eq(s));
                }
            }
            if let Some(sv) = vertex {
                if ilo < sv && sv < ihi && sv > 0.0 {
                    lb = lb.max(q(sv) - eq(sv) - (2.0 * qa.abs() * sv + qb.abs()) * 4.0 * U * sv);
                }
            }
            if lb > 0.0 {
                return Dec::Above;
            }
        }
        Dec::Tie
    }

    /// The patches T's ray crosses over s in (0, smax], by an exact-order DDA: the cells the open
    /// ray passes through, in order of the crossing parameters (float errors ~1e-15 over the few
    /// steps of the local stretch). Where the two crossings are within 1e-9 of each other the ray
    /// passes (near) a vertex and the order is uncertain: both side cells are evaluated. A ray on a
    /// grid line (dx or dy exactly 0) also evaluates the cells across that line (positive-length
    /// stretches along their shared edge). T itself is a vertex on the grid lines through it, so a
    /// ray merely near one leaves into one side only. VIEWSHED_SHADOW_CHECK asserts the superset.
    fn walk(&self, r: i64, c: i64, smax: f64) -> Dec {
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        let (dx, dy) = (self.dx, self.dy);
        let mut tie = false;
        let mut seen: Vec<(i64, i64)> = Vec::new();
        let eval = |r0: i64, c0: i64, tie: &mut bool, seen: &mut Vec<(i64, i64)>| -> bool {
            if r0 < -1 || c0 < -1 || r0 >= rows || c0 >= cols {
                return false;
            }
            if self.check {
                seen.push((r0, c0));
            }
            match self.patch_on_ray(r, c, r0, c0, smax) {
                Dec::Above => true,
                Dec::Tie => {
                    *tie = true;
                    false
                }
                Dec::Clear => false,
            }
        };
        let (sx, sy) = (if dx > 0.0 { 1 } else { -1 }, if dy > 0.0 { 1 } else { -1 });
        let mut cx = if dx > 0.0 { c } else { c - 1 };
        let mut cy = if dy > 0.0 { r } else { r - 1 };
        // crossing parameters of the next vertical (x) and horizontal (y) grid lines
        let tdx = if dx != 0.0 { (1.0 / dx).abs() } else { f64::INFINITY };
        let tdy = if dy != 0.0 { (1.0 / dy).abs() } else { f64::INFINITY };
        let mut tx = if dx != 0.0 { tdx } else { f64::INFINITY };
        let mut ty = if dy != 0.0 { tdy } else { f64::INFINITY };
        let mut t = 0.0f64;
        while t < smax && cx >= -1 && cy >= -1 && cx < cols && cy < rows {
            if eval(cy, cx, &mut tie, &mut seen) {
                return Dec::Above;
            }
            // a ray on a grid line: the cell across it shares a positive-length edge stretch
            if dx == 0.0 && eval(cy, cx + 1, &mut tie, &mut seen) {
                return Dec::Above;
            }
            if dy == 0.0 && eval(cy + 1, cx, &mut tie, &mut seen) {
                return Dec::Above;
            }
            let near = (tx - ty).abs() <= 1e-9 * tx.min(ty).max(1.0);
            if near && tx.min(ty) < smax {
                // through (near) a vertex: either side cell may carry a stretch
                if eval(cy, cx + sx, &mut tie, &mut seen) || eval(cy + sy, cx, &mut tie, &mut seen) {
                    return Dec::Above;
                }
                t = tx.max(ty);
                cx += sx;
                cy += sy;
                tx += tdx;
                ty += tdy;
            } else if tx < ty {
                t = tx;
                cx += sx;
                tx += tdx;
            } else {
                t = ty;
                cy += sy;
                ty += tdy;
            }
        }
        if self.check {
            self.check_walk(r, c, smax, &seen);
        }
        if tie { Dec::Tie } else { Dec::Clear }
    }

    /// VIEWSHED_SHADOW_CHECK: every patch whose stretch of T's ray has positive length and starts
    /// before smax, by the exact interval test (rational, as hidden_exact), over all patches within
    /// smax + 2 cells, was evaluated by the walk.
    fn check_walk(&self, r: i64, c: i64, smax: f64, seen: &[(i64, i64)]) {
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        let q = |v: f64| BigRational::from_float(v).expect("finite");
        let int = |v: i64| BigRational::from_integer(BigInt::from(v));
        let (dx, dy, sm) = (q(self.dx), q(self.dy), q(smax));
        let reach = (smax * self.l2.sqrt()).ceil() as i64 + 2;
        for r0 in (r - reach).max(-1)..=(r + reach).min(rows - 1) {
            for c0 in (c - reach).max(-1)..=(c + reach).min(cols - 1) {
                if self.corners(r0, c0).iter().any(|v| v.is_nan()) {
                    continue;
                }
                let mut lo = BigRational::zero();
                let mut hi: Option<BigRational> = None;
                let mut meets = true;
                for (p0, d, k0) in [(int(c), &dx, int(c0)), (int(r), &dy, int(r0))] {
                    if d.is_zero() {
                        meets &= k0 <= p0 && p0 <= &k0 + int(1);
                        continue;
                    }
                    let x = (&k0 - &p0) / d;
                    let y = (&k0 + int(1) - &p0) / d;
                    let (a_, b_) = if x <= y { (x, y) } else { (y, x) };
                    if a_ > lo {
                        lo = a_;
                    }
                    hi = Some(match hi {
                        Some(h) if h < b_ => h,
                        _ => b_,
                    });
                }
                let Some(hi) = hi else { continue };
                if meets && hi > lo && lo < sm {
                    assert!(seen.contains(&(r0, c0)), "VIEWSHED_SHADOW_CHECK: patch ({r0}, {c0}) crossed by ({r}, {c})'s ray at s {}..{} (< {smax}) not walked", lo.to_f64().unwrap_or(f64::NAN), hi.to_f64().unwrap_or(f64::NAN));
                }
            }
        }
    }

    fn run(&self) -> Vec<bool> {
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        let n = (rows * cols) as usize;
        // lateral range of the domain (patch corners -1..=rows, -1..=cols) and strips
        let mut lmin = f64::INFINITY;
        let mut lmax = f64::NEG_INFINITY;
        for (u, v) in [(-1.0, -1.0), (cols as f64, -1.0), (-1.0, rows as f64), (cols as f64, rows as f64)] {
            lmin = lmin.min(self.l_of(u, v));
            lmax = lmax.max(self.l_of(u, v));
        }
        let smin = ((lmin - self.e_l) / self.w).floor() as i64 - 1;
        let ns = (((lmax + self.e_l) / self.w).floor() as i64 - smin + 2) as usize;
        // targets bucketed by strip
        let mut tstart = vec![0usize; ns + 1];
        let tstrip: Vec<u32> = (0..n)
            .map(|i| {
                let (r, c) = ((i as i64) / cols, (i as i64) % cols);
                ((self.l_of(c as f64, r as f64) / self.w).floor() as i64 - smin) as u32
            })
            .collect();
        for &s in &tstrip {
            tstart[s as usize + 1] += 1;
        }
        for i in 0..ns {
            tstart[i + 1] += tstart[i];
        }
        let mut tpos = tstart.clone();
        let mut tlist = vec![0u32; n];
        for (i, &s) in tstrip.iter().enumerate() {
            tlist[tpos[s as usize]] = i as u32;
            tpos[s as usize] += 1;
        }
        // per strip: gather, sort, sweep against the sun
        let decided: Vec<(u32, u8)> = (0..ns)
            .into_par_iter()
            .filter(|&s| tstart[s + 1] > tstart[s])
            .flat_map_iter(|s| self.strip(s as i64 + smin, &tlist[tstart[s]..tstart[s + 1]]).into_iter())
            .collect();
        let mut lit = vec![false; n];
        let mut ties = Vec::new();
        for (i, st) in decided {
            match st {
                0 => lit[i as usize] = true,
                2 => ties.push(i),
                _ => {}
            }
        }
        // exact rational decisions for the ties
        let exact: Vec<(u32, bool)> = ties
            .par_iter()
            .map(|&i| {
                let (r, c) = ((i as i64) / cols, (i as i64) % cols);
                (i, !self.hidden_exact(r, c))
            })
            .collect();
        for (i, v) in exact {
            lit[i as usize] = v;
        }
        lit
    }

    /// One strip j (l in [j w, (j + 1) w)): (cell, 0 lit / 1 shadowed / 2 tie) for its targets.
    fn strip(&self, j: i64, targets: &[u32]) -> Vec<(u32, u8)> {
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        let (l0, l1) = (j as f64 * self.w, (j + 1) as f64 * self.w);
        // patches meeting the widened slab: walk the major axis, exact filter
        let mut ents: Vec<Ent> = Vec::new();
        let push = |r0: i64, c0: i64, ents: &mut Vec<Ent>| {
            if r0 < -1 || c0 < -1 || r0 >= rows || c0 >= cols {
                return;
            }
            if let Some((g, am, lo, hi)) = self.entry(r0, c0) {
                if hi >= l0 && lo <= l1 {
                    ents.push(Ent { a: am, g, r0: r0 as i32, c0: c0 as i32 });
                }
            }
        };
        let (dx, dy) = (self.dx, self.dy);
        let pad = 2;
        if dx.abs() >= dy.abs() {
            // l = -u dy + v dx: v = (l + u dy) / dx
            for c0 in -1..cols {
                let (u0, u1) = (c0 as f64, (c0 + 1) as f64);
                let vs = [(l0 + u0 * dy) / dx, (l0 + u1 * dy) / dx, (l1 + u0 * dy) / dx, (l1 + u1 * dy) / dx];
                let vlo = vs.iter().cloned().fold(f64::INFINITY, f64::min);
                let vhi = vs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for r0 in ((vlo.floor() as i64) - pad).max(-1)..=((vhi.floor() as i64) + pad).min(rows - 1) {
                    push(r0, c0, &mut ents);
                }
            }
        } else {
            // u = (v dx - l) / dy
            for r0 in -1..rows {
                let (v0, v1) = (r0 as f64, (r0 + 1) as f64);
                let us = [(v0 * dx - l0) / dy, (v1 * dx - l0) / dy, (v0 * dx - l1) / dy, (v1 * dx - l1) / dy];
                let ulo = us.iter().cloned().fold(f64::INFINITY, f64::min);
                let uhi = us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for c0 in ((ulo.floor() as i64) - pad).max(-1)..=((uhi.floor() as i64) + pad).min(cols - 1) {
                    push(r0, c0, &mut ents);
                }
            }
        }
        ents.sort_unstable_by(|x, y| y.a.total_cmp(&x.a));
        let m = ents.len();
        let p2 = m.next_power_of_two().max(1);
        let mut tree = vec![f64::NEG_INFINITY; 2 * p2];
        for (i, e) in ents.iter().enumerate() {
            tree[p2 + i] = e.g;
        }
        for i in (1..p2).rev() {
            tree[i] = tree[2 * i].max(tree[2 * i + 1]);
        }
        // queries: a-cut key a_T + c0 L^2 (+ e_a), descending
        let mut qs: Vec<(f64, u32)> = targets
            .iter()
            .filter(|&&i| !self.grid.z[i as usize].is_nan())
            .map(|&i| {
                let (r, c) = ((i as i64) / cols, (i as i64) % cols);
                (up(self.a_of(c as f64, r as f64) + C0 * self.l2 + self.e_a), i)
            })
            .collect();
        qs.sort_unstable_by(|x, y| y.0.total_cmp(&x.0));
        let mut out = Vec::with_capacity(qs.len());
        let mut jj = 0usize;
        let mut runmax = f64::NEG_INFINITY;
        let mut stack: Vec<(usize, usize, usize)> = Vec::new();
        for &(q, i) in &qs {
            while jj < m && ents[jj].a >= q {
                runmax = runmax.max(ents[jj].g);
                jj += 1;
            }
            let (r, c) = ((i as i64) / cols, (i as i64) % cols);
            if self.check {
                self.check_cover(r, c, q, &ents[..jj]);
            }
            let zt = self.grid.z[i as usize] + self.th;
            let at = self.a_of(c as f64, r as f64);
            let gt = zt - self.lam * at;
            let egt = ERR * (zt.abs() + (self.lam * at).abs() + self.lam.abs() * ((c as f64 * self.dx).abs() + (r as f64 * self.dy).abs()));
            let thr = down(gt - egt);
            // local stretch
            let mut tie = match self.walk(r, c, self.s_loc) {
                Dec::Above => {
                    out.push((i, 1u8));
                    continue;
                }
                Dec::Tie => true,
                Dec::Clear => false,
            };
            if jj == 0 || runmax < thr {
                out.push((i, if tie { 2 } else { 0 }));
                continue;
            }
            // enumerate entries [0, jj) with bound >= thr, highest first; stop at the first blocker
            let mut hidden = false;
            stack.clear();
            stack.push((1, 0, p2));
            while let Some((nd, lo, hi)) = stack.pop() {
                if lo >= jj || tree[nd] < thr {
                    continue;
                }
                if nd >= p2 {
                    let e = ents[nd - p2];
                    match self.patch_on_ray(r, c, e.r0 as i64, e.c0 as i64, f64::INFINITY) {
                        Dec::Above => {
                            hidden = true;
                            break;
                        }
                        Dec::Tie => tie = true,
                        Dec::Clear => {}
                    }
                    continue;
                }
                let mid = (lo + hi) / 2;
                let (a_, b_) = ((2 * nd, lo, mid), (2 * nd + 1, mid, hi));
                if tree[a_.0] >= tree[b_.0] {
                    stack.push(b_);
                    stack.push(a_);
                } else {
                    stack.push(a_);
                    stack.push(b_);
                }
            }
            out.push((i, if hidden { 1 } else if tie { 2 } else { 0 }));
        }
        out
    }

    /// VIEWSHED_SHADOW_CHECK: every patch T's ray crosses (positive-length stretch, beyond the local
    /// stretch) is among the strip's qualifying entries.
    fn check_cover(&self, r: i64, c: i64, q: f64, qual: &[Ent]) {
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        for r0 in -1..rows {
            for c0 in -1..cols {
                if self.corners(r0, c0).iter().any(|v| v.is_nan()) {
                    continue;
                }
                let (mut s1, mut s2) = (0.0f64, f64::INFINITY);
                let mut meets = true;
                for (p0, d, k0) in [(c as f64, self.dx, c0 as f64), (r as f64, self.dy, r0 as f64)] {
                    if d == 0.0 {
                        meets &= k0 <= p0 && p0 <= k0 + 1.0;
                        continue;
                    }
                    let (x, y) = ((k0 - p0) / d, (k0 + 1.0 - p0) / d);
                    s1 = s1.max(x.min(y));
                    s2 = s2.min(x.max(y));
                }
                if !meets || s2 <= s1 + 1e-9 || s1 < self.s_loc - 1e-6 {
                    continue; // not crossed, or (partly) inside the local stretch
                }
                assert!(
                    qual.iter().any(|e| e.r0 as i64 == r0 && e.c0 as i64 == c0),
                    "VIEWSHED_SHADOW_CHECK: patch ({r0}, {c0}) crossed by ({r}, {c})'s ray at s {s1}..{s2} is not a qualifying entry (key {q})"
                );
            }
        }
    }

    /// Exact rational decision (the oracle's procedure): is cell (r, c) shadowed?
    fn hidden_exact(&self, r: i64, c: i64) -> bool {
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        let q = |v: f64| BigRational::from_float(v).expect("finite");
        let (dx, dy, dz) = (q(self.dx), q(self.dy), q(self.dz));
        let zt = q(self.at(r, c)) + q(self.th);
        let curv = q(self.k) * q(self.res) * q(self.res) * (&dx * &dx + &dy * &dy);
        let two = BigRational::from_integer(BigInt::from(2));
        let four = BigRational::from_integer(BigInt::from(4));
        let int = |v: i64| BigRational::from_integer(BigInt::from(v));
        // candidates: a float walk widened to 3 x 3 (a superset; the exact test decides)
        let l = self.l2.sqrt();
        let (ux, uy) = (self.dx / l, self.dy / l);
        let mut cand = std::collections::BTreeSet::new();
        let mut t = 0.0f64;
        while t <= 2.0 * (rows + cols) as f64 + 4.0 {
            let (cx, cy) = ((c as f64 + t * ux).floor() as i64, (r as f64 + t * uy).floor() as i64);
            if cx < -2 || cy < -2 || cx > cols + 1 || cy > rows + 1 {
                break;
            }
            for a in -1..=1 {
                for b in -1..=1 {
                    if (-1..rows).contains(&(cy + a)) && (-1..cols).contains(&(cx + b)) {
                        cand.insert((cy + a, cx + b));
                    }
                }
            }
            t += 0.25;
        }
        for (r0, c0) in cand {
            let zc = self.corners(r0, c0);
            if zc.iter().any(|v| v.is_nan()) {
                continue;
            }
            // s-interval of the patch (closed); positive length required
            let mut lo = BigRational::zero();
            let mut hi: Option<BigRational> = None;
            let mut meets = true;
            for (p0, d, k0) in [(int(c), &dx, int(c0)), (int(r), &dy, int(r0))] {
                if d.is_zero() {
                    meets &= k0 <= p0 && p0 <= &k0 + int(1);
                    continue;
                }
                let x = (&k0 - &p0) / d;
                let y = (&k0 + int(1) - &p0) / d;
                let (a_, b_) = if x <= y { (x, y) } else { (y, x) };
                if a_ > lo {
                    lo = a_;
                }
                hi = Some(match hi {
                    Some(h) if h < b_ => h,
                    _ => b_,
                });
            }
            let Some(hi) = hi else { continue };
            if !meets || hi <= lo {
                continue;
            }
            let [z00, z01, z10, z11] = zc.map(q);
            let e3 = &z00 - &z01 - &z10 + &z11;
            let (u0, v0) = (int(c - c0), int(r - r0));
            let qa = &e3 * &dx * &dy - &curv;
            let qb = (&z01 - &z00) * &dx + (&z10 - &z00) * &dy + &e3 * (&u0 * &dy + &v0 * &dx) - &dz;
            let qc = &z00 + (&z01 - &z00) * &u0 + (&z10 - &z00) * &v0 + &e3 * &u0 * &v0 - &zt;
            let qf = |s: &BigRational| (&qa * s + &qb) * s + &qc;
            let mut cands = vec![hi.clone()];
            if lo > BigRational::zero() {
                cands.push(lo.clone());
            }
            if qa < BigRational::zero() {
                let sv = -&qb / (&two * &qa);
                if lo < sv && sv < hi {
                    cands.push(sv);
                }
            }
            if cands.iter().any(|s| qf(s) > BigRational::zero()) {
                return true;
            }
            if lo.is_zero() && qc.is_zero() && (qb > BigRational::zero() || (qb.is_zero() && qa > BigRational::zero())) {
                return true;
            }
            let _ = &four;
        }
        let _ = zt.to_f64();
        false
    }
}
