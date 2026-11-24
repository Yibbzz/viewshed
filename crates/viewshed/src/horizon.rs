//! Exact horizon maps on the bilinear surface.
//!
//! For a direction (an azimuth, or a float vector `(dx, dy)` for tests) and every cell T (its
//! centre, height `z_T + target_height`), the horizon *rise*
//! `Hs(T) = sup (B(X(s)) - z_T - th - k (s L res)^2) / s` over `s > 0` on positive-length stretches of
//! patches with data, `X(s) = T + s (dx, dy)`, `L = |(dx, dy)|`: the tangent of the horizon's
//! elevation angle is `Hs / (L res)`. The surface, no-data and curvature rules are the viewshed's
//! (as [`shadow`](crate::shadow)); rays leave the raster: nothing beyond the edge counts. T is
//! shadowed by a sun in that direction iff `Hs(T) > dz` (consistency with shadows).
//!
//! Guarantee: each value is within `1e-9 res + 1e-12 |Hs|` of the exact supremum (certified per cell;
//! cells whose certified enclosure is wider are recomputed with the stretches decided in rational
//! arithmetic). Not correctly rounded: an enclosure would have to collapse to one float.
//!
//! Method (research notes: horizon-maps-2026-11-24.md, horizon-rigour-2026-11-24.md): lateral strips
//! along the direction; per strip a segment tree over the patches sorted by their along-ray
//! coordinate, each node holding the upper convex hull of its corners' `(a, z)` (a patch projects
//! into its corners' hull); per cell the exact local stretch, a warm start from the previous cell's
//! best patch, then branch and bound: nodes whose hull bound can't beat the certified lower bound
//! are pruned, the rest evaluated in closed form (`A s + B + C / s`) with certified error bounds.

use std::collections::BTreeSet;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rayon::prelude::*;

use crate::shadow::{azimuth_vector, down, up, C0, ERR, U};
use crate::{Error, Grid, MEAN_EARTH_RADIUS};

/// Horizon settings: as [`ShadowOptions`](crate::ShadowOptions).
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonOptions {
    /// Height above the surface of the point in each cell the horizon is seen from.
    pub target_height: f64,
    pub curvature: bool,
    pub refraction: f64,
    pub earth_radius: f64,
}

impl Default for HorizonOptions {
    fn default() -> Self {
        HorizonOptions { target_height: 0.0, curvature: true, refraction: 0.13, earth_radius: MEAN_EARTH_RADIUS }
    }
}

/// Horizon elevation angle in degrees per cell, looking towards `azimuth_deg` (clockwise from grid
/// north): row-major; -90 where nothing lies ahead (no data all the way to the edge), NaN on
/// no-data. Not clamped at 0: a horizon below the horizontal is negative.
pub fn horizon_map(grid: &Grid, azimuth_deg: f64, opts: &HorizonOptions) -> Result<Vec<f64>, Error> {
    let v = azimuth_vector(grid, azimuth_deg)?;
    let rise = horizon_rise_vec(grid, v, opts)?;
    let scale = (v[0] * v[0] + v[1] * v[1]).sqrt() * grid.res();
    Ok(rise.into_iter().map(|h| if h == f64::NEG_INFINITY { -90.0 } else { (h / scale).atan().to_degrees() }).collect())
}

/// The horizon rise `Hs` per cell for a direction vector `(dx, dy)` given directly (exact for those
/// floats; -inf where nothing lies ahead, NaN on no-data). For tests and exact comparisons.
#[doc(hidden)]
pub fn horizon_rise_vec(grid: &Grid, v: [f64; 2], opts: &HorizonOptions) -> Result<Vec<f64>, Error> {
    let [dx, dy] = v;
    if !(dx.is_finite() && dy.is_finite()) || (dx == 0.0 && dy == 0.0) {
        return Err(Error::InvalidSun);
    }
    if opts.target_height.is_nan() || opts.target_height < 0.0 {
        return Err(Error::InvalidTargetHeight);
    }
    let k = if opts.curvature { (1.0 - opts.refraction) / (2.0 * opts.earth_radius) } else { 0.0 };
    if k < 0.0 || !k.is_finite() {
        return Err(Error::UnsupportedRefraction);
    }
    Ok(H::new(grid, dx, dy, k, opts.target_height).run())
}

struct H<'a> {
    grid: &'a Grid<'a>,
    rows: i64,
    cols: i64,
    dx: f64,
    dy: f64,
    th: f64,
    res: f64,
    l2: f64,
    curv: f64,
    w: f64,
    e_l: f64,
    e_a: f64,
    s_loc: f64,
    stencil: Vec<(i64, i64)>,
    zmax: f64,
    amax: f64,
    check: bool,
    /// test hook (VIEWSHED_HORIZON_FALLBACK_ALL): every cell through the exact fallback
    force_fallback: bool,
    /// test hook (VIEWSHED_HORIZON_EXACT_ALL, with the above): the fallback's rational branch for all
    force_exact: bool,
}

/// One patch in a strip: its corners' (a, z) sorted by a, a_min (rounded down), position.
struct Ent {
    amin: f64,
    pts: [(f64, f64); 4],
    r0: i32,
    c0: i32,
}

impl<'a> H<'a> {
    fn new(grid: &'a Grid<'a>, dx: f64, dy: f64, k: f64, th: f64) -> H<'a> {
        let res = grid.res();
        let l2 = dx * dx + dy * dy;
        let (rows, cols) = (grid.rows as i64, grid.cols as i64);
        let span = (rows + cols + 4) as f64 * (dx.abs() + dy.abs());
        let zmax = grid.z.iter().filter(|v| !v.is_nan()).fold(0.0f64, |m, v| m.max(v.abs())) + th;
        let mut h = H {
            grid,
            rows,
            cols,
            dx,
            dy,
            th,
            res,
            l2,
            curv: k * res * res * l2,
            w: 2f64.powi(l2.sqrt().log2().round() as i32),
            e_l: 4.0 * ERR * span + 1e-300,
            e_a: 4.0 * ERR * span + 1e-300,
            s_loc: C0 + (dx.abs() + dy.abs()) / l2 * (1.0 + 1e-9) + 1e-9,
            stencil: Vec::new(),
            zmax,
            amax: span,
            check: std::env::var("VIEWSHED_HORIZON_CHECK").is_ok(),
            force_fallback: std::env::var("VIEWSHED_HORIZON_FALLBACK_ALL").is_ok(),
            force_exact: std::env::var("VIEWSHED_HORIZON_EXACT_ALL").is_ok(),
        };
        h.stencil = h.dda(0, 0, h.s_loc, false);
        h
    }

    #[inline]
    fn at(&self, r: i64, c: i64) -> f64 {
        self.grid.at(r.clamp(0, self.rows - 1) as usize, c.clamp(0, self.cols - 1) as usize)
    }
    #[inline]
    fn corners(&self, r0: i64, c0: i64) -> [f64; 4] {
        [self.at(r0, c0), self.at(r0, c0 + 1), self.at(r0 + 1, c0), self.at(r0 + 1, c0 + 1)]
    }
    #[inline]
    fn inside(&self, r0: i64, c0: i64) -> bool {
        r0 >= -1 && c0 >= -1 && r0 < self.rows && c0 < self.cols
    }
    #[inline]
    fn a_of(&self, u: f64, v: f64) -> f64 {
        u * self.dx + v * self.dy
    }
    #[inline]
    fn l_of(&self, u: f64, v: f64) -> f64 {
        -u * self.dy + v * self.dx
    }
    #[inline]
    fn tol(&self, h: f64) -> f64 {
        1e-9 * self.res + 1e-12 * h.abs()
    }

    /// Patch offsets (relative to T when `rel`, else absolute from (r, c)) the exact-order DDA visits
    /// from vertex (r, c) up to smax: both side cells at near-vertex crossings, the cell across on
    /// grid-line rays (as shadow.rs). Stops when leaving the domain if smax is infinite.
    fn dda(&self, r: i64, c: i64, smax: f64, bounded: bool) -> Vec<(i64, i64)> {
        let (dx, dy) = (self.dx, self.dy);
        let mut out = Vec::new();
        let (sx, sy) = (if dx > 0.0 { 1 } else { -1 }, if dy > 0.0 { 1 } else { -1 });
        let mut cx = if dx > 0.0 { c } else { c - 1 };
        let mut cy = if dy > 0.0 { r } else { r - 1 };
        let tdx = if dx != 0.0 { (1.0 / dx).abs() } else { f64::INFINITY };
        let tdy = if dy != 0.0 { (1.0 / dy).abs() } else { f64::INFINITY };
        let (mut tx, mut ty) = (tdx, tdy);
        let mut t = 0.0f64;
        while t < smax && (!bounded || (cx >= -2 && cy >= -2 && cx <= self.cols && cy <= self.rows)) {
            out.push((cy, cx));
            if dx == 0.0 {
                out.push((cy, cx + 1));
            }
            if dy == 0.0 {
                out.push((cy + 1, cx));
            }
            let near = (tx - ty).abs() <= 1e-9 * tx.min(ty).max(1.0);
            if near && tx.min(ty) < smax {
                out.push((cy, cx + sx));
                out.push((cy + sy, cx));
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
        out
    }

    /// The patch's stretch of T's ray: (s1, s2) in floats (each end one division: rel. error U),
    /// or None if T's ray certainly misses it.
    fn stretch(&self, r: i64, c: i64, r0: i64, c0: i64) -> Option<(f64, f64)> {
        let (mut s1, mut s2) = (0.0f64, f64::INFINITY);
        for (p0, d, k0) in [(c as f64, self.dx, c0 as f64), (r as f64, self.dy, r0 as f64)] {
            if d == 0.0 {
                if !(k0 <= p0 && p0 <= k0 + 1.0) {
                    return None;
                }
                continue;
            }
            let (x, y) = ((k0 - p0) / d, (k0 + 1.0 - p0) / d);
            s1 = s1.max(x.min(y));
            s2 = s2.min(x.max(y));
        }
        // s2 <= 0: the ray only touches the patch at T (an own patch behind the ray: its far bound is
        // 0 / d, exactly 0 in floats); nothing at s > 0
        if s2 <= 0.0 {
            return None;
        }
        let es = 4.0 * U * s1.abs().max(s2.abs()) + 1e-300;
        if s2 < s1 - es {
            return None;
        }
        Some((s1, s2))
    }

    /// Certified interval [v_lo, v_hi] for the sup of the quotient over patch (r0, c0)'s stretch of
    /// T's ray (rigour note sec. 2, local form): v_hi >= the sup; v_lo <= the sup, or -inf (no data,
    /// no stretch, or a near-degenerate stretch whose existence isn't certain). All values are
    /// evaluated in the patch's own coordinates (well conditioned at any distance from T):
    /// f(s) = N(s) / s, N(s) = B(u', v') - z_T - th - curv s^2, u' = (c - c0) + s dx, v' = (r - r0) + s dy.
    /// f = A s + B + C / s is convex or concave on s > 0: its sup over an interval is at most the
    /// larger of the endpoint values and the tangent bound f(x) + f'(x) (y - x) at any x (valid when
    /// concave; when convex the endpoints already bound it), so the sign of C is never needed.
    fn patch_iv(&self, r: i64, c: i64, r0: i64, c0: i64) -> (f64, f64) {
        const NONE: (f64, f64) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        if !self.inside(r0, c0) {
            return NONE;
        }
        let zc = self.corners(r0, c0);
        if zc.iter().any(|v| v.is_nan()) {
            return NONE;
        }
        let Some((s1, s2)) = self.stretch(r, c, r0, c0) else { return NONE };
        let es = 4.0 * U * s1.abs().max(s2.abs()) + 1e-300;
        let (dx, dy) = (self.dx, self.dy);
        let [z00, z01, z10, z11] = zc;
        let (g01, g10, e3) = (z01 - z00, z10 - z00, z00 - z01 - z10 + z11);
        let zs = z00.abs() + z01.abs() + z10.abs() + z11.abs();
        // bounds on |dB/du'|, |dB/dv'| for u', v' in [-1, 2] (positions slightly outside by rounding)
        let gb = g01.abs() + g10.abs() + 2.0 * e3.abs() + ERR * zs;
        let (u0, v0) = ((c - c0) as f64, (r - r0) as f64);
        let zt = self.at(r, c);
        let base = zt + self.th;
        let own = (r0 == r || r0 == r - 1) && (c0 == c || c0 == c - 1);
        // N(s) and its error bound, evaluated locally
        let n_at = |s: f64| -> (f64, f64) {
            let (up_, vp) = (u0 + s * dx, v0 + s * dy);
            let b = z00 + g01 * up_ + g10 * vp + e3 * up_ * vp;
            let cs = self.curv * s * s;
            let n = b - base - cs;
            let epos = ERR * (u0.abs() + (s * dx).abs() + v0.abs() + (s * dy).abs());
            let e = gb * epos + ERR * (zs * 4.0 + b.abs() + base.abs() + cs.abs() + zt.abs());
            (n, e)
        };
        let f_at = |s: f64| -> (f64, f64) {
            let (n, e) = n_at(s);
            let f = n / s;
            (f, e / s + ERR * f.abs())
        };
        // f'(s) = (N'(s) s - N(s)) / s^2 with N'(s) = (g01 + e3 v') dx + (g10 + e3 u') dy - 2 curv s
        let fp_at = |s: f64| -> (f64, f64) {
            let (up_, vp) = (u0 + s * dx, v0 + s * dy);
            let np = (g01 + e3 * vp) * dx + (g10 + e3 * up_) * dy - 2.0 * self.curv * s;
            let enp = ERR * (gb * (dx.abs() + dy.abs()) * (1.0 + up_.abs() + vp.abs()) + self.curv * s.abs());
            let (n, en) = n_at(s);
            let fp = (np * s - n) / (s * s);
            (fp, (enp * s + en) / (s * s) + ERR * fp.abs())
        };
        // the slope at T on an own patch (the s -> 0 limit when th = 0): u0, v0 in {0, 1}
        let b0 = g01 * dx + g10 * dy + e3 * (u0 * dy + v0 * dx);
        let eb0 = ERR * zs * (dx.abs() + dy.abs()) * 3.0;
        let lim0 = own && self.th == 0.0;
        if lim0 && s1 == 0.0 {
            // T's own patch, th = 0: N(s) = (A s + b0) s exactly (C = 0), so f = b0 + A s is linear on
            // (0, s2]: sup = max(b0, b0 + A s2), evaluated directly (N(s) / s near s = 0 would divide a
            // rounding error by s)
            let al = e3 * dx * dy - self.curv;
            let ea = ERR * (e3.abs() * (dx * dy).abs() + self.curv);
            let hi_e = s2 + es;
            let vhi = up((b0 + eb0).max(b0 + al * hi_e + eb0 + ea * hi_e + ERR * (al * hi_e).abs()));
            if s2 <= 4.0 * es {
                return (f64::NEG_INFINITY, vhi);
            }
            let ihi = s2 - es;
            let vlo = down((b0 - eb0).max(b0 + al * ihi - eb0 - ea * ihi - ERR * (al * ihi).abs()));
            return (vlo, vhi);
        }
        // a near-optimal tangent point from the local quadratic N(s_m + t) = al t^2 + be t + ga:
        // f' = 0 at s* = sqrt(s_m^2 - (be s_m - ga) / al)
        let tangent_x = |lo: f64, hi: f64| -> f64 {
            let sm = 0.5 * (lo + hi);
            let al = e3 * dx * dy - self.curv;
            if sm > 0.0 && al != 0.0 {
                let (ga, _) = n_at(sm);
                let (up_, vp) = (u0 + sm * dx, v0 + sm * dy);
                let be = (g01 + e3 * vp) * dx + (g10 + e3 * up_) * dy - 2.0 * self.curv * sm;
                let d = sm * sm - (be * sm - ga) / al;
                if d > 0.0 {
                    let x = d.sqrt();
                    if x > lo && x < hi {
                        return x;
                    }
                }
            }
            if lo > 0.0 { sm } else { hi }
        };
        // upper bound over the enlarged stretch (a superset)
        let (lo_e, hi_e) = ((s1 - es).max(0.0), s2 + es);
        let (fh, eh) = f_at(hi_e);
        let mut vhi = up(fh + eh);
        if lo_e > 0.0 {
            let (fl, el) = f_at(lo_e);
            vhi = vhi.max(up(fl + el));
        } else if lim0 {
            vhi = vhi.max(up(b0 + eb0));
        }
        // an interior maximum needs f' > 0 at the left end and f' < 0 at the right end; when f' is
        // certainly > 0 at the right end or certainly < 0 at the left end, the endpoints bound the
        // sup whether f is convex or concave (f' monotone either way)
        let (fph, eph) = fp_at(hi_e);
        let left_neg = lo_e > 0.0 && {
            let (fpl, epl) = fp_at(lo_e);
            fpl + epl < 0.0
        };
        if !(fph - eph > 0.0 || left_neg) {
            // f = A s + B + C / s is convex or concave on the whole stretch (f'' = 2 C / s^3). Convex:
            // the endpoint values above bound the sup. Concave: f lies below each of its tangent
            // lines, so both bounds below hold and the smaller is used (the reach bound alone is
            // loose when tangent_x misses the optimum, e.g. nearly planar patches: 7e-5 seen)
            let mut interior = f64::INFINITY;
            let x = tangent_x(lo_e, hi_e);
            if x > 0.0 {
                let (fx, ex) = f_at(x);
                let (fpx, epx) = fp_at(x);
                let reach = (fpx.abs() + epx) * (x - lo_e).max(hi_e - x);
                interior = up(fx + ex + reach);
            }
            if lo_e > 0.0 {
                // tangents at both ends, errors in the safe direction: f(s) <= f1 + m1 (s - lo_e) and
                // f(s) <= f2 + m2 (s - hi_e) on [lo_e, hi_e]; here m1 >= 0 >= m2 (this branch)
                let (fl, el) = f_at(lo_e);
                let (fpl, epl) = fp_at(lo_e);
                let (f1, m1, f2, m2) = (fl + el, fpl + epl, fh + eh, fph - eph);
                let w = hi_e - lo_e;
                // max over [0, w] of min(f1 + m1 t, f2 + m2 (t - w)): at the crossing, clamped
                let best = if m1 - m2 > 0.0 {
                    let t = ((f2 - m2 * w - f1) / (m1 - m2)).clamp(0.0, w);
                    (f1 + m1 * t).min(f2 + m2 * (t - w))
                } else {
                    f1.min(f2)
                };
                // rounding of the lines' values, and of t: min of the lines is (m1 - m2)-Lipschitz
                // in t, and |t error| (m1 - m2) <= 4U (|f1| + |f2| + |m2| w) + 4U (m1 - m2) w
                let slack = ERR * (f1.abs() + f2.abs() + (m1.abs() + m2.abs()) * w);
                interior = interior.min(up(best + slack));
            }
            if interior < f64::INFINITY {
                vhi = vhi.max(interior);
            }
        }
        // lower bound at points certainly inside the stretch
        if s2 - s1 <= 4.0 * es {
            return (f64::NEG_INFINITY, vhi); // near-degenerate: existence not certain
        }
        let (ilo, ihi) = (s1 + es, s2 - es);
        let (fh, eh) = f_at(ihi);
        let mut vlo = down(fh - eh);
        if s1 > 0.0 {
            let (fl, el) = f_at(ilo);
            vlo = vlo.max(down(fl - el));
        } else if lim0 {
            vlo = vlo.max(down(b0 - eb0)); // s1 = 0 exactly (own patch): the limit belongs to the sup
        }
        let x = tangent_x(ilo, ihi);
        if x > ilo && x < ihi {
            let (fx, ex) = f_at(x);
            vlo = vlo.max(down(fx - ex));
        }
        (vlo, vhi)
    }

    fn run(&self) -> Vec<f64> {
        let (rows, cols) = (self.rows, self.cols);
        let n = (rows * cols) as usize;
        let (mut lmin, mut lmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for (u, v) in [(-1.0, -1.0), (cols as f64, -1.0), (-1.0, rows as f64), (cols as f64, rows as f64)] {
            lmin = lmin.min(self.l_of(u, v));
            lmax = lmax.max(self.l_of(u, v));
        }
        let smin = ((lmin - self.e_l) / self.w).floor() as i64 - 1;
        let ns = (((lmax + self.e_l) / self.w).floor() as i64 - smin + 2) as usize;
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
        let decided: Vec<(u32, f64, bool)> = (0..ns)
            .into_par_iter()
            .filter(|&s| tstart[s + 1] > tstart[s])
            .flat_map_iter(|s| self.strip(s as i64 + smin, &tlist[tstart[s]..tstart[s + 1]]).into_iter())
            .collect();
        let mut out = vec![f64::NAN; n];
        let mut redo = Vec::new();
        for (i, h, ok) in decided {
            if ok {
                out[i as usize] = h;
            } else {
                redo.push(i);
            }
        }
        if std::env::var("VIEWSHED_HORIZON_STATS").is_ok() {
            eprintln!("horizon: {} of {} cells to the exact fallback", redo.len(), n);
        }
        let exact: Vec<(u32, f64)> = redo
            .par_iter()
            .map(|&i| {
                let (r, c) = ((i as i64) / cols, (i as i64) % cols);
                (i, self.fallback(r, c))
            })
            .collect();
        for (i, h) in exact {
            out[i as usize] = h;
        }
        out
    }

    /// One strip: (cell, horizon rise, certified) for its targets with data.
    fn strip(&self, j: i64, targets: &[u32]) -> Vec<(u32, f64, bool)> {
        let (rows, cols, dx, dy) = (self.rows, self.cols, self.dx, self.dy);
        let (l0, l1) = (j as f64 * self.w, (j + 1) as f64 * self.w);
        let mut ents: Vec<Ent> = Vec::new();
        let mut push = |r0: i64, c0: i64| {
            if !self.inside(r0, c0) {
                return;
            }
            let zc = self.corners(r0, c0);
            if zc.iter().any(|v| v.is_nan()) {
                return;
            }
            let mut pts = [(0.0, 0.0); 4];
            let (mut lo, mut hi, mut am) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY);
            for (k, (dr, dc)) in [(0i64, 0i64), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let (u, v) = ((c0 + dc) as f64, (r0 + dr) as f64);
                let l = self.l_of(u, v);
                lo = lo.min(l);
                hi = hi.max(l);
                let a = self.a_of(u, v);
                am = am.min(a);
                pts[k] = (a, zc[k]);
            }
            if up(hi + self.e_l) >= l0 && down(lo - self.e_l) <= l1 {
                pts.sort_by(|x, y| x.0.total_cmp(&y.0));
                ents.push(Ent { amin: down(am - self.e_a), pts, r0: r0 as i32, c0: c0 as i32 });
            }
        };
        // patches meeting the widened slab: walk the major axis (one cell of padding: the float
        // error of the row/column range; the exact lateral filter decides membership)
        if dx.abs() >= dy.abs() {
            for c0 in -1..cols {
                let (u0, u1) = (c0 as f64, (c0 + 1) as f64);
                let vs = [(l0 + u0 * dy) / dx, (l0 + u1 * dy) / dx, (l1 + u0 * dy) / dx, (l1 + u1 * dy) / dx];
                let vlo = vs.iter().cloned().fold(f64::INFINITY, f64::min);
                let vhi = vs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for r0 in ((vlo.floor() as i64) - 1).max(-1)..=((vhi.floor() as i64) + 1).min(rows - 1) {
                    push(r0, c0);
                }
            }
        } else {
            for r0 in -1..rows {
                let (v0, v1) = (r0 as f64, (r0 + 1) as f64);
                let us = [(v0 * dx - l0) / dy, (v1 * dx - l0) / dy, (v0 * dx - l1) / dy, (v1 * dx - l1) / dy];
                let ulo = us.iter().cloned().fold(f64::INFINITY, f64::min);
                let uhi = us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for c0 in ((ulo.floor() as i64) - 1).max(-1)..=((uhi.floor() as i64) + 1).min(cols - 1) {
                    push(r0, c0);
                }
            }
        }
        ents.sort_unstable_by(|x, y| y.amin.total_cmp(&x.amin));
        let m = ents.len();
        let p2 = m.next_power_of_two().max(1);
        // hull tree: node -> range of hull points (a ascending, slopes decreasing)
        let mut hpts: Vec<(f64, f64)> = Vec::with_capacity(m * 8);
        let mut hr = vec![(0usize, 0usize); 2 * p2];
        // dd[node]: how far any corner point under the node can lie above the node's float hull
        // envelope (measured with an outward error term at each merge, plus the children's: the
        // envelope is concave, so a child's bound carries up the tree; review 2026-11-24)
        let mut dd = vec![0.0f64; 2 * p2];
        for (i, e) in ents.iter().enumerate() {
            let st = hpts.len();
            upper_hull(&e.pts, &mut hpts);
            hr[p2 + i] = (st, hpts.len());
            dd[p2 + i] = deficit(&e.pts, &hpts[st..]);
        }
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for nd in (1..p2).rev() {
            let (x, y) = (hr[2 * nd], hr[2 * nd + 1]);
            merged.clear();
            let (mut i, mut k) = (x.0, y.0);
            while i < x.1 || k < y.1 {
                if k >= y.1 || (i < x.1 && hpts[i].0 <= hpts[k].0) {
                    merged.push(hpts[i]);
                    i += 1;
                } else {
                    merged.push(hpts[k]);
                    k += 1;
                }
            }
            let st = hpts.len();
            upper_hull(&merged, &mut hpts);
            hr[nd] = (st, hpts.len());
            if hr[nd].1 > hr[nd].0 {
                // + 16 U Zmax: kept vertices are concave only up to the orientation test's rounding,
                // so a child's envelope can exceed this node's by that much between measured points
                dd[nd] = up(deficit(&merged, &hpts[st..]) + dd[2 * nd].max(dd[2 * nd + 1]) + 16.0 * U * self.zmax);
            }
        }
        // targets: a-cut key a_T + C0 L^2 (rounded up), descending
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
        let mut stack: Vec<(usize, usize, usize)> = Vec::new();
        let mut prev = (i64::MIN, i64::MIN);
        for &(q, i) in &qs {
            while jj < m && ents[jj].amin >= q {
                jj += 1;
            }
            let (r, c) = ((i as i64) / cols, (i as i64) % cols);
            let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            let mut arg = (i64::MIN, i64::MIN);
            let take = |r0: i64, c0: i64, lo: &mut f64, hi: &mut f64, arg: &mut (i64, i64)| {
                let (vl, vh) = self.patch_iv(r, c, r0, c0);
                if vl > *lo {
                    *lo = vl;
                    *arg = (r0, c0);
                }
                *hi = hi.max(vh);
            };
            // the local stretch (patches too near for the a-cut)
            for &(dr, dc) in &self.stencil {
                take(r + dr, c + dc, &mut lo, &mut hi, &mut arg);
            }
            // warm start: the previous target's best patch and its neighbours (extra evaluations
            // only: a patch off T's ray gives -inf)
            if prev.0 != i64::MIN {
                for wr in -1..=1 {
                    for wc in -1..=1 {
                        take(prev.0 + wr, prev.1 + wc, &mut lo, &mut hi, &mut arg);
                    }
                }
            }
            if self.check {
                self.check_cover(r, c, &ents[..jj]);
            }
            let zt = self.grid.z[i as usize] + self.th;
            let at = self.a_of(c as f64, r as f64);
            if jj > 0 {
                stack.clear();
                stack.push((1, 0, p2));
                while let Some((nd, nlo, nhi)) = stack.pop() {
                    if nlo >= jj || hr[nd].0 == hr[nd].1 {
                        continue;
                    }
                    if lo > f64::NEG_INFINITY {
                        // lam' with L^2 lam' <= lo (fl(L^2) has rel. error <= 2U)
                        let lp = if lo >= 0.0 { down(lo / (self.l2 * (1.0 + 4.0 * U))) } else { down(lo / (self.l2 * (1.0 - 4.0 * U))) };
                        let bound = up(hull_max(&hpts[hr[nd].0..hr[nd].1], lp) + dd[nd] + self.margin(lp));
                        let rhs = down(zt - lp * at - ERR * (zt.abs() + (lp * at).abs()) - lp.abs() * self.e_a);
                        if bound < rhs {
                            if self.check {
                                self.check_pruned(r, c, &ents, nd, nlo, nhi.min(jj), p2, lo);
                            }
                            continue;
                        }
                    }
                    if nd >= p2 {
                        let e = &ents[nd - p2];
                        take(e.r0 as i64, e.c0 as i64, &mut lo, &mut hi, &mut arg);
                        continue;
                    }
                    let mid = (nlo + nhi) / 2;
                    let (ca, cb) = ((2 * nd, nlo, mid), (2 * nd + 1, mid, nhi));
                    let lp = if lo > f64::NEG_INFINITY { lo / self.l2 } else { -1e300 };
                    let bnd = |n: usize| if hr[n].0 == hr[n].1 { f64::NEG_INFINITY } else { hull_max(&hpts[hr[n].0..hr[n].1], lp) };
                    if bnd(ca.0) >= bnd(cb.0) {
                        stack.push(cb);
                        stack.push(ca);
                    } else {
                        stack.push(ca);
                        stack.push(cb);
                    }
                }
            }
            prev = arg;
            let ok = !self.force_fallback && (hi <= lo + self.tol(lo) || (lo == f64::NEG_INFINITY && hi == f64::NEG_INFINITY));
            // flat ground: an enclosure containing 0 (within the tolerance) reports exactly 0
            let v = if ok && lo < 0.0 && hi >= 0.0 { 0.0 } else { lo };
            out.push((i, v, ok));
        }
        out
    }

    /// Margin for a hull query at slope lp (rigour note sec. 3).
    #[inline]
    fn margin(&self, lp: f64) -> f64 {
        64.0 * U * (self.zmax + lp.abs() * self.amax) + lp.abs() * self.e_a
    }

    /// VIEWSHED_HORIZON_CHECK: every patch under a pruned node has a certified lower bound <= lo
    /// (else pruning dropped a patch that would raise the horizon).
    #[allow(clippy::too_many_arguments)]
    fn check_pruned(&self, r: i64, c: i64, ents: &[Ent], nd: usize, nlo: usize, nhi: usize, p2: usize, lo: f64) {
        let _ = (nd, p2);
        for e in &ents[nlo..nhi] {
            let (vl, _) = self.patch_iv(r, c, e.r0 as i64, e.c0 as i64);
            assert!(vl <= lo + self.tol(lo), "VIEWSHED_HORIZON_CHECK: pruned patch ({}, {}) reaches {vl} > {lo} for ({r}, {c})", e.r0, e.c0);
        }
    }

    /// VIEWSHED_HORIZON_CHECK: every patch with data that T's ray crosses (stretch longer than 1e-9)
    /// is in the local stencil or a qualifying entry of T's strip (the a-cut).
    fn check_cover(&self, r: i64, c: i64, qual: &[Ent]) {
        let local: BTreeSet<(i64, i64)> = self.stencil.iter().map(|&(dr, dc)| (r + dr, c + dc)).collect();
        let q: BTreeSet<(i64, i64)> = qual.iter().map(|e| (e.r0 as i64, e.c0 as i64)).collect();
        for (cy, cx) in self.dda(r, c, f64::INFINITY, true) {
            for a in -1..=1 {
                for b in -1..=1 {
                    let (r0, c0) = (cy + a, cx + b);
                    if !self.inside(r0, c0) || self.corners(r0, c0).iter().any(|v| v.is_nan()) {
                        continue;
                    }
                    if let Some((s1, s2)) = self.stretch(r, c, r0, c0) {
                        if s2 > s1 + 1e-9 {
                            assert!(local.contains(&(r0, c0)) || q.contains(&(r0, c0)),
                                "VIEWSHED_HORIZON_CHECK: patch ({r0}, {c0}) on ({r}, {c})'s ray at s {s1}..{s2} is neither local nor a qualifying entry");
                        }
                    }
                }
            }
        }
    }

    /// Exact fallback (rigour note sec. 4): every patch near T's ray (DDA, widened 3 x 3), its
    /// stretch decided in rationals (positive length), its sup from the closed form in floats.
    fn fallback(&self, r: i64, c: i64) -> f64 {
        let q = |v: f64| BigRational::from_float(v).expect("finite");
        let int = |v: i64| BigRational::from_integer(BigInt::from(v));
        let (dxq, dyq) = (q(self.dx), q(self.dy));
        let mut cand = BTreeSet::new();
        for (cy, cx) in self.dda(r, c, f64::INFINITY, true) {
            for a in -1..=1 {
                for b in -1..=1 {
                    if self.inside(cy + a, cx + b) {
                        cand.insert((cy + a, cx + b));
                    }
                }
            }
        }
        let (mut lo_all, mut hi_all) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut stretches: Vec<(i64, i64, BigRational, BigRational)> = Vec::new();
        for (r0, c0) in cand {
            if self.corners(r0, c0).iter().any(|v| v.is_nan()) {
                continue;
            }
            let mut lo = BigRational::zero();
            let mut hi: Option<BigRational> = None;
            let mut meets = true;
            for (p0, d, k0) in [(int(c), &dxq, int(c0)), (int(r), &dyq, int(r0))] {
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
            // positive length certain (decided in rationals): the certified enclosure of patch_iv,
            // whose lower bound is -inf only for near-degenerate stretches; for those (now known to
            // exist) a lower bound is f at a float point within an ulp of the exact stretch, minus
            // |f'| times that distance
            let (mut vl, vh) = self.patch_iv(r, c, r0, c0);
            if vl == f64::NEG_INFINITY {
                let mid = (&lo + &hi) / int(2);
                let sm = rat_to_f64(&mid);
                vl = self.value_near(r, c, r0, c0, sm);
            }
            lo_all = lo_all.max(vl);
            hi_all = hi_all.max(vh);
            stretches.push((r0, c0, lo, hi));
        }
        if self.force_exact || hi_all > lo_all + self.tol(lo_all) || (lo_all == f64::NEG_INFINITY && hi_all > f64::NEG_INFINITY) {
            // not expected (local errors are ~1e-10 res at most), but certified anyway: every
            // patch's sup in rational arithmetic, bracketed between adjacent floats
            assert!(!self.check || self.force_exact, "VIEWSHED_HORIZON_CHECK: fallback enclosure wider than the tolerance at ({r}, {c}): [{lo_all:e}, {hi_all:e}]");
            let mut best = f64::NEG_INFINITY;
            for (r0, c0, lo, hi) in &stretches {
                best = best.max(self.exact_patch_sup(r, c, *r0, *c0, lo, hi));
            }
            return best;
        }
        if lo_all < 0.0 && hi_all >= 0.0 {
            return 0.0;
        }
        lo_all
    }

    /// The sup of A s + B + C / s over the exact stretch [lo, hi] (lo = 0 only for own patches), in
    /// rationals: endpoint values (the s -> 0 limit B when lo = 0 and C = 0), and the vertex
    /// B - 2 sqrt(A C) when A < 0, C < 0 and C / A lies in (lo^2, hi^2), with sqrt(A C) bracketed by a
    /// float q, q^2 <= A C <= next_up(q)^2. Returns a float lower bound within ~2 ulps of the sup.
    fn exact_patch_sup(&self, r: i64, c: i64, r0: i64, c0: i64, lo: &BigRational, hi: &BigRational) -> f64 {
        let q = |v: f64| BigRational::from_float(v).expect("finite");
        let int = |v: i64| BigRational::from_integer(BigInt::from(v));
        let [z00, z01, z10, z11] = self.corners(r0, c0).map(q);
        let (dx, dy) = (q(self.dx), q(self.dy));
        let e3 = &z00 - &z01 - &z10 + &z11;
        let (u0, v0) = (int(c - c0), int(r - r0));
        let l2 = &dx * &dx + &dy * &dy;
        let curv = q(self.curv / (self.res * self.res * self.l2)) * q(self.res) * q(self.res) * &l2;
        let a = &e3 * &dx * &dy - &curv;
        let b = (&z01 - &z00) * &dx + (&z10 - &z00) * &dy + &e3 * (&u0 * &dy + &v0 * &dx);
        let cc = &z00 + (&z01 - &z00) * &u0 + (&z10 - &z00) * &v0 + &e3 * &u0 * &v0 - q(self.at(r, c)) - q(self.th);
        let f = |s: &BigRational| &a * s + &b + &cc / s;
        let mut vals: Vec<BigRational> = vec![f(hi)];
        if lo.is_positive() {
            vals.push(f(lo));
        } else if cc.is_zero() {
            vals.push(b.clone());
        }
        let mut best = f64::NEG_INFINITY;
        for v in &vals {
            best = best.max(bracket_lo(v));
        }
        if a.is_negative() && cc.is_negative() {
            let m = &cc / &a;
            if lo * lo < m && m < hi * hi {
                let ac = &a * &cc;
                let mut qf = rat_to_f64(&ac).sqrt();
                while q(qf) * q(qf) > ac {
                    qf = next_dn(qf);
                }
                while q(next_upf(qf)) * q(next_upf(qf)) < ac {
                    qf = next_upf(qf);
                }
                // sqrt(AC) in [qf, next_up(qf)]: the vertex value is >= B - 2 next_up(qf)
                let v = &b - int(2) * q(next_upf(qf));
                best = best.max(bracket_lo(&v));
            }
        }
        best
    }

    /// A certified lower bound on f at the exact stretch point nearest to the float s (within one
    /// ulp of s plus the stretch's own rounding): f(s) - error - |f'(s)| (ulp(s) + 4U s).
    fn value_near(&self, r: i64, c: i64, r0: i64, c0: i64, s: f64) -> f64 {
        let [z00, z01, z10, z11] = self.corners(r0, c0);
        let (dx, dy) = (self.dx, self.dy);
        let (g01, g10, e3) = (z01 - z00, z10 - z00, z00 - z01 - z10 + z11);
        let zs = z00.abs() + z01.abs() + z10.abs() + z11.abs();
        let gb = g01.abs() + g10.abs() + 2.0 * e3.abs() + ERR * zs;
        let (u0, v0) = ((c - c0) as f64, (r - r0) as f64);
        let zt = self.at(r, c);
        let base = zt + self.th;
        let (up_, vp) = (u0 + s * dx, v0 + s * dy);
        let b = z00 + g01 * up_ + g10 * vp + e3 * up_ * vp;
        let cs = self.curv * s * s;
        let n = b - base - cs;
        let en = gb * ERR * (u0.abs() + (s * dx).abs() + v0.abs() + (s * dy).abs()) + ERR * (zs * 4.0 + b.abs() + base.abs() + cs.abs() + zt.abs());
        let f = n / s;
        let np = (g01 + e3 * vp) * dx + (g10 + e3 * up_) * dy - 2.0 * self.curv * s;
        let fp = ((np * s - n) / (s * s)).abs() + (gb * (dx.abs() + dy.abs()) * 3.0 + self.curv * s) / s + en / (s * s);
        let ds = 2.0 * U * s.abs() + 4.0 * U * s.abs() + f64::MIN_POSITIVE;
        down(f - en / s - ERR * f.abs() - fp * ds)
    }

}

/// The next float above / below a finite x (f64::next_up/next_down need Rust 1.86; MSRV 1.80).
fn next_upf(x: f64) -> f64 {
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let b = x.to_bits();
    f64::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}
fn next_dn(x: f64) -> f64 {
    -next_upf(-x)
}

/// The largest float <= x.
fn bracket_lo(x: &BigRational) -> f64 {
    let mut f = rat_to_f64(x);
    while f.is_finite() && BigRational::from_float(f).expect("finite") > *x {
        f = next_dn(f);
    }
    f
}

fn rat_to_f64(x: &BigRational) -> f64 {
    use num_traits::ToPrimitive;
    x.to_f64().unwrap_or_else(|| if x.is_negative() { f64::NEG_INFINITY } else { f64::INFINITY })
}

/// Upper bound on how far any of `pts` (sorted by a) lies above the piecewise-linear envelope
/// through `hull` (its vertices, a ascending, spanning the points' a range), with an outward error
/// term for the envelope's evaluation; 0 if none is above.
fn deficit(pts: &[(f64, f64)], hull: &[(f64, f64)]) -> f64 {
    let mut d = 0.0f64;
    let mut k = 0usize;
    for &(a, z) in pts {
        while k + 2 < hull.len() && hull[k + 1].0 < a {
            k += 1;
        }
        let env = if hull.len() == 1 {
            hull[0].1
        } else {
            let (q0, q1) = (hull[k], hull[(k + 1).min(hull.len() - 1)]);
            if q1.0 > q0.0 {
                let t = ((a - q0.0) / (q1.0 - q0.0)).clamp(0.0, 1.0);
                q0.1 + (q1.1 - q0.1) * t
            } else {
                q0.1.max(q1.1)
            }
        };
        let (q0, q1) = (hull[k], hull[(k + 1).min(hull.len() - 1)]);
        let err = 8.0 * U * (z.abs() + q0.1.abs() + q1.1.abs() + (q1.1 - q0.1).abs());
        d = d.max(z - env + err);
    }
    up(d.max(0.0))
}

/// Upper hull (a ascending, edge slopes decreasing) of points sorted by a, appended to `out`.
fn upper_hull(pts: &[(f64, f64)], out: &mut Vec<(f64, f64)>) {
    let start = out.len();
    for &p in pts {
        while out.len() >= start + 2 {
            let (o, a) = (out[out.len() - 2], out[out.len() - 1]);
            if (a.0 - o.0) * (p.1 - o.1) - (a.1 - o.1) * (p.0 - o.0) >= 0.0 {
                out.pop();
            } else {
                break;
            }
        }
        if out.len() > start && out[out.len() - 1].0 == p.0 {
            if out[out.len() - 1].1 >= p.1 {
                continue;
            }
            out.pop();
        }
        out.push(p);
    }
}

/// max over the hull's vertices of z - lam a, found by binary search on the edge slopes (the
/// caller adds a margin for rounding: dropped near-collinear points, non-monotone float slopes).
#[inline]
fn hull_max(h: &[(f64, f64)], lam: f64) -> f64 {
    let (mut lo, mut hi) = (0usize, h.len() - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let s = (h[mid + 1].1 - h[mid].1) / (h[mid + 1].0 - h[mid].0);
        if s > lam {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // the neighbours too: float slopes may be slightly out of order near the optimum
    let mut v = h[lo].1 - lam * h[lo].0;
    if lo > 0 {
        v = v.max(h[lo - 1].1 - lam * h[lo - 1].0);
    }
    if lo + 1 < h.len() {
        v = v.max(h[lo + 1].1 - lam * h[lo + 1].0);
    }
    v
}
