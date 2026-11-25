//! Fresnel clearance of links: how much of the first Fresnel zone the terrain leaves clear.
//!
//! Model: the bilinear surface and sightline of [`crate::intervisible`] (each end a map
//! position plus a height above the surface there, the curvature drop below the chord,
//! no-data patches have no surface). With D the horizontal length of the link and t in
//! [0, 1] the position along it from the first point:
//!
//! ```text
//! gap(t) = sightline(t) - ground(t) - drop * D^2 * t (1 - t)
//! r1(t)  = sqrt(lambda * D * t (1 - t)),   lambda = SPEED_OF_LIGHT / frequency
//! ratio  = inf over t in (0, 1) of gap(t) / r1(t)   (over patches with data)
//! ```
//!
//! 1 = the first zone is clear, 0.6 the usual planning target, 0 grazing, below 0 the
//! line itself is blocked. The ITU-R P.526 diffraction parameter of the worst point is
//! `nu = -sqrt(2) * ratio`. Distances are horizontal and the clearance vertical, the
//! usual planning approximation; the map units must be metres.
//!
//! The ratio is certified to within `1e-9 + 1e-12 |ratio|` of the exact value (the
//! conformance oracle `fresnel_oracle` computes it in rational arithmetic).
//!
//! Method: best-first branch and bound over t. A stretch's lower bound comes from the
//! max pyramid (a bilinear patch never rises above its highest corner), as in
//! `intervisible`; stretches of two cells or less become their patches, each with its
//! gap as a quadratic q in a local parameter and an absolute error bound covering the
//! floats. On a piece of a patch, phi = q / sqrt(w) (w = t (1 - t)) has derivative
//! N / w^1.5 with the cubic N = q' w - q w' / 2: where N certainly keeps one sign the
//! minimum is at an end, else a mean-value bound converges like the piece's length
//! squared. Upper bounds are phi at points well inside a patch. The search stops when
//! the lowest lower bound is within the tolerance of the best point; pieces that can't
//! get there (slivers, a minimum on a grid line next to no-data) are decided in exact
//! rational arithmetic: the pieces' quadratics, the real roots of N isolated with a
//! Sturm sequence.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use rayon::prelude::*;

use crate::bilinear::{ifloor, rat};
use crate::los::{end, end_f, LosOptions, Point};
use crate::ztv::MaxPyramid;
use crate::{Error, Grid};

/// The speed of light in m/s (lambda = SPEED_OF_LIGHT / frequency).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

const U: f64 = f64::EPSILON / 2.0;

static N_EXACT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static N_POPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (links that needed exact arithmetic, search nodes popped) since the last call (benchmarks).
#[doc(hidden)]
pub fn fresnel_stats() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (N_EXACT.swap(0, Relaxed), N_POPS.swap(0, Relaxed))
}

/// A link's Fresnel clearance: the worst ratio of gap to first-zone radius, and where it
/// occurs (`t` in [0, 1] from the first point to the second). With no data under the
/// whole link, `ratio` is +inf and `t` NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clearance {
    pub ratio: f64,
    pub t: f64,
}

/// For each pair, the Fresnel clearance of the link between them at `frequency` (Hz).
/// Heights must be > 0 (the zone's radius is 0 at the ends). Pairs run in parallel.
pub fn fresnel_clearance(grid: &Grid, pairs: &[(Point, Point)], frequency: f64, opts: &LosOptions) -> Result<Vec<Clearance>, Error> {
    if !(frequency > 0.0 && frequency.is_finite()) {
        return Err(Error::InvalidFrequency);
    }
    let pyramid = MaxPyramid::new(grid);
    let drop = opts.drop_factor();
    let lambda = SPEED_OF_LIGHT / frequency;
    pairs
        .par_iter()
        .enumerate()
        .map(|(i, (a, b))| {
            let fa = end_f(grid, a, i)?;
            let fb = end_f(grid, b, i)?;
            if !(a.height > 0.0 && b.height > 0.0) {
                return Err(Error::ZeroPointHeight { pair: i });
            }
            if a.x == b.x && a.y == b.y {
                return Err(Error::CoincidentPoints { pair: i });
            }
            Ok(Link::new(grid, &pyramid, (a, fa), (b, fb), drop, lambda).solve())
        })
        .collect()
}

/// A patch met by a short stretch: gap(tp + s) = g0 + g1 s + g2 s^2 for s in [s_lo, s_hi]
/// (the patch's t-interval, widened beyond the crossings' rounding, clipped to the stretch),
/// within `eg` of the exact gap; `own`: local s-range certainly inside the patch (points for
/// upper bounds), if any.
struct Pat {
    tp: f64,
    g: [f64; 3],
    eg: f64,
    s_lo: f64,
    s_hi: f64,
    own: Option<(f64, f64)>,
}

#[derive(Clone, Copy)]
enum Kind {
    Stretch(f64, f64),
    Piece(usize, f64, f64),
}

struct Node {
    lo: f64,
    kind: Kind,
}

impl PartialEq for Node {
    fn eq(&self, o: &Self) -> bool {
        self.lo.total_cmp(&o.lo) == Ordering::Equal
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Node {
    // BinaryHeap is a max-heap: the lowest lower bound first
    fn cmp(&self, o: &Self) -> Ordering {
        o.lo.total_cmp(&self.lo)
    }
}

struct Link<'a> {
    grid: &'a Grid<'a>,
    pyr: &'a MaxPyramid,
    a: &'a Point,
    b: &'a Point,
    drop: f64,
    lambda: f64,
    fu: f64,
    fv: f64,
    fdu: f64,
    fdv: f64,
    fza: f64,
    fzb: f64,
    fbend: f64,
    /// magnitude of grid coordinates along the link (for rounding bounds), in cells
    coord: f64,
    cells: f64,
    /// 1 / sqrt(lambda D): ratio = phi * scale
    scale: f64,
}

fn w(t: f64) -> f64 {
    t * (1.0 - t)
}

/// Lower bound of g / sqrt(w) for g a lower bound of the gap, w in [wmin, wmax] (floats
/// within a few ulps of the true range).
fn phi_lo(g: f64, wmin: f64, wmax: f64) -> f64 {
    if g >= 0.0 {
        let r = g / (wmax * (1.0 + 8.0 * U)).sqrt();
        r * (1.0 - 4.0 * U)
    } else if wmin > 0.0 {
        let r = g / (wmin * (1.0 - 8.0 * U)).sqrt();
        r * (1.0 + 4.0 * U)
    } else {
        f64::NEG_INFINITY
    }
}

impl<'a> Link<'a> {
    fn new(grid: &'a Grid<'a>, pyr: &'a MaxPyramid, (a, fa): (&'a Point, (f64, f64, f64)), (b, fb): (&'a Point, (f64, f64, f64)), drop: f64, lambda: f64) -> Self {
        let (fdu, fdv) = (fb.0 - fa.0, fb.1 - fa.1);
        let res = grid.res();
        let d = fdu.hypot(fdv) * res;
        Link {
            grid,
            pyr,
            a,
            b,
            drop,
            lambda,
            fu: fa.0,
            fv: fa.1,
            fdu,
            fdv,
            fza: fa.2,
            fzb: fb.2,
            fbend: drop * (fdu * fdu + fdv * fdv) * res * res,
            coord: fa.0.abs() + fa.1.abs() + fdu.abs() + fdv.abs() + 2.0,
            cells: fdu.abs() + fdv.abs(),
            scale: 1.0 / (lambda * d).sqrt(),
        }
    }

    /// Tolerance in phi units for a best value `best` (ratio tolerance 1e-9 + 1e-12 |ratio|).
    fn tol(&self, best: f64) -> f64 {
        1e-9 / self.scale + 1e-12 * best.abs()
    }

    fn solve(&self) -> Clearance {
        if !(self.scale.is_finite() && self.scale > 0.0) || self.cells == 0.0 {
            // the floats can't resolve the link's length: all in exact arithmetic
            let (_, hi, t) = self.exact_min(&[(0.0, 1.0)]);
            return Clearance { ratio: hi * self.exact_scale(), t };
        }
        let mut best = f64::INFINITY;
        let mut best_t = f64::NAN;
        let mut pats: Vec<Pat> = Vec::new();
        let mut exact: Vec<(f64, f64)> = Vec::new();
        let mut heap = BinaryHeap::new();
        let useful = |lo: f64, best: f64| -> bool { lo < f64::INFINITY && !(lo >= best - self.tol(best)) };
        let l0 = self.stretch_lo(0.0, 1.0);
        if useful(l0, best) {
            heap.push(Node { lo: l0, kind: Kind::Stretch(0.0, 1.0) });
        }
        let mut pops = 0usize;
        while let Some(top) = heap.peek() {
            if !useful(top.lo, best) {
                break;
            }
            let node = heap.pop().expect("peeked");
            pops += 1;
            if pops > 1_000_000 {
                // a safety net, not expected: the rest exactly
                exact.push((0.0, 1.0));
                break;
            }
            match node.kind {
                Kind::Stretch(t0, t1) => {
                    if self.cells * (t1 - t0) > 2.0 {
                        let tm = 0.5 * (t0 + t1);
                        for (x, y) in [(t0, tm), (tm, t1)] {
                            let lo = self.stretch_lo(x, y);
                            if useful(lo, best) {
                                heap.push(Node { lo, kind: Kind::Stretch(x, y) });
                            }
                        }
                    } else {
                        for p in self.patches(t0, t1) {
                            let i = pats.len();
                            pats.push(p);
                            let p = &pats[i];
                            if let Some((oa, ob)) = p.own {
                                for s in [0.5 * (oa + ob), oa, ob] {
                                    self.point(p, s, &mut best, &mut best_t);
                                }
                            }
                            let lo = self.piece_lo(p, p.s_lo, p.s_hi);
                            if useful(lo, best) {
                                heap.push(Node { lo, kind: Kind::Piece(i, p.s_lo, p.s_hi) });
                            }
                        }
                    }
                }
                Kind::Piece(i, s0, s1) => {
                    let p = &pats[i];
                    let sm = 0.5 * (s0 + s1);
                    if !(s0 < sm && sm < s1) || s1 - s0 <= 1e-14 * (p.tp.abs() + 1.0) {
                        exact.push((p.tp + s0, p.tp + s1));
                        continue;
                    }
                    for (x, y) in [(s0, sm), (sm, s1)] {
                        self.point(p, 0.5 * (x + y), &mut best, &mut best_t);
                        let lo = self.piece_lo(p, x, y);
                        if useful(lo, best) {
                            heap.push(Node { lo, kind: Kind::Piece(i, x, y) });
                        }
                    }
                }
            }
        }
        N_POPS.fetch_add(pops, std::sync::atomic::Ordering::Relaxed);
        if !exact.is_empty() {
            N_EXACT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let (_, hi, t) = self.exact_min(&exact);
            // hi: phi at a point (within ulps); the exact minimum over those ranges is within
            // ulps below it, everything else is >= best - tol
            if hi < best {
                best = hi;
                best_t = t;
            }
        }
        Clearance { ratio: best * self.scale, t: if best.is_finite() { best_t } else { f64::NAN } }
    }

    /// Certified lower bound of phi over t in [t0, t1] from the max pyramid (as
    /// intervisible's clear_by_bound; its margin, 1e-9 of the magnitudes, is far above the
    /// floats' error; stretches near the optimum are refined to patches anyway).
    fn stretch_lo(&self, t0: f64, t1: f64) -> f64 {
        let (ua, ub) = (self.fu + t0 * self.fdu, self.fu + t1 * self.fdu);
        let (va, vb) = (self.fv + t0 * self.fdv, self.fv + t1 * self.fdv);
        let (c_lo, c_hi) = (ua.min(ub).floor() as i64 - 1, ua.max(ub).floor() as i64 + 2);
        let (r_lo, r_hi) = (va.min(vb).floor() as i64 - 1, va.max(vb).floor() as i64 + 2);
        let zmax = self.pyr.max_in(r_lo, r_hi, c_lo, c_hi);
        if zmax == f64::NEG_INFINITY {
            return f64::INFINITY; // only no-data: no surface
        }
        let chord = |t: f64| self.fza + t * (self.fzb - self.fza);
        let wpk = w(0.5f64.clamp(t0, t1));
        let wend = w(t0).min(w(t1));
        let curv_min = if self.fbend >= 0.0 { -self.fbend * wpk } else { -self.fbend * wend };
        let m = 1e-9 * (self.fza.abs() + self.fzb.abs() + zmax.abs() + self.fbend.abs() + 1.0);
        let g = chord(t0).min(chord(t1)) + curv_min - zmax - m;
        phi_lo(g, wend, wpk)
    }

    /// The patches a short stretch [t0, t1] may meet (as intervisible's float_tier).
    fn patches(&self, t0: f64, t1: f64) -> Vec<Pat> {
        let grid = self.grid;
        let (rows, cols) = (grid.rows as i64, grid.cols as i64);
        // the crossings' parameters are within ~4U coord cells of the exact ones (the float
        // ends and the division): widen by twice that; points for upper bounds stay 64U coord
        // cells inside
        let eps = 8.0 * U * self.coord;
        let inner = 64.0 * U * self.coord;
        let (ua, ub) = (self.fu + t0 * self.fdu, self.fu + t1 * self.fdu);
        let (va, vb) = (self.fv + t0 * self.fdv, self.fv + t1 * self.fdv);
        let fdz = self.fzb - self.fbend - self.fza;
        let mut out = Vec::new();
        for r0 in (va.min(vb).floor() as i64 - 1)..=(va.max(vb).floor() as i64 + 1) {
            for c0 in (ua.min(ub).floor() as i64 - 1)..=(ua.max(ub).floor() as i64 + 1) {
                if r0 < -1 || c0 < -1 || r0 >= rows || c0 >= cols {
                    continue;
                }
                let at = |r: i64, c: i64| grid.z[(r.clamp(0, rows - 1) * cols + c.clamp(0, cols - 1)) as usize];
                let (z00, z01, z10, z11) = (at(r0, c0), at(r0, c0 + 1), at(r0 + 1, c0), at(r0 + 1, c0 + 1));
                if z00.is_nan() || z01.is_nan() || z10.is_nan() || z11.is_nan() {
                    continue;
                }
                // t-interval where the line is within the patch widened (or shrunk) by `e` cells
                let span = |p: f64, d: f64, k: f64, e: f64| -> (f64, f64) {
                    if d == 0.0 {
                        if p >= k - e && p <= k + 1.0 + e {
                            (f64::NEG_INFINITY, f64::INFINITY)
                        } else {
                            (1.0, 0.0)
                        }
                    } else {
                        let (x, y) = ((k - e - p) / d, (k + 1.0 + e - p) / d);
                        (x.min(y), x.max(y))
                    }
                };
                let (ia, ib) = span(self.fu, self.fdu, c0 as f64, eps);
                let (ja, jb) = span(self.fv, self.fdv, r0 as f64, eps);
                let (ta, tb) = (ia.max(ja).max(t0), ib.min(jb).min(t1));
                if !(ta < tb) {
                    continue;
                }
                let (oa1, ob1) = span(self.fu, self.fdu, c0 as f64, -inner);
                let (oa2, ob2) = span(self.fv, self.fdv, r0 as f64, -inner);
                let (oa, ob) = (oa1.max(oa2).max(ta).max(0.0), ob1.min(ob2).min(tb).min(1.0));
                let tp = ta;
                let (d01, d10, e3) = (z01 - z00, z10 - z00, z00 - z01 - z10 + z11);
                let tc = self.fu + tp * self.fdu - c0 as f64;
                let tr = self.fv + tp * self.fdv - r0 as f64;
                let s0 = z00 + d01 * tc + d10 * tr + e3 * tc * tr;
                let s1 = d01 * self.fdu + d10 * self.fdv + e3 * (tc * self.fdv + tr * self.fdu);
                let s2 = e3 * self.fdu * self.fdv;
                let g = [self.fza + tp * fdz - s0 + self.fbend * tp * tp, fdz - s1 + 2.0 * self.fbend * tp, self.fbend - s2];
                // |g(s) - exact gap(tp + s)| over the patch: the ends' floats (positions within
                // ~4U coord cells, heights within ~8U |z|), the coefficients' roundings, and the
                // patch slope times the position errors
                let zabs = z00.abs() + z01.abs() + z10.abs() + z11.abs();
                let slope = d01.abs() + d10.abs() + e3.abs() * (2.0 + tc.abs() + tr.abs());
                let eg = 64.0 * U * (self.fza.abs() + self.fzb.abs() + self.fbend.abs() + zabs * (1.0 + tc.abs()) * (1.0 + tr.abs()) + slope * self.coord);
                out.push(Pat {
                    tp,
                    g,
                    eg,
                    s_lo: 0.0,
                    s_hi: tb - tp,
                    own: if oa < ob && ob > 0.0 && oa < 1.0 { Some((oa - tp, ob - tp)) } else { None },
                });
            }
        }
        out
    }

    /// Upper bound of phi at the point tp + s (s certainly inside the patch): updates best.
    fn point(&self, p: &Pat, s: f64, best: &mut f64, best_t: &mut f64) {
        let Some((oa, ob)) = p.own else { return };
        let s = s.clamp(oa, ob);
        let t = p.tp + s;
        if !(t > 0.0 && t < 1.0) {
            return;
        }
        let [g0, g1, g2] = p.g;
        let q = g0 + s * (g1 + s * g2);
        let e = p.eg + 4.0 * U * (g0.abs() + (s * g1).abs() + (s * s * g2).abs());
        let num = q + e;
        let wt = w(t);
        // the exact point is tp + s; t = fl(tp + s) is within U t of it: w within ~2U t (+ 4U w)
        let werr = 4.0 * U * (t + wt);
        let v = if num >= 0.0 {
            if wt - werr <= 0.0 {
                return;
            }
            num / (wt - werr).sqrt() * (1.0 + 4.0 * U)
        } else {
            num / (wt + werr).sqrt() * (1.0 - 4.0 * U)
        };
        if v < *best {
            *best = v;
            *best_t = t;
        }
    }

    /// Certified lower bound of the exact phi over local s in [s0, s1] of patch p.
    fn piece_lo(&self, p: &Pat, s0: f64, s1: f64) -> f64 {
        let [g0, g1, g2] = p.g;
        let tp = p.tp;
        let (t0, t1) = (tp + s0, tp + s1);
        // w over the piece: concave, min at an end; values within ~4U (t + w) of the exact
        let werr = |t: f64| 4.0 * U * (t.abs() + w(t).abs());
        let wmin = (w(t0) - werr(t0)).min(w(t1) - werr(t1));
        let wpk = w(0.5f64.clamp(t0, t1));
        let wmax = wpk + werr(0.5f64.clamp(t0, t1));
        let qv = |s: f64| g0 + s * (g1 + s * g2);
        let qe = |s: f64| p.eg + 4.0 * U * (g0.abs() + (s * g1).abs() + (s * s * g2).abs());
        // bound 1: the quadratic's minimum over the piece over sqrt(w) (good where gap > 0)
        let mut qmin = qv(s0).min(qv(s1));
        let mut qmine = qe(s0).max(qe(s1));
        if g2 > 0.0 {
            let sv = -g1 / (2.0 * g2);
            if sv > s0 && sv < s1 {
                qmin = qmin.min(qv(sv));
                qmine = qmine.max(qe(sv));
            }
        }
        let b1 = phi_lo(qmin - qmine, wmin, wmax);
        if wmin <= 0.0 {
            return b1;
        }
        // bound 2: phi~ = q~ / sqrt(w), w = w0 + w1 s - s^2 (exact w about tp); phi~' = N / w^1.5
        let (w0, w1) = (w(tp), 1.0 - 2.0 * tp);
        let m = 0.5 * (s0 + s1);
        let h = 0.5 * (s1 - s0);
        let (q, q1, q2) = (qv(m), g1 + 2.0 * g2 * m, 2.0 * g2);
        let (wm, wd) = (w0 + m * (w1 - m), w1 - 2.0 * m);
        let n = q1 * wm - 0.5 * q * wd;
        let nd = q2 * wm + 0.5 * q1 * wd + q;
        let ndd = 1.5 * q2 * wd;
        let nddd = 3.0 * q2.abs();
        let dev = nd.abs() * h + ndd.abs() * h * h / 2.0 + nddd * h * h * h / 6.0;
        let mag = (g0.abs() + g1.abs() + g2.abs() + 1.0) * (w0.abs() + w1.abs() + m.abs() + 1.0) * (1.0 + m.abs()) * (1.0 + m.abs());
        let en = 32.0 * U * mag + (dev + n.abs()) * 1e-12;
        let b2 = if n.abs() > dev + en {
            // phi~ monotone: its minimum is at an end
            let end = |s: f64, t: f64| -> f64 {
                let wt = w(t) - werr(t);
                if wt <= 0.0 {
                    // the end touches t = 0 or 1: phi~ -> +inf there if q > 0
                    return if qv(s) - qe(s) > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
                }
                phi_lo(qv(s) - 4.0 * U * (g0.abs() + (s * g1).abs() + (s * s * g2).abs()), wt, w(t) + werr(t))
            };
            end(s0, t0).min(end(s1, t1))
        } else {
            let phim = q / wm.sqrt();
            phim - phim.abs() * 8.0 * U - (n.abs() + dev + en) / (wmin * wmin.sqrt()) * h * (1.0 + 8.0 * U)
        };
        // the exact phi differs from phi~ by at most eg / sqrt(w)
        let b2 = b2 - p.eg / wmin.sqrt() * (1.0 + 8.0 * U);
        b1.max(b2)
    }

    fn exact_scale(&self) -> f64 {
        let (a, b) = (end(self.grid, self.a, 0).expect("checked end"), end(self.grid, self.b, 0).expect("checked end"));
        let (du, dv) = (&b.u - &a.u, &b.v - &a.v);
        let d2 = (&du * &du + &dv * &dv) * rat(self.grid.res()) * rat(self.grid.res());
        1.0 / (self.lambda * d2.to_f64().unwrap_or(0.0).sqrt()).sqrt()
    }

    /// Exact minimum of phi over the pieces meeting the t-ranges: (lower bound, phi at a point
    /// within ulps of the minimum, that point's t), the bounds within a few ulps of the exact
    /// minimum.
    fn exact_min(&self, ranges: &[(f64, f64)]) -> (f64, f64, f64) {
        let a = end(self.grid, self.a, 0).expect("checked end");
        let b = end(self.grid, self.b, 0).expect("checked end");
        let (du, dv) = (&b.u - &a.u, &b.v - &a.v);
        let res = rat(self.grid.res());
        let bend = rat(self.drop) * (&du * &du + &dv * &dv) * &res * &res;
        let zero = BigRational::zero();
        let one = BigRational::from_integer(BigInt::from(1));
        let two = BigRational::from_integer(BigInt::from(2));
        let mut ts: Vec<BigRational> = vec![zero.clone(), one.clone()];
        for &(t0, t1) in ranges {
            for (p0, d, f0, fd) in [(&a.u, &du, self.fu, self.fdu), (&a.v, &dv, self.fv, self.fdv)] {
                if d.is_zero() {
                    continue;
                }
                let (pa, pb) = (f0 + t0 * fd, f0 + t1 * fd);
                let (k0, k1) = (pa.min(pb).floor() as i64 - 1, pa.max(pb).ceil() as i64 + 1);
                for k in k0..=k1 {
                    let t = (BigRational::from_integer(BigInt::from(k)) - p0) / d;
                    if t > zero && t < one {
                        ts.push(t);
                    }
                }
            }
        }
        ts.sort();
        ts.dedup();
        let (rows, cols) = (self.grid.rows as i64, self.grid.cols as i64);
        let at = |r: i64, c: i64| self.grid.z[(r.clamp(0, rows - 1) * cols + c.clamp(0, cols - 1)) as usize];
        let (mut lo_all, mut hi_all, mut t_all) = (f64::INFINITY, f64::INFINITY, f64::NAN);
        for win in ts.windows(2) {
            let (ta, tb) = (&win[0], &win[1]);
            let (fa, fb) = (ta.to_f64().unwrap_or(0.0), tb.to_f64().unwrap_or(1.0));
            if !ranges.iter().any(|&(t0, t1)| fb >= t0 - 1e-9 && fa <= t1 + 1e-9) {
                continue;
            }
            let mid = (ta + tb) / &two;
            let c0 = ifloor(&(&a.u + &mid * &du));
            let r0 = ifloor(&(&a.v + &mid * &dv));
            let zc = [at(r0, c0), at(r0, c0 + 1), at(r0 + 1, c0), at(r0 + 1, c0 + 1)];
            if zc.iter().any(|v| v.is_nan()) {
                continue;
            }
            let [z00, z01, z10, z11] = zc.map(rat);
            let (cr, rr) = (BigRational::from_integer(BigInt::from(c0)), BigRational::from_integer(BigInt::from(r0)));
            let gap = |t: &BigRational| -> BigRational {
                let tc = &a.u + t * &du - &cr;
                let tr = &a.v + t * &dv - &rr;
                let surf = (&z00 * (&one - &tc) + &z01 * &tc) * (&one - &tr) + (&z10 * (&one - &tc) + &z11 * &tc) * &tr;
                &a.z + t * (&b.z - &a.z) - surf - &bend * t * (&one - t)
            };
            // the gap is exactly quadratic in t: through three points
            let (g0, g1, g2) = (gap(ta), gap(&mid), gap(tb));
            let hh = (tb - ta) / &two;
            let a2 = (&g0 - &two * &g1 + &g2) / (&two * &hh * &hh);
            let a1 = (&g2 - &g0) / (&two * &hh) - &two * &a2 * &mid;
            let a0 = &g1 - &a2 * &mid * &mid - &a1 * &mid;
            let q = vec![a0, a1, a2];
            // N = q' w - q w' / 2, w = t - t^2
            let qd = poly_deriv(&q);
            let wp = vec![zero.clone(), one.clone(), -one.clone()];
            let half_wd = vec![BigRational::new(BigInt::from(1), BigInt::from(2)), -one.clone()];
            let n = poly_sub(&poly_mul(&qd, &wp), &poly_mul(&q, &half_wd));
            let mut cand: Vec<BigRational> = Vec::new();
            for t in [ta, tb] {
                if t > &zero && t < &one {
                    cand.push(t.clone());
                }
            }
            let width = (tb - ta) / BigRational::from_integer(BigInt::from(2).pow(80));
            let nsum: f64 = n.iter().map(|c| c.to_f64().unwrap_or(f64::INFINITY).abs()).sum();
            let mut slack: Vec<f64> = vec![0.0; cand.len()];
            for (u, v) in sturm_roots(&n, ta, tb, &width) {
                // the minimum in [u, v] is >= the smaller end value minus max|phi'| (v - u),
                // |phi'| = |N| / w^1.5 <= sum |N coefficients| / wmin^1.5 (t in [0, 1])
                let wmin = poly_eval(&wp, &u).min(poly_eval(&wp, &v)).to_f64().unwrap_or(0.0);
                let lip = if wmin > 0.0 { nsum / (wmin * wmin.sqrt()) * 2.0 } else { f64::INFINITY };
                let sl = lip * (&v - &u).to_f64().unwrap_or(f64::INFINITY);
                for t in [u, v] {
                    if t > zero && t < one {
                        cand.push(t);
                        slack.push(sl);
                    }
                }
            }
            for (t, sl) in cand.iter().zip(slack) {
                let qt = poly_eval(&q, t);
                let wt = poly_eval(&wp, t);
                // phi = sign(q) sqrt(q^2 / w), to_f64 and sqrt within ~2 ulps
                let r = (&qt * &qt / &wt).to_f64().unwrap_or(f64::INFINITY).sqrt();
                let phi = if qt.is_negative() { -r } else { r };
                let (vl, vh) = (phi - phi.abs() * 8.0 * U - sl, phi + phi.abs() * 8.0 * U);
                lo_all = lo_all.min(vl);
                if vh < hi_all {
                    hi_all = vh;
                    t_all = t.to_f64().unwrap_or(f64::NAN);
                }
            }
        }
        (lo_all, hi_all, t_all)
    }
}

fn poly_trim(mut p: Vec<BigRational>) -> Vec<BigRational> {
    while p.last().is_some_and(|c| c.is_zero()) {
        p.pop();
    }
    p
}

fn poly_mul(p: &[BigRational], q: &[BigRational]) -> Vec<BigRational> {
    if p.is_empty() || q.is_empty() {
        return Vec::new();
    }
    let mut out = vec![BigRational::zero(); p.len() + q.len() - 1];
    for (i, a) in p.iter().enumerate() {
        for (j, b) in q.iter().enumerate() {
            out[i + j] += a * b;
        }
    }
    poly_trim(out)
}

fn poly_sub(p: &[BigRational], q: &[BigRational]) -> Vec<BigRational> {
    let n = p.len().max(q.len());
    let z = BigRational::zero();
    poly_trim((0..n).map(|i| p.get(i).unwrap_or(&z) - q.get(i).unwrap_or(&z)).collect())
}

fn poly_deriv(p: &[BigRational]) -> Vec<BigRational> {
    poly_trim(p.iter().enumerate().skip(1).map(|(i, c)| c * BigRational::from_integer(BigInt::from(i as i64))).collect())
}

fn poly_eval(p: &[BigRational], t: &BigRational) -> BigRational {
    let mut v = BigRational::zero();
    for c in p.iter().rev() {
        v = v * t + c;
    }
    v
}

fn poly_rem(p: &[BigRational], q: &[BigRational]) -> Vec<BigRational> {
    let mut p = p.to_vec();
    let lead = q.last().expect("non-zero divisor").clone();
    while p.len() >= q.len() && !p.is_empty() {
        let f = p.last().expect("non-empty") / &lead;
        let k = p.len() - q.len();
        for (i, b) in q.iter().enumerate() {
            p[k + i] -= &f * b;
        }
        p = poly_trim(p);
    }
    p
}

/// Sign changes of a Sturm sequence at t (zeros skipped).
fn sturm_changes(seq: &[Vec<BigRational>], t: &BigRational) -> usize {
    let mut last: Option<bool> = None;
    let mut n = 0;
    for p in seq {
        let v = poly_eval(p, t);
        if v.is_zero() {
            continue;
        }
        let pos = v.is_positive();
        if last.is_some_and(|l| l != pos) {
            n += 1;
        }
        last = Some(pos);
    }
    n
}

/// Brackets [u, v] of width <= `width`, each holding exactly one distinct real root of p in
/// (lo, hi] (Sturm's theorem, then bisection).
fn sturm_roots(p: &[BigRational], lo: &BigRational, hi: &BigRational, width: &BigRational) -> Vec<(BigRational, BigRational)> {
    if p.len() < 2 {
        return Vec::new();
    }
    let mut seq = vec![p.to_vec(), poly_deriv(p)];
    loop {
        let k = seq.len();
        if seq[k - 1].is_empty() {
            seq.pop();
            break;
        }
        let r = poly_rem(&seq[k - 2], &seq[k - 1]);
        if r.is_empty() {
            break;
        }
        seq.push(r.into_iter().map(|c| -c).collect());
    }
    let two = BigRational::from_integer(BigInt::from(2));
    let mut out = Vec::new();
    let mut stack = vec![(lo.clone(), hi.clone())];
    while let Some((u, v)) = stack.pop() {
        let n = sturm_changes(&seq, &u) - sturm_changes(&seq, &v).min(sturm_changes(&seq, &u));
        if n == 0 {
            continue;
        }
        if n == 1 && &v - &u <= *width {
            out.push((u, v));
            continue;
        }
        let m = (&u + &v) / &two;
        stack.push((u, m.clone()));
        stack.push((m, v));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_ground_closed_form() {
        // flat ground, no curvature: ratio = 2 h / sqrt(lambda D) at the middle
        let z = vec![0.0; 20 * 120];
        let grid = Grid::new(&z, 20, 120, [0.0, 10.0, 0.0, 200.0, 0.0, -10.0]).unwrap();
        let opts = LosOptions { curvature: false, ..LosOptions::default() };
        let a = Point { x: 15.0, y: 105.0, height: 10.0 };
        let b = Point { x: 1015.0, y: 105.0, height: 10.0 };
        let c = fresnel_clearance(&grid, &[(a, b)], 1e9, &opts).unwrap()[0];
        let want = 2.0 * 10.0 / (SPEED_OF_LIGHT / 1e9 * 1000.0f64).sqrt();
        assert!((c.ratio - want).abs() <= 1e-9 + 1e-12 * want, "{} vs {want}", c.ratio);
        assert!((c.t - 0.5).abs() < 1e-3, "{}", c.t);
    }

    #[test]
    fn errors() {
        let z = vec![0.0; 100];
        let grid = Grid::new(&z, 10, 10, [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]).unwrap();
        let o = LosOptions::default();
        let p = |x: f64, y: f64, h: f64| Point { x, y, height: h };
        assert_eq!(fresnel_clearance(&grid, &[(p(1.5, 1.5, 1.0), p(5.5, 5.5, 1.0))], 0.0, &o), Err(Error::InvalidFrequency));
        assert_eq!(fresnel_clearance(&grid, &[(p(1.5, 1.5, 0.0), p(5.5, 5.5, 1.0))], 1e9, &o), Err(Error::ZeroPointHeight { pair: 0 }));
        assert_eq!(fresnel_clearance(&grid, &[(p(1.5, 1.5, 1.0), p(1.5, 1.5, 2.0))], 1e9, &o), Err(Error::CoincidentPoints { pair: 0 }));
        assert_eq!(fresnel_clearance(&grid, &[(p(1.5, 1.5, -1.0), p(5.5, 5.5, 1.0))], 1e9, &o), Err(Error::InvalidPointHeight { pair: 0 }));
    }
}
