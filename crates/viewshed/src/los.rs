//! Exact intervisibility between pairs of points (radio links, sensor pairs).
//!
//! Model: the bilinear surface of [`crate::Surface::Bilinear`]. Each end is a map
//! position plus a height above the surface there (ground from any patch containing
//! the point). Two points see each other when the segment between them is nowhere
//! strictly below the surface along a stretch of positive length, after lowering
//! every point by the curvature drop at its distance from the first end (the drop
//! below the chord, `drop * d_a * d_b`, is the same seen from either end, so the
//! answer doesn't depend on the order of the two points). The segment is cut where it
//! crosses the lines joining cell centres; each piece lies in the patch holding its
//! midpoint, and a patch with a no-data corner has no surface (never blocks).
//!
//! Method: a float filter proves stretches clear (the sightline's lowest point over a
//! stretch, minus a margin far above any rounding error, is above the highest cell
//! height around it: a bilinear patch never rises above its highest corner); the
//! stretches it can't clear are halved down to a couple of patches and then decided
//! in exact rational arithmetic, piece by piece, as the conformance oracle does
//! (`oracle.line_of_sight`). So every answer is the exact one.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use rayon::prelude::*;

use crate::bilinear::{ground_exact, ifloor, rat};
use crate::ztv::MaxPyramid;
use crate::{Error, Grid, MEAN_EARTH_RADIUS};

/// A point: map position and height above the surface there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub height: f64,
}

/// Earth curvature and refraction for [`intervisible`], as in [`crate::Options`].
#[derive(Debug, Clone, PartialEq)]
pub struct LosOptions {
    pub curvature: bool,
    pub refraction: f64,
    pub earth_radius: f64,
}

impl Default for LosOptions {
    fn default() -> Self {
        LosOptions { curvature: true, refraction: 0.13, earth_radius: MEAN_EARTH_RADIUS }
    }
}

impl LosOptions {
    pub(crate) fn drop_factor(&self) -> f64 {
        if self.curvature {
            (1.0 - self.refraction) / (2.0 * self.earth_radius)
        } else {
            0.0
        }
    }
}

/// For each pair, whether the two points see each other (exact, see the module docs).
/// Pairs are decided in parallel.
pub fn intervisible(grid: &Grid, pairs: &[(Point, Point)], opts: &LosOptions) -> Result<Vec<bool>, Error> {
    let pyramid = MaxPyramid::new(grid);
    let drop = opts.drop_factor();
    pairs
        .par_iter()
        .enumerate()
        .map(|(i, (a, b))| {
            let fa = end_f(grid, a, i)?;
            let fb = end_f(grid, b, i)?;
            Ok(visible(grid, &pyramid, (a, fa), (b, fb), drop))
        })
        .collect()
}

/// One end, exact: grid coordinates (u = column, v = row, cell centres at integers) and
/// the eye height (ground + height).
pub(crate) struct End {
    pub(crate) u: BigRational,
    pub(crate) v: BigRational,
    pub(crate) z: BigRational,
}

pub(crate) fn end(grid: &Grid, p: &Point, pair: usize) -> Result<End, Error> {
    if !(p.height >= 0.0 && p.height.is_finite()) {
        return Err(Error::InvalidPointHeight { pair });
    }
    if !(p.x.is_finite() && p.y.is_finite()) {
        return Err(Error::PointOutside { pair });
    }
    let t = grid.transform;
    let half = BigRational::new(BigInt::from(1), BigInt::from(2));
    let u = (rat(p.x) - rat(t[0])) / rat(t[1]) - &half;
    let v = (rat(p.y) - rat(t[3])) / rat(t[5]) - &half;
    let lo = -half.clone();
    let (cmax, rmax) = (rat(grid.cols as f64) - &half, rat(grid.rows as f64) - &half);
    if u < lo || v < lo || u > cmax || v > rmax {
        return Err(Error::PointOutside { pair });
    }
    let ground = ground_exact(grid, &u, &v).ok_or(Error::PointOnNoData { pair })?;
    let z = ground + rat(p.height);
    Ok(End { u, v, z })
}

/// One end in floats (u, v, eye height), with the same validity checks as [`end`]: decided
/// in floats when the point is clearly inside one patch (more than 1e-6 cells from its
/// edges and from the raster's border, far beyond the floats' error), else exactly.
pub(crate) fn end_f(grid: &Grid, p: &Point, pair: usize) -> Result<(f64, f64, f64), Error> {
    const D: f64 = 1e-6;
    if !(p.height >= 0.0 && p.height.is_finite()) {
        return Err(Error::InvalidPointHeight { pair });
    }
    if !(p.x.is_finite() && p.y.is_finite()) {
        return Err(Error::PointOutside { pair });
    }
    let (u, v) = grid.frac(p.x, p.y);
    let (cols, rows) = (grid.cols as f64, grid.rows as f64);
    if u < -0.5 - D || v < -0.5 - D || u > cols - 0.5 + D || v > rows - 0.5 + D {
        return Err(Error::PointOutside { pair });
    }
    let (c0, r0) = (u.floor(), v.floor());
    let (tc, tr) = (u - c0, v - r0);
    let interior = tc > D && tc < 1.0 - D && tr > D && tr < 1.0 - D;
    let inside = u > -0.5 + D && v > -0.5 + D && u < cols - 0.5 - D && v < rows - 0.5 - D;
    if interior && inside {
        let (ri, ci) = (r0 as i64, c0 as i64);
        let (rs, cs) = (grid.rows as i64, grid.cols as i64);
        let at = |r: i64, c: i64| grid.z[(r.clamp(0, rs - 1) * cs + c.clamp(0, cs - 1)) as usize];
        let (z00, z01, z10, z11) = (at(ri, ci), at(ri, ci + 1), at(ri + 1, ci), at(ri + 1, ci + 1));
        if z00.is_nan() || z01.is_nan() || z10.is_nan() || z11.is_nan() {
            return Err(Error::PointOnNoData { pair });
        }
        let ground = (z00 * (1.0 - tc) + z01 * tc) * (1.0 - tr) + (z10 * (1.0 - tc) + z11 * tc) * tr;
        return Ok((u, v, ground + p.height));
    }
    let e = end(grid, p, pair)?;
    let f = |q: &BigRational| q.to_f64().unwrap_or(f64::NAN);
    Ok((f(&e.u), f(&e.v), f(&e.z)))
}

/// The segment in floats for the filters (their margins cover the floats' errors), and
/// its exact form, built only if a stretch needs exact arithmetic.
struct Seg<'p> {
    a: &'p Point,
    b: &'p Point,
    drop: f64,
    exact: std::cell::OnceCell<ExactSeg>,
    fu: f64,
    fv: f64,
    fdu: f64,
    fdv: f64,
    fza: f64,
    fzb: f64,
    fbend: f64,
}

struct ExactSeg {
    ue: BigRational,
    ve: BigRational,
    ze: BigRational,
    du: BigRational,
    dv: BigRational,
    dz: BigRational,
    bend: BigRational,
}

impl Seg<'_> {
    fn exact(&self, grid: &Grid) -> &ExactSeg {
        self.exact.get_or_init(|| {
            // both ends passed the checks already
            let a = end(grid, self.a, 0).expect("checked end");
            let b = end(grid, self.b, 0).expect("checked end");
            let du = &b.u - &a.u;
            let dv = &b.v - &a.v;
            let res = rat(grid.res());
            let bend = rat(self.drop) * (&du * &du + &dv * &dv) * &res * &res;
            let dz = &b.z - &bend - &a.z;
            ExactSeg { ue: a.u, ve: a.v, ze: a.z, du, dv, dz, bend }
        })
    }
}

fn visible(grid: &Grid, pyr: &MaxPyramid, (pa, a): (&Point, (f64, f64, f64)), (pb, b): (&Point, (f64, f64, f64)), drop: f64) -> bool {
    let (fdu, fdv) = (b.0 - a.0, b.1 - a.1);
    let res = grid.res();
    let s = Seg {
        a: pa,
        b: pb,
        drop,
        exact: std::cell::OnceCell::new(),
        fu: a.0,
        fv: a.1,
        fdu,
        fdv,
        fza: a.2,
        fzb: b.2,
        fbend: drop * (fdu * fdu + fdv * fdv) * res * res,
    };
    let cells = s.fdu.abs() + s.fdv.abs();
    // depth-first over [0, 1], left to right, so a blocker near A stops the walk early
    let mut stack = vec![(0.0f64, 1.0f64)];
    while let Some((t0, t1)) = stack.pop() {
        if clear_by_bound(grid, pyr, &s, t0, t1) {
            continue;
        }
        if cells * (t1 - t0) > 2.0 {
            let tm = 0.5 * (t0 + t1);
            stack.push((tm, t1));
            stack.push((t0, tm));
            continue;
        }
        match float_tier(grid, &s, t0, t1) {
            Some(true) => return false,
            Some(false) => {}
            None => {
                if blocked_exact(grid, &s, t0, t1) {
                    return false;
                }
            }
        }
    }
    true
}

/// Certified: the gap (sightline minus surface, curvature included) is positive over
/// t in [t0, t1]. Bilinear patches are bounded by their highest corner, so the highest
/// cell in a box around the stretch (padded by a cell for the floats' errors, and by
/// the corners of the patches it touches) bounds the surface; the sightline's lowest
/// point is the lower of the chord's ends plus the curvature term's minimum. The margin
/// (1e-9 of the magnitudes involved) is many orders above the few roundings in these
/// float expressions.
fn clear_by_bound(grid: &Grid, pyr: &MaxPyramid, s: &Seg, t0: f64, t1: f64) -> bool {
    let (ua, ub) = (s.fu + t0 * s.fdu, s.fu + t1 * s.fdu);
    let (va, vb) = (s.fv + t0 * s.fdv, s.fv + t1 * s.fdv);
    let (c_lo, c_hi) = (ua.min(ub).floor() as i64 - 1, ua.max(ub).floor() as i64 + 2);
    let (r_lo, r_hi) = (va.min(vb).floor() as i64 - 1, va.max(vb).floor() as i64 + 2);
    let zmax = pyr.max_in(r_lo, r_hi, c_lo, c_hi);
    if zmax == f64::NEG_INFINITY {
        return true; // nothing but no-data: no surface
    }
    let _ = grid;
    let chord = |t: f64| s.fza + t * (s.fzb - s.fza);
    let chord_min = chord(t0).min(chord(t1));
    // gap = chord - surface - bend t (1 - t); t (1 - t) is concave with its peak at 1/2
    let w = |t: f64| t * (1.0 - t);
    let curv_min = if s.fbend >= 0.0 {
        -s.fbend * w(0.5f64.clamp(t0, t1))
    } else {
        -s.fbend * w(t0).min(w(t1))
    };
    let m = 1e-9 * (s.fza.abs() + s.fzb.abs() + zmax.abs() + s.fbend.abs() + 1.0);
    chord_min + curv_min - m > zmax
}

/// Certified float decision over t in [t0, t1]: Some(true) if the segment is certainly
/// below the surface somewhere (a point strictly inside a patch with data where the gap
/// is below minus the margin: the exact gap there is negative, so a stretch of positive
/// length is below), Some(false) if every patch the stretch can touch certainly keeps
/// the gap positive, None if undecided (left to exact arithmetic).
///
/// Every patch that may hold part of the stretch is tried: each candidate's t-interval
/// is widened by 1e-7 cells on every side, far beyond the floats' error in the crossing
/// parameters, so each exact piece lies inside its patch's widened interval. Over that
/// interval the gap is the patch's quadratic G0 + G1 s + G2 s^2 (s from its start); a
/// lower bound is its smaller end value if G2 <= 0, else the larger of the tangent bound
/// G0 + min(0, G1) L and the vertex bound G0 - G1^2 / (4 G2). Extending the polynomial a
/// little past the patch only lowers the bound, which stays sound.
fn float_tier(grid: &Grid, s: &Seg, t0: f64, t1: f64) -> Option<bool> {
    const EPS: f64 = 1e-7; // in cells
    const IN: f64 = 1e-6; // strictly inside a patch, in cells
    let (rows, cols) = (grid.rows as i64, grid.cols as i64);
    let (ua, ub) = (s.fu + t0 * s.fdu, s.fu + t1 * s.fdu);
    let (va, vb) = (s.fv + t0 * s.fdv, s.fv + t1 * s.fdv);
    let fdz = s.fzb - s.fbend - s.fza;
    let coord = s.fu.abs() + s.fv.abs() + s.fdu.abs() + s.fdv.abs() + 1.0;
    let mut all_clear = true;
    for r0 in (va.min(vb).floor() as i64 - 1)..=(va.max(vb).floor() as i64 + 1) {
        for c0 in (ua.min(ub).floor() as i64 - 1)..=(ua.max(ub).floor() as i64 + 1) {
            if r0 < -1 || c0 < -1 || r0 >= rows || c0 >= cols {
                continue;
            }
            let at = |r: i64, c: i64| grid.z[(r.clamp(0, rows - 1) * cols + c.clamp(0, cols - 1)) as usize];
            let (z00, z01, z10, z11) = (at(r0, c0), at(r0, c0 + 1), at(r0 + 1, c0), at(r0 + 1, c0 + 1));
            if z00.is_nan() || z01.is_nan() || z10.is_nan() || z11.is_nan() {
                continue; // no surface
            }
            // t-interval where the line is within the widened patch, clipped to [t0, t1]
            let span = |p: f64, d: f64, k: f64| -> (f64, f64) {
                if d == 0.0 {
                    if p >= k - EPS && p <= k + 1.0 + EPS {
                        (f64::NEG_INFINITY, f64::INFINITY)
                    } else {
                        (1.0, 0.0)
                    }
                } else {
                    let (a, b) = ((k - EPS - p) / d, (k + 1.0 + EPS - p) / d);
                    (a.min(b), a.max(b))
                }
            };
            let (ia, ib) = span(s.fu, s.fdu, c0 as f64);
            let (ja, jb) = span(s.fv, s.fdv, r0 as f64);
            let (ta, tb) = (ia.max(ja).max(t0), ib.min(jb).min(t1));
            if ta > tb {
                continue;
            }
            let zabs = z00.abs() + z01.abs() + z10.abs() + z11.abs();
            let m = 1e-8 * (s.fza.abs() + s.fzb.abs() + s.fbend.abs() + zabs + 1.0) + 1e-12 * coord * (zabs + 1.0);
            let (d01, d10, e3) = (z01 - z00, z10 - z00, z00 - z01 - z10 + z11);
            let gap = |t: f64| -> (f64, f64, f64) {
                let tc = s.fu + t * s.fdu - c0 as f64;
                let tr = s.fv + t * s.fdv - r0 as f64;
                let s0 = z00 + d01 * tc + d10 * tr + e3 * tc * tr;
                let s1 = d01 * s.fdu + d10 * s.fdv + e3 * (tc * s.fdv + tr * s.fdu);
                let s2 = e3 * s.fdu * s.fdv;
                (s.fza + t * fdz - s0 + s.fbend * t * t, fdz - s1 + 2.0 * s.fbend * t, s.fbend - s2)
            };
            let (g0, g1, g2) = gap(ta);
            let l = tb - ta;
            let lb = if g2 <= 0.0 {
                g0.min(g0 + g1 * l + g2 * l * l)
            } else {
                let tangent = g0 + g1.min(0.0) * l;
                let vertex = g0 - g1 * g1 / (4.0 * g2);
                if vertex.is_finite() { tangent.max(vertex) } else { tangent }
            };
            if lb > m {
                continue;
            }
            all_clear = false;
            // try to certify a blocker: the lowest point (or the middle), strictly inside
            let tm = if g2 > 0.0 { (ta - g1 / (2.0 * g2)).clamp(ta, tb) } else { 0.5 * (ta + tb) };
            for t in [tm, 0.5 * (ta + tb)] {
                if !(t > 0.0 && t < 1.0) {
                    continue;
                }
                let (tc, tr) = (s.fu + t * s.fdu - c0 as f64, s.fv + t * s.fdv - r0 as f64);
                if tc > IN && tc < 1.0 - IN && tr > IN && tr < 1.0 - IN && gap(t).0 < -m {
                    return Some(true);
                }
            }
        }
    }
    if all_clear { Some(false) } else { None }
}

/// Exact: does the segment dip strictly below the surface anywhere in a piece that
/// overlaps [t0, t1]? (Pieces are cut at the exact grid-line crossings; enumerating the
/// crossings a cell beyond the stretch on each side covers it, and deciding a piece
/// twice, from two neighbouring stretches, is harmless.)
fn blocked_exact(grid: &Grid, s: &Seg, t0: f64, t1: f64) -> bool {
    let e = s.exact(grid);
    let zero = BigRational::zero();
    let one = BigRational::from_integer(BigInt::from(1));
    let mut ts: Vec<BigRational> = vec![zero.clone(), one.clone()];
    for (p0, d, f0, fd) in [(&e.ue, &e.du, s.fu, s.fdu), (&e.ve, &e.dv, s.fv, s.fdv)] {
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
    ts.sort();
    ts.dedup();
    let (lo, hi) = (t0 - 1e-9, t1 + 1e-9);
    for w in ts.windows(2) {
        let (ta, tb) = (&w[0], &w[1]);
        let (fa, fb) = (ta.to_f64().unwrap_or(0.0), tb.to_f64().unwrap_or(1.0));
        if fb < lo || fa > hi {
            continue;
        }
        if let Some(m) = segment_min(grid, e, ta, tb) {
            if m.is_negative() {
                return true;
            }
        }
    }
    false
}

/// Smallest gap on [ta, tb] (None if its patch has no data): the oracle's
/// `_segment_min` in exact rationals.
fn segment_min(grid: &Grid, s: &ExactSeg, ta: &BigRational, tb: &BigRational) -> Option<BigRational> {
    let two = BigRational::from_integer(BigInt::from(2));
    let one = BigRational::from_integer(BigInt::from(1));
    let mid = (ta + tb) / &two;
    let c0 = ifloor(&(&s.ue + &mid * &s.du));
    let r0 = ifloor(&(&s.ve + &mid * &s.dv));
    let (rows, cols) = (grid.rows as i64, grid.cols as i64);
    let at = |r: i64, c: i64| grid.z[(r.clamp(0, rows - 1) * cols + c.clamp(0, cols - 1)) as usize];
    let zc = [at(r0, c0), at(r0, c0 + 1), at(r0 + 1, c0), at(r0 + 1, c0 + 1)];
    if zc.iter().any(|v| v.is_nan()) {
        return None;
    }
    let [z00, z01, z10, z11] = zc.map(rat);
    let cr = BigRational::from_integer(BigInt::from(c0));
    let rr = BigRational::from_integer(BigInt::from(r0));
    let gap = |t: &BigRational| -> BigRational {
        let tc = &s.ue + t * &s.du - &cr;
        let tr = &s.ve + t * &s.dv - &rr;
        let surf = (&z00 * (&one - &tc) + &z01 * &tc) * (&one - &tr) + (&z10 * (&one - &tc) + &z11 * &tc) * &tr;
        &s.ze + t * &s.dz - surf + &s.bend * t * t
    };
    let mut m = gap(ta).min(gap(tb));
    let e3 = &z00 - &z01 - &z10 + &z11;
    let g2 = -(&e3 * &s.du * &s.dv) + &s.bend;
    if g2.is_positive() {
        let tca = &s.ue + ta * &s.du - &cr;
        let trb = &s.ve + ta * &s.dv - &rr;
        let g1 = &s.dz - ((&z01 - &z00) * &s.du + (&z10 - &z00) * &s.dv + &e3 * (&tca * &s.dv + &trb * &s.du)) + &two * &s.bend * ta;
        let sv = -g1 / (&two * &g2);
        if sv.is_positive() && sv < tb - ta {
            m = m.min(gap(&(ta + sv)));
        }
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{viewshed, Options, Surface};

    #[test]
    fn matches_the_bilinear_viewshed_for_targets_on_cell_centres() {
        let n = 31;
        let gt = [0.0, 2.0, 0.0, 62.0, 0.0, -2.0];
        let mut z: Vec<f64> = (0..n * n)
            .map(|i| {
                let (r, c) = ((i / n) as f64, (i % n) as f64);
                (9.0 * (r * 0.37).sin() * (c * 0.29).cos() + 3.0 * ((r + c) * 0.9).sin()).round()
            })
            .collect();
        for &(r, c) in &[(4, 4), (20, 7), (11, 25)] {
            z[r * n + c] = f64::NAN;
        }
        let g = Grid::new(&z, n, n, gt).unwrap();
        for (k, &(ox, oy)) in [(31.0, 31.0), (10.3, 50.7), (2.0, 2.0), (45.0, 13.0)].iter().enumerate() {
            for curvature in [false, true] {
                let (oh, th) = (1.0 + k as f64, [0.0, 1.5][k % 2]);
                let opts = Options { observer_height: oh, target_height: th, surface: Surface::Bilinear, curvature, earth_radius: 900.0, ..Options::default() };
                let seen = viewshed(&g, ox, oy, &opts).unwrap();
                let a = Point { x: ox, y: oy, height: oh };
                let mut pairs = Vec::new();
                let mut want = Vec::new();
                for r in 0..n {
                    for c in 0..n {
                        if z[r * n + c].is_nan() {
                            continue;
                        }
                        let b = Point { x: c as f64 * 2.0 + 1.0, y: 62.0 - r as f64 * 2.0 - 1.0, height: th };
                        // alternate the order: the answer must not depend on it
                        pairs.push(if (r + c) % 2 == 0 { (a, b) } else { (b, a) });
                        want.push(seen[r * n + c]);
                    }
                }
                let lo = LosOptions { curvature, earth_radius: 900.0, ..LosOptions::default() };
                let got = intervisible(&g, &pairs, &lo).unwrap();
                let wrong = got.iter().zip(&want).filter(|(x, y)| x != y).count();
                assert_eq!(wrong, 0, "observer ({ox}, {oy}) curvature {curvature}");
                let vis = want.iter().filter(|&&v| v).count();
                assert!(vis > 20 && vis < want.len(), "{vis} of {}", want.len());
            }
        }
    }

    #[test]
    fn bad_points_name_their_pair() {
        let z = vec![0.0; 25];
        let mut zn = z.clone();
        zn[12] = f64::NAN;
        let g = Grid::new(&z, 5, 5, [0.0, 1.0, 0.0, 5.0, 0.0, -1.0]).unwrap();
        let ok = Point { x: 1.5, y: 1.5, height: 1.0 };
        let out = Point { x: 9.0, y: 1.0, height: 1.0 };
        let neg = Point { x: 1.5, y: 1.5, height: -1.0 };
        let o = LosOptions::default();
        assert_eq!(intervisible(&g, &[(ok, ok), (ok, out)], &o).unwrap_err(), Error::PointOutside { pair: 1 });
        assert_eq!(intervisible(&g, &[(neg, ok)], &o).unwrap_err(), Error::InvalidPointHeight { pair: 0 });
        let gn = Grid::new(&zn, 5, 5, [0.0, 1.0, 0.0, 5.0, 0.0, -1.0]).unwrap();
        let hole = Point { x: 2.5, y: 2.5, height: 1.0 };
        assert_eq!(intervisible(&gn, &[(ok, ok), (ok, ok), (hole, ok)], &o).unwrap_err(), Error::PointOnNoData { pair: 2 });
        assert_eq!(intervisible(&g, &[], &o).unwrap(), Vec::<bool>::new());
    }

    #[test]
    fn a_dip_too_small_for_floats_is_decided_exactly() {
        // 6 x 6 zeros, one cell (3, 3) at 4: patch (2, 2) has corners 0, 0, 0, 4, so along its
        // anti-diagonal the surface is 4 t (1 - t), peaking at 1 in the middle of the patch.
        // A level sightline along that diagonal at height 1 - d dips below it by d there and
        // nowhere else; d = 2^-40 is far inside the float filters' margin.
        let mut z = vec![0.0; 36];
        z[3 * 6 + 3] = 4.0;
        let g = Grid::new(&z, 6, 6, [0.0, 1.0, 0.0, 6.0, 0.0, -1.0]).unwrap();
        let o = LosOptions { curvature: false, ..LosOptions::default() };
        let d = 2f64.powi(-40);
        let at = |u: f64, v: f64, h: f64| Point { x: u + 0.5, y: 6.0 - (v + 0.5), height: h };
        for (h, want) in [(1.0 - d, false), (1.0, true), (1.0 + d, true)] {
            let (a, b) = (at(0.0, 5.0, h), at(5.0, 0.0, h));
            assert_eq!(intervisible(&g, &[(a, b), (b, a)], &o).unwrap(), vec![want, want], "height {h}");
        }
    }
}
