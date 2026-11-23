//! Exact line of sight on the bilinear surface.
//!
//! Model: heights at cell centres, bilinear between them, clamped at the raster
//! border; the observer at its exact point, the eye `observer_height` above the
//! surface there. A cell is visible when the segment from the eye to its centre
//! (plus `target_height`) is nowhere strictly below the surface, after lowering
//! every point by the curvature drop at its distance. No-data never blocks and is
//! never visible. "Exact" means exact for the inputs as given: every f64 taken at
//! its exact value, everything derived from them in rational arithmetic.
//!
//! Method, per target cell: the sightline is cut where it crosses the lines
//! joining cell centres; in each patch the gap (line minus surface) is a quadratic
//! in t. Each segment is evaluated in floats and certified, from rigorous bounds
//! on the rounding error, as certainly blocking or certainly clear. The walk skips
//! whole blocks of patches when a certified lower bound on the gap over the block
//! (a max-pyramid of heights) is positive, stops at the first certainly blocking
//! segment, and tries the previous cell's blocker first. A cell left undecided is
//! settled in exact rational arithmetic on its undecided patches; every other
//! segment is certified clear, so that is the exact answer.
//!
//! Two ways through the cells, both exact: row-major chunks (cheap sightlines;
//! neighbours share blockers and caches), or rings of increasing distance with a
//! horizon of everything already passed (see `Bins`): per thin angular bin, a lower
//! bound on the horizon and a short list of candidate patches, with analytic bounds
//! from the closed form of a bilinear patch's horizon (sigma = alpha d + beta + gamma/d
//! along a ray). A target is decided by its bin, or by its candidates at its own
//! direction; the walk then only covers the stretches near the observer and the
//! target, and any straddling candidate. Hidden-block culling: as each ring starts,
//! every 4 x 4 block of cells whose nearest point lies in it is tested against the
//! bins; a block certainly below them is hidden, and its patches can never be horizon,
//! so they are never prepared or inserted (only surviving terrain is keyed and
//! sorted). Large rasters take the rings, small ones the row-major walk.
//!
//! This follows the conformance suite's `certify.py` and `oracle_numba.py`, which
//! hold the method to the plain exact oracle (`oracle.py`) cell for cell.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use rayon::prelude::*;

use crate::{Error, Grid, Options};

const U: f64 = f64::EPSILON / 2.0; // 2^-53
const INFL: f64 = 1.0 + 64.0 * U;
const G3: f64 = 3.0 * U / (1.0 - 3.0 * U);
const G4: f64 = 4.0 * U / (1.0 - 4.0 * U);
const G12: f64 = 12.0 * U / (1.0 - 12.0 * U);
const SPLIT: f64 = 134_217_729.0; // 2^27 + 1

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Clear,
    Blocking,
    Undecided,
    NoData,
}

/// The floats of one sightline, each with a bound on its distance from the exact value.
#[derive(Clone, Copy)]
struct Cell {
    ue: f64,
    ve: f64,
    ze: f64,
    du: f64,
    dv: f64,
    bend: f64,
    dz: f64,
    th: f64,
    e_ue: f64,
    e_ve: f64,
    e_ze: f64,
    e_du: f64,
    e_dv: f64,
    e_bend: f64,
    e_dz: f64,
    delta: f64,
}

/// Per-observer floats and their exactly measured errors.
struct Setup {
    ue: f64,
    ve: f64,
    ze: f64,
    e_ue: f64,
    e_ve: f64,
    e_ze: f64,
    th: f64,
    drop: f64,
    res: f64,
    // exact values for the fallback
    ue_x: BigRational,
    ve_x: BigRational,
    ze_x: BigRational,
}

fn rat(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite")
}

/// A float >= the non-negative rational q.
fn up(q: &BigRational) -> f64 {
    let v = q.to_f64().unwrap_or(f64::INFINITY);
    if v.is_finite() && rat(v) < *q {
        next_up(v)
    } else {
        v
    }
}

fn next_up(v: f64) -> f64 {
    if v.is_nan() || v == f64::INFINITY {
        return v;
    }
    if v == 0.0 {
        return f64::from_bits(1);
    }
    let b = v.to_bits();
    f64::from_bits(if v > 0.0 { b + 1 } else { b - 1 })
}

fn two_sum_err(a: f64, b: f64, s: f64) -> f64 {
    let bb = s - a;
    (a - (s - bb)) + (b - bb)
}

fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let t = SPLIT * a;
    let ah = t - (t - a);
    let al = a - ah;
    let t = SPLIT * b;
    let bh = t - (t - b);
    let bl = b - bh;
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

fn make_cell(s: &Setup, z_rc: f64, r: i64, c: i64) -> Cell {
    let (ue, ve, ze, th, drop, res) = (s.ue, s.ve, s.ze, s.th, s.drop, s.res);
    let (cf, rf) = (c as f64, r as f64);
    let du = cf - ue;
    let dv = rf - ve;
    let e_du = INFL * (two_sum_err(cf, -ue, du).abs() + s.e_ue);
    let e_dv = INFL * (two_sum_err(rf, -ve, dv).abs() + s.e_ve);
    let bend = drop * (du * du + dv * dv) * res * res;
    let e_s = 2.0 * du.abs() * e_du + e_du * e_du + 2.0 * dv.abs() * e_dv + e_dv * e_dv;
    let e_bend = INFL * (6.0 * U * bend.abs() + drop.abs() * res * res * e_s * (1.0 + 8.0 * U));
    let dz = z_rc - bend + th - ze;
    let e_dz = INFL
        * (3.0 * U * (z_rc.abs() + bend.abs() + th.abs() + ze.abs()) * (1.0 + 4.0 * U)
            + e_bend
            + s.e_ze);
    let mut delta: f64 = 0.0;
    for (p0, d, e_p, e_d, end) in [(ue, du, s.e_ue, e_du, cf), (ve, dv, s.e_ve, e_dv, rf)] {
        if d == 0.0 {
            continue;
        }
        if e_d > 1e-3 * d.abs() {
            // too imprecise to place crossings by, but harmless if it has none
            if no_crossings(p0, d, e_p, e_d, end) {
                continue;
            }
            delta = f64::INFINITY;
            break;
        }
        delta = delta.max(INFL * 1.01 * (3.0 * U + (e_p + e_d) / d.abs()) * (1.0 + 2.0 * U));
    }
    Cell {
        ue,
        ve,
        ze,
        du,
        dv,
        bend,
        dz,
        th,
        e_ue: s.e_ue,
        e_ve: s.e_ve,
        e_ze: s.e_ze,
        e_du,
        e_dv,
        e_bend,
        e_dz,
        delta,
    }
}

/// Whether an axis (float start p0, direction d, errors e_p, e_d; exact end `end`, the
/// target's coordinate) certainly has no crossing for t in (0, 1): the floats list
/// none, and no integer is within reach of the exact coordinate (between the exact
/// start and `end`) except `end` itself, which the sightline reaches at exactly t = 1
/// (or lies on throughout, when the exact direction is 0: then no crossing either).
/// Such an axis adds no error to the crossings' positions. Typical case: an observer
/// a rounding error off a cell centre, and targets in its own row or column (before
/// this check, delta was infinite there: every boundary flagged, the whole sightline
/// in rational arithmetic, ~100x slower on such rasters). With `end` an integer and
/// e_d > 1e-3 |d| forcing |d| below ~1e-7, this holds for every valid input; it is
/// checked anyway, and the floats' own crossing list must agree.
fn no_crossings(p0: f64, d: f64, e_p: f64, e_d: f64, end: f64) -> bool {
    if axis(p0, d).2 != 0 {
        return false;
    }
    let m = (e_p + e_d) * (1.0 + 8.0 * U) + 4.0 * U * (p0.abs() + d.abs() + end.abs() + 1.0);
    let lo = p0.min(p0 + d).min(end) - m;
    let hi = p0.max(p0 + d).max(end) + m;
    let k = lo.ceil();
    k > hi || (k == end && k + 1.0 > hi)
}

// boundary labels
const AX0: u8 = 0;
const AX1: u8 = 1;
const AXU: u8 = 2;
const AXV: u8 = 3;

/// True only if two crossings are certainly equal in exact arithmetic.
fn crossings_equal(
    cell: &Cell,
    mut a_ax: u8,
    mut a_k: i64,
    mut b_ax: u8,
    mut b_k: i64,
    r: i64,
    c: i64,
) -> bool {
    if a_ax > b_ax {
        std::mem::swap(&mut a_ax, &mut b_ax);
        std::mem::swap(&mut a_k, &mut b_k);
    }
    if a_ax == b_ax {
        return a_ax < 2;
    }
    if a_ax == AX0 {
        return false;
    }
    if a_ax == AX1 {
        return b_k == if b_ax == AXU { c } else { r };
    }
    if cell.e_ue != 0.0 || cell.e_ve != 0.0 || cell.e_du != 0.0 || cell.e_dv != 0.0 {
        return false;
    }
    let (ak, bk) = (a_k as f64, b_k as f64);
    let a = ak - cell.ue;
    let b = bk - cell.ve;
    if two_sum_err(ak, -cell.ue, a) != 0.0 || two_sum_err(bk, -cell.ve, b) != 0.0 {
        return false;
    }
    let (p1, e1) = two_prod(a, cell.dv);
    let (p2, e2) = two_prod(b, cell.du);
    p1 == p2 && e1 == e2
}

fn round_half(p: f64) -> f64 {
    (p + 0.5).floor()
}

fn near_observer_flag(cell: &Cell) -> bool {
    for (p0, d, e) in [(cell.ue, cell.du, cell.e_ue), (cell.ve, cell.dv, cell.e_ve)] {
        if d != 0.0
            && e > 0.0
            && (p0 - round_half(p0)).abs() <= e * (1.0 + 4.0 * U) + 2.0 * U * p0.abs()
        {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy)]
struct Patch {
    z00: f64,
    z01: f64,
    z10: f64,
    z11: f64,
    c0: f64,
    r0: f64,
}

fn coord_errors(cell: &Cell, c0: f64, r0: f64, t: f64) -> (f64, f64) {
    let at = t.abs();
    let e_tc = 3.0 * U * (cell.ue.abs() + (t * cell.du).abs() + c0.abs()) * (1.0 + 4.0 * U)
        + cell.e_ue
        + at * cell.e_du;
    let e_tr = 3.0 * U * (cell.ve.abs() + (t * cell.dv).abs() + r0.abs()) * (1.0 + 4.0 * U)
        + cell.e_ve
        + at * cell.e_dv;
    (e_tc, e_tr)
}

/// The float gap at t (the oracle's formula) and a bound on its distance from this
/// patch's exact polynomial at t; also the coordinate error bounds.
fn gap_and_error(cell: &Cell, p: &Patch, t: f64) -> (f64, f64, f64, f64) {
    let (z00, z01, z10, z11) = (p.z00, p.z01, p.z10, p.z11);
    let tc = cell.ue + t * cell.du - p.c0;
    let tr = cell.ve + t * cell.dv - p.r0;
    let bil = (z00 * (1.0 - tc) + z01 * tc) * (1.0 - tr) + (z10 * (1.0 - tc) + z11 * tc) * tr;
    let g = cell.ze + t * cell.dz - bil + cell.bend * t * t;
    let at = t.abs();
    let (e_tc, e_tr) = coord_errors(cell, p.c0, p.r0, t);
    let b_abs = (z00.abs() * (1.0 - tc).abs() + z01.abs() * tc.abs()) * (1.0 - tr).abs()
        + (z10.abs() * (1.0 - tc).abs() + z11.abs() * tc.abs()) * tr.abs();
    let dbc = (z01 - z00).abs() * (1.0 - tr).abs() + (z11 - z10).abs() * tr.abs();
    let dbr = (z10 - z00).abs() * (1.0 - tc).abs() + (z11 - z01).abs() * tc.abs();
    let e3a = (z00 - z01 - z10 + z11).abs();
    let pert = (dbc * e_tc + dbr * e_tr + e3a * e_tc * e_tr) * (1.0 + 8.0 * U);
    let err = INFL
        * (G12 * (cell.ze.abs() + at * cell.dz.abs() + b_abs + cell.bend.abs() * t * t)
            + pert
            + cell.e_ze
            + at * cell.e_dz
            + t * t * cell.e_bend);
    (g, err, e_tc, e_tr)
}

/// Float slope at ta and curvature, with bounds (e_tc, e_tr: coordinate errors at ta).
fn derivatives(cell: &Cell, p: &Patch, ta: f64, e_tc: f64, e_tr: f64) -> (f64, f64, f64, f64) {
    let (z00, z01, z10, z11) = (p.z00, p.z01, p.z10, p.z11);
    let (du, dv, bend) = (cell.du, cell.dv, cell.bend);
    let e3 = z00 - z01 - z10 + z11;
    let g2 = -e3 * du * dv + bend;
    let tca = cell.ue + ta * du - p.c0;
    let trb = cell.ve + ta * dv - p.r0;
    let g1 = cell.dz - ((z01 - z00) * du + (z10 - z00) * dv + e3 * (tca * dv + trb * du))
        + 2.0 * bend * ta;
    let e3a = e3.abs();
    let e_e3 = G3 * (z00.abs() + z01.abs() + z10.abs() + z11.abs());
    let e3m = e3a + e_e3;
    let a01 = (z01 - z00).abs();
    let a10 = (z10 - z00).abs();
    let e_g1 = INFL
        * (G12
            * (cell.dz.abs()
                + a01 * du.abs()
                + a10 * dv.abs()
                + e3a * ((tca * dv).abs() + (trb * du).abs())
                + 2.0 * bend.abs() * ta.abs())
            + cell.e_dz
            + a01 * cell.e_du
            + a10 * cell.e_dv
            + e_e3 * ((tca * dv).abs() + (trb * du).abs())
            + e3m
                * (e_tc * dv.abs()
                    + tca.abs() * cell.e_dv
                    + e_tc * cell.e_dv
                    + e_tr * du.abs()
                    + trb.abs() * cell.e_du
                    + e_tr * cell.e_du)
            + 2.0 * ta.abs() * cell.e_bend)
        * (1.0 + 8.0 * U);
    let e_g2 = INFL
        * (G4 * (e3a * (du * dv).abs() + bend.abs())
            + e_e3 * (du * dv).abs()
            + e3m * (du.abs() * cell.e_dv + dv.abs() * cell.e_du + cell.e_du * cell.e_dv)
            + cell.e_bend)
        * (1.0 + 8.0 * U);
    (g1, e_g1, g2, e_g2)
}

/// True if the exact gap is < 0 somewhere in this patch's exact segment (ends within
/// delta of ta, tb), checked at points certainly inside it.
fn blocking_at(cell: &Cell, p: &Patch, ta: f64, tb: f64, g1: f64, g2: f64) -> bool {
    let delta = cell.delta;
    if !(tb - ta > 8.0 * delta) {
        return false;
    }
    let lo_p = ta + 3.0 * delta;
    let hi_p = tb - 3.0 * delta;
    let mut pts = [lo_p, hi_p, lo_p];
    let mut n = 2;
    if g2 > 0.0 {
        pts[2] = (ta + (-g1 / (2.0 * g2))).max(lo_p).min(hi_p);
        n = 3;
    }
    for &t in &pts[..n] {
        let (g, err, _, _) = gap_and_error(cell, p, t);
        if g + err * (1.0 + 16.0 * U) + 16.0 * U * g.abs() < 0.0 {
            return true;
        }
    }
    false
}

fn certainly_blocking(cell: &Cell, p: &Patch, ta: f64, tb: f64) -> bool {
    if !(tb - ta > 8.0 * cell.delta) {
        return false;
    }
    let (e_tc, e_tr) = coord_errors(cell, p.c0, p.r0, ta);
    let (g1, _, g2, _) = derivatives(cell, p, ta, e_tc, e_tr);
    blocking_at(cell, p, ta, tb, g1, g2)
}

fn corners(grid: &Grid, r0: i64, c0: i64) -> Patch {
    let (rows, cols) = (grid.rows as i64, grid.cols as i64);
    let ra = r0.clamp(0, rows - 1) as usize;
    let rb = (r0 + 1).clamp(0, rows - 1) as usize;
    let ca = c0.clamp(0, cols - 1) as usize;
    let cb = (c0 + 1).clamp(0, cols - 1) as usize;
    Patch {
        z00: grid.at(ra, ca),
        z01: grid.at(ra, cb),
        z10: grid.at(rb, ca),
        z11: grid.at(rb, cb),
        c0: c0 as f64,
        r0: r0 as f64,
    }
}

fn nodata(p: &Patch) -> bool {
    p.z00.is_nan() || p.z01.is_nan() || p.z10.is_nan() || p.z11.is_nan()
}

/// Certify one segment of the float boundary list (patch from its midpoint).
fn segment(cell: &Cell, grid: &Grid, ta: f64, tb: f64, last: bool) -> (State, i64, i64) {
    let mid = (ta + tb) / 2.0;
    let c0 = (cell.ue + mid * cell.du).floor() as i64;
    let r0 = (cell.ve + mid * cell.dv).floor() as i64;
    let delta = cell.delta;
    if !delta.is_finite() {
        return (State::Undecided, r0, c0);
    }
    let (e_tc, e_tr) = coord_errors(cell, c0 as f64, r0 as f64, mid);
    let fu = cell.ue + mid * cell.du - c0 as f64;
    let fv = cell.ve + mid * cell.dv - r0 as f64;
    for (f, d, e_d, p0, e_p0, e_t) in [
        (fu, cell.du, cell.e_du, cell.ue, cell.e_ue, e_tc),
        (fv, cell.dv, cell.e_dv, cell.ve, cell.e_ve, e_tr),
    ] {
        if d == 0.0 && e_d == 0.0 {
            if e_p0 > 0.0
                && (p0 - round_half(p0)).abs() <= e_p0 * (1.0 + 4.0 * U) + 2.0 * U * p0.abs()
            {
                return (State::Undecided, r0, c0);
            }
        } else if f.min(1.0 - f) <= 2.0 * (e_t + d.abs() * delta) {
            return (State::Undecided, r0, c0);
        }
    }
    let p = corners(grid, r0, c0);
    if nodata(&p) {
        return (State::NoData, r0, c0);
    }
    let (g0, e0, e_tc, e_tr) = gap_and_error(cell, &p, ta);
    let (g1, e_g1, g2, e_g2) = derivatives(cell, &p, ta, e_tc, e_tr);
    let (g1_lo, g1_hi, g2_lo, g2_hi) = (g1 - e_g1, g1 + e_g1, g2 - e_g2, g2 + e_g2);
    let g1m = g1_lo.abs().max(g1_hi.abs());
    let g2m = g2_lo.abs().max(g2_hi.abs());
    if last && cell.th == 0.0 {
        // exact gap(1) = 0: gap = (1 - t) K(t), K linear; clear iff K(ta) >= 0 and -gap'(1) >= 0
        let shift = (g1m + 2.0 * g2m * delta) * delta;
        let k_ta = g0 - e0 - shift;
        let slope1_hi = if g2_hi > 0.0 {
            g1_hi + 2.0 * g2_hi * ((1.0 - ta) * (1.0 + U) + delta)
        } else {
            g1_hi + 2.0 * g2_hi * (1.0 - ta - delta).max(0.0) * (1.0 - U)
        };
        let slop = 16.0 * U * (g0.abs() + e0 + shift);
        if k_ta - slop > 0.0 && slope1_hi * (1.0 + 16.0 * U) + 16.0 * U * g1.abs() < 0.0 {
            return (State::Clear, r0, c0);
        }
    } else {
        let lh = ((tb - ta) + delta) * (1.0 + 2.0 * U);
        let mut q = (g1_lo * lh + g2_lo * lh * lh).min(0.0);
        if g2_lo > 0.0 {
            let s0 = -g1_lo / (2.0 * g2_lo);
            if 0.0 < s0 && s0 < lh {
                q = q.min(-(g1_lo * g1_lo) / (4.0 * g2_lo) * (1.0 + 4.0 * U));
            }
        }
        q = q.min(-g1m * delta + g2_lo.min(0.0) * delta * delta);
        let slop = 16.0 * U * (g0.abs() + e0 + q.abs() + g1m * lh + g2m * lh * lh);
        if g0 - e0 + q - slop > 0.0 {
            return (State::Clear, r0, c0);
        }
    }
    if blocking_at(cell, &p, ta, tb, g1, g2) {
        return (State::Blocking, r0, c0);
    }
    (State::Undecided, r0, c0)
}

// --- boundaries -------------------------------------------------------------------

/// One axis's crossings as a stream: (k of the first in increasing t, step, count).
fn axis(p0: f64, d: f64) -> (i64, i64, i64) {
    if d == 0.0 {
        return (0, 1, 0);
    }
    let p1 = p0 + d;
    let (lo, hi) = (p0.min(p1), p0.max(p1));
    let first = lo.floor() as i64 + 1;
    let end = hi.ceil() as i64;
    let n = (end - first).max(0);
    if d > 0.0 {
        (first, 1, n)
    } else {
        (end - 1, -1, n)
    }
}

#[inline]
fn xval(k0: i64, step: i64, i: i64, p0: f64, d: f64) -> f64 {
    ((k0 + i * step) as f64 - p0) / d
}

/// Smallest index >= i whose crossing is > t.
fn first_above(k0: i64, step: i64, n: i64, i: i64, p0: f64, d: f64, t: f64) -> i64 {
    let (mut lo, mut hi) = (i, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xval(k0, step, mid, p0, d) > t {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// The full sorted boundary list with flags (the oracle's order: equal values come
/// 0, 1, u, v). Used for the exact fallback's candidate patches.
fn boundaries(cell: &Cell, r: i64, c: i64, ts: &mut Vec<f64>, fl: &mut Vec<bool>) {
    let (uk, us, un) = axis(cell.ue, cell.du);
    let (vk, vs, vn) = axis(cell.ve, cell.dv);
    let mut raw: Vec<(f64, u8, i64)> = Vec::with_capacity((un + vn + 2) as usize);
    raw.push((0.0, AX0, 0));
    let (mut ui, mut vi, mut one) = (0, 0, false);
    while ui < un || vi < vn || !one {
        let mut best = f64::INFINITY;
        let mut src = (AX1, 0i64);
        if !one {
            best = 1.0;
        }
        if ui < un {
            let t = xval(uk, us, ui, cell.ue, cell.du);
            if t < best {
                best = t;
                src = (AXU, uk + ui * us);
            }
        }
        if vi < vn {
            let t = xval(vk, vs, vi, cell.ve, cell.dv);
            if t < best {
                best = t;
                src = (AXV, vk + vi * vs);
            }
        }
        match src.0 {
            AX1 => one = true,
            AXU => ui += 1,
            _ => vi += 1,
        }
        raw.push((best, src.0, src.1));
    }
    ts.clear();
    fl.clear();
    let lim = 16.0 * cell.delta;
    ts.push(raw[0].0);
    fl.push(false);
    for i in 1..raw.len() {
        let (a, b) = (raw[i - 1], raw[i]);
        if b.0 != a.0 {
            ts.push(b.0);
            fl.push(false);
        }
        if b.0 - a.0 <= lim && !(b.0 == a.0 && crossings_equal(cell, a.1, a.2, b.1, b.2, r, c)) {
            let m = fl.len() - 1;
            fl[m] = true;
            if b.0 != a.0 {
                fl[m - 1] = true;
            }
        }
    }
    if near_observer_flag(cell) {
        fl[0] = true;
    }
    if !cell.delta.is_finite() {
        fl.iter_mut().for_each(|f| *f = true);
    }
}

// --- pyramid ----------------------------------------------------------------------

/// Max-pyramid over patches: patch (r0, c0), r0 in -1..rows, at [r0 + 1, c0 + 1],
/// value the highest of its four (clamped) corners, -inf for no-data; each widened
/// to its 3 x 3 neighbourhood (a float segment's patch can be one off on a sliver).
struct Pyramid {
    data: Vec<f64>,
    offs: Vec<usize>,
    shapes: Vec<(i64, i64)>,
}

impl Pyramid {
    fn new(grid: &Grid) -> Pyramid {
        let (rows, cols) = (grid.rows as i64, grid.cols as i64);
        let (h, w) = ((rows + 1) as usize, (cols + 1) as usize);
        let mut patch = vec![f64::NEG_INFINITY; h * w];
        patch.par_chunks_mut(w).enumerate().for_each(|(pr, row)| {
            for (pc, v) in row.iter_mut().enumerate() {
                let p = corners(grid, pr as i64 - 1, pc as i64 - 1);
                if !nodata(&p) {
                    *v = p.z00.max(p.z01).max(p.z10).max(p.z11);
                }
            }
        });
        let mut lvl = vec![f64::NEG_INFINITY; h * w];
        lvl.par_chunks_mut(w).enumerate().for_each(|(pr, row)| {
            for (pc, v) in row.iter_mut().enumerate() {
                let mut m = f64::NEG_INFINITY;
                for a in pr.saturating_sub(1)..(pr + 2).min(h) {
                    for b in pc.saturating_sub(1)..(pc + 2).min(w) {
                        m = m.max(patch[a * w + b]);
                    }
                }
                *v = m;
            }
        });
        let mut data = lvl.clone();
        let mut offs = vec![0usize];
        let mut shapes = vec![(h as i64, w as i64)];
        let (mut ch, mut cw, mut cur) = (h, w, lvl);
        while ch > 1 || cw > 1 {
            let (nh, nw) = ((ch + 1) / 2, (cw + 1) / 2);
            let mut next = vec![f64::NEG_INFINITY; nh * nw];
            for a in 0..ch {
                for b in 0..cw {
                    let i = (a / 2) * nw + b / 2;
                    next[i] = next[i].max(cur[a * cw + b]);
                }
            }
            offs.push(data.len());
            data.extend_from_slice(&next);
            shapes.push((nh as i64, nw as i64));
            ch = nh;
            cw = nw;
            cur = next;
        }
        Pyramid { data, offs, shapes }
    }
}

// --- the walk ---------------------------------------------------------------------

/// Whether the patch that blocked the previous cell certainly blocks this sightline.
fn reuse_blocks(cell: &Cell, grid: &Grid, r0: i64, c0: i64) -> bool {
    let (mut ta, mut tb) = (0.0f64, 1.0f64);
    for (p0, d, k0, e_p, e_d) in [
        (cell.ue, cell.du, c0, cell.e_ue, cell.e_du),
        (cell.ve, cell.dv, r0, cell.e_ve, cell.e_dv),
    ] {
        if d == 0.0 {
            if !(e_d == 0.0 && e_p == 0.0 && p0.floor() as i64 == k0) {
                return false;
            }
            continue;
        }
        let a = (k0 as f64 - p0) / d;
        let b = ((k0 + 1) as f64 - p0) / d;
        ta = ta.max(a.min(b));
        tb = tb.min(a.max(b));
    }
    let p = corners(grid, r0, c0);
    !nodata(&p) && certainly_blocking(cell, &p, ta, tb)
}

/// Whether the raw boundaries sharing one value (order 1, u, v) are certainly equal.
fn group_ok(cell: &Cell, r: i64, c: i64, members: &[(bool, u8, i64); 3]) -> bool {
    let mut prev: Option<(u8, i64)> = None;
    for &(present, ax, k) in members {
        if !present {
            continue;
        }
        if let Some((pa, pk)) = prev {
            if !crossings_equal(cell, pa, pk, ax, k, r, c) {
                return false;
            }
        }
        prev = Some((ax, k));
    }
    true
}

/// Certified: the sightline at every t in [t1, t2] is above height m, so its gap to
/// any terrain no higher than m is > 0 there.
fn above(cell: &Cell, t1: f64, t2: f64, m: f64) -> bool {
    let (ze, dz) = (cell.ze, cell.dz);
    let bend_lo = cell.bend - cell.e_bend;
    let l1 = ze + t1 * dz - (cell.e_ze + t1.abs() * cell.e_dz + 3.0 * U * (ze.abs() + (t1 * dz).abs()));
    let l2 = ze + t2 * dz - (cell.e_ze + t2.abs() * cell.e_dz + 3.0 * U * (ze.abs() + (t2 * dz).abs()));
    let lo = l1.min(l2);
    let bl = if bend_lo >= 0.0 {
        let t0 = t1.max(0.0);
        bend_lo * t0 * t0 * (1.0 - 4.0 * U)
    } else {
        bend_lo * (t1 * t1).max(t2 * t2) * (1.0 + 4.0 * U)
    };
    let lb = lo - m + bl;
    lb - 16.0 * U * (lo.abs() + m.abs() + bl.abs()) > 0.0
}

/// Certified: the float segment [ta, tb] (consecutive boundaries, so at most one
/// patch long) is clear, whatever the exact order of the crossings near it. Every
/// point of the sightline with t in [ta, tb] is within sqrt(2)/2 cells (plus
/// rounding) of the float midpoint, so in the 3 x 3 patches around the midpoint's
/// float patch, which the pyramid's level 0 bounds; the bilinear surface is never
/// above its patch's highest corner.
fn clear_level0(cell: &Cell, pyr: &Pyramid, ta: f64, tb: f64) -> bool {
    let delta = cell.delta;
    if !delta.is_finite() {
        return false;
    }
    let mid = (ta + tb) / 2.0;
    let pr = (cell.ve + mid * cell.dv).floor() as i64 + 1;
    let pc = (cell.ue + mid * cell.du).floor() as i64 + 1;
    let (h, w) = pyr.shapes[0];
    if pr < 0 || pc < 0 || pr >= h || pc >= w {
        return false;
    }
    let m = pyr.data[pyr.offs[0] + (pr * w + pc) as usize];
    above(cell, ta - delta, tb + delta, m)
}

/// Certified: wherever the sightline (t in [0, 1]) meets the closed patch (r0, c0),
/// its gap to the patch is > 0; or it doesn't meet it. The t range is the float
/// crossings of the patch's faces widened by twice their error bound (delta, which
/// scales with |t| beyond 1), and the patch is never above its highest corner.
fn patch_clear(cell: &Cell, grid: &Grid, r0: i64, c0: i64) -> bool {
    let delta = cell.delta;
    if !delta.is_finite() {
        return false;
    }
    let p = corners(grid, r0, c0);
    if nodata(&p) {
        return false; // the exact routine skips it anyway
    }
    let (mut t1, mut t2) = (0.0f64, 1.0f64);
    for (p0, d, e_p, e_d, k0) in [
        (cell.ue, cell.du, cell.e_ue, cell.e_du, c0 as f64),
        (cell.ve, cell.dv, cell.e_ve, cell.e_dv, r0 as f64),
    ] {
        if d == 0.0 || e_d > 1e-3 * d.abs() {
            // delta doesn't cover this axis: compare the exact coordinate's whole
            // range over t in [0, 1] with the patch's slab
            let m = 2.0 * (e_p + e_d) + 4.0 * U * (p0.abs() + d.abs() + 1.0);
            let (lo, hi) = (p0.min(p0 + d) - m, p0.max(p0 + d) + m);
            if k0 < lo && hi < k0 + 1.0 {
                continue;
            }
            if hi < k0 || lo > k0 + 1.0 {
                return true;
            }
            return false;
        }
        let a = (k0 - p0) / d;
        let b = (k0 + 1.0 - p0) / d;
        let (lo, hi) = (a.min(b), a.max(b));
        t1 = t1.max(lo - 2.0 * delta * lo.abs().max(1.0));
        t2 = t2.min(hi + 2.0 * delta * hi.abs().max(1.0));
    }
    if t1 > t2 {
        return true;
    }
    let m = p.z00.max(p.z01).max(p.z10).max(p.z11);
    above(cell, t1, t2, m)
}

/// One cell: Clear (visible), Blocking (hidden) or Undecided (exact check), and the
/// blocking patch when there is one.
fn walk(
    cell: &Cell,
    grid: &Grid,
    pyr: &Pyramid,
    r: i64,
    c: i64,
    failed: &mut [i64],
    mid: &[(f64, f64)],
    count: &mut usize,
) -> (State, i64, i64) {
    let (ue, ve, du, dv, delta) = (cell.ue, cell.ve, cell.du, cell.dv, cell.delta);
    let (uk, us, un) = axis(ue, du);
    let (vk, vs, vn) = axis(ve, dv);
    let (mut ui, mut vi, mut one) = (0i64, 0i64, false);
    let lim = 16.0 * delta;
    let nlev = pyr.shapes.len();
    failed.iter_mut().for_each(|f| *f = -1);
    let peek = |ui: i64, vi: i64, one: bool| -> f64 {
        let mut best = if one { f64::INFINITY } else { 1.0 };
        if ui < un {
            best = best.min(xval(uk, us, ui, ue, du));
        }
        if vi < vn {
            best = best.min(xval(vk, vs, vi, ve, dv));
        }
        best
    };
    // pyramid level to try first: one coarser after each skip, the finest after
    // each evaluated segment (max-mipmap traversal)
    let mut cur = nlev.saturating_sub(1);
    let mut next_mid = 0usize;
    let mut ta = 0.0f64;
    let mut fa = near_observer_flag(cell) || peek(ui, vi, one) - ta <= lim || !delta.is_finite();
    let mut state = State::Clear;
    loop {
        let b = peek(ui, vi, one);
        if b == f64::INFINITY {
            break;
        }
        let mut members = [(false, AX1, 0i64), (false, AXU, 0), (false, AXV, 0)];
        if !one && b == 1.0 {
            one = true;
            members[0].0 = true;
        }
        if ui < un && xval(uk, us, ui, ue, du) == b {
            members[1] = (true, AXU, uk + ui * us);
            ui += 1;
        }
        if vi < vn && xval(vk, vs, vi, ve, dv) == b {
            members[2] = (true, AXV, vk + vi * vs);
            vi += 1;
        }
        let nxt = peek(ui, vi, one);
        let grouped = members.iter().filter(|m| m.0).count();
        let fb = (grouped > 1 && !group_ok(cell, r, c, &members))
            || b - ta <= lim
            || nxt - b <= lim
            || !delta.is_finite();
        let last = nxt == f64::INFINITY;
        let tb = b;
        // a stretch the horizon certified clear (see Horizon), or else a block of
        // patches on which the exact gap is certainly > 0, is skipped
        let mut skip_to = -1.0f64;
        while next_mid < mid.len() && mid[next_mid].1 <= ta {
            next_mid += 1;
        }
        if next_mid < mid.len() && ta >= mid[next_mid].0 && mid[next_mid].1 > tb {
            skip_to = mid[next_mid].1;
        }
        if skip_to < 0.0 && delta.is_finite() {
            let mid = (ta + tb) / 2.0;
            let pr = (ve + mid * dv).floor() as i64 + 1;
            let pc = (ue + mid * du).floor() as i64 + 1;
            for lev in (1..=cur.min(nlev.saturating_sub(1))).rev() {
                let (brr, bcc) = (pr >> lev, pc >> lev);
                let (h, w) = pyr.shapes[lev];
                if brr < 0 || bcc < 0 || brr >= h || bcc >= w {
                    continue;
                }
                let key = brr * w + bcc;
                if failed[lev] == key {
                    continue;
                }
                let m = pyr.data[pyr.offs[lev] + key as usize];
                let mut tx = f64::INFINITY;
                if du > 0.0 {
                    tx = tx.min(((((bcc + 1) << lev) - 1) as f64 - ue) / du);
                } else if du < 0.0 {
                    tx = tx.min((((bcc << lev) - 1) as f64 - ue) / du);
                }
                if dv > 0.0 {
                    tx = tx.min(((((brr + 1) << lev) - 1) as f64 - ve) / dv);
                } else if dv < 0.0 {
                    tx = tx.min((((brr << lev) - 1) as f64 - ve) / dv);
                }
                if tx <= tb {
                    continue;
                }
                if above(cell, ta - delta, tx + delta, m) {
                    skip_to = tx;
                    cur = (lev + 1).min(nlev.saturating_sub(1));
                    break;
                }
                failed[lev] = key;
            }
        }
        if skip_to >= 0.0 {
            // drop every segment ending at or before skip_to; the new ta is the
            // largest raw boundary <= skip_to, its flag worked out afresh
            let ninf = f64::NEG_INFINITY;
            let j = first_above(uk, us, un, ui, ue, du, skip_to);
            let u1 = if j > ui {
                xval(uk, us, j - 1, ue, du)
            } else {
                ninf
            };
            let u2 = if j - 2 >= ui {
                xval(uk, us, j - 2, ue, du)
            } else {
                ninf
            };
            let ku = uk + (j - 1) * us;
            let jv = first_above(vk, vs, vn, vi, ve, dv, skip_to);
            let v1 = if jv > vi {
                xval(vk, vs, jv - 1, ve, dv)
            } else {
                ninf
            };
            let v2 = if jv - 2 >= vi {
                xval(vk, vs, jv - 2, ve, dv)
            } else {
                ninf
            };
            let kv = vk + (jv - 1) * vs;
            let one_now = !one && 1.0 <= skip_to;
            let o1 = if one_now { 1.0 } else { ninf };
            let nta = tb.max(u1).max(v1).max(o1);
            ui = j;
            vi = jv;
            if one_now {
                one = true;
            }
            if nta == tb {
                fa = fb;
            } else {
                let mut prev = tb;
                for val in [u1, u2, v1, v2, o1] {
                    if val < nta && val > prev {
                        prev = val;
                    }
                }
                let members = [
                    (o1 == nta, AX1, 0),
                    (u1 == nta, AXU, ku),
                    (v1 == nta, AXV, kv),
                ];
                let grouped = members.iter().filter(|m| m.0).count();
                let nxt = peek(ui, vi, one);
                fa = (grouped > 1 && !group_ok(cell, r, c, &members))
                    || nta - prev <= lim
                    || nxt - nta <= lim;
            }
            ta = nta;
            continue;
        }
        cur = 1;
        *count += 1;
        if fa || fb {
            // the crossings' exact order is uncertain here, but the terrain under
            // this stretch is no higher than the 3 x 3 patch max around it
            if !clear_level0(cell, pyr, ta, tb) {
                state = State::Undecided;
            }
        } else {
            let (st, r0, c0) = segment(cell, grid, ta, tb, last);
            match st {
                State::Blocking => return (State::Blocking, r0, c0),
                State::Undecided if !clear_level0(cell, pyr, ta, tb) => state = State::Undecided,
                _ => {}
            }
        }
        ta = tb;
        fa = fb;
    }
    (state, 0, 0)
}

// --- exact fallback ---------------------------------------------------------------

struct Exact {
    ue: BigRational,
    ve: BigRational,
    ze: BigRational,
    du: BigRational,
    dv: BigRational,
    dz: BigRational,
    bend: BigRational,
}

fn exact_setup(s: &Setup, z_rc: f64, r: i64, c: i64) -> Exact {
    let ue = s.ue_x.clone();
    let ve = s.ve_x.clone();
    let du = BigRational::from_integer(BigInt::from(c)) - &ue;
    let dv = BigRational::from_integer(BigInt::from(r)) - &ve;
    let res = rat(s.res);
    let bend = rat(s.drop) * (&du * &du + &dv * &dv) * &res * &res;
    let dz = rat(z_rc) - &bend + rat(s.th) - &s.ze_x;
    Exact {
        ue,
        ve,
        ze: s.ze_x.clone(),
        du,
        dv,
        dz,
        bend,
    }
}

fn ifloor(q: &BigRational) -> i64 {
    q.floor().to_integer().to_i64().expect("patch index fits")
}

/// The exact segment of the sightline inside patch (r0, c0), if any.
fn exact_patch_segment(e: &Exact, r0: i64, c0: i64) -> Option<(BigRational, BigRational)> {
    let mut ta = BigRational::zero();
    let mut tb = BigRational::from_integer(BigInt::from(1));
    for (p0, d, k0) in [(&e.ue, &e.du, c0), (&e.ve, &e.dv, r0)] {
        if d.is_zero() {
            if ifloor(p0) != k0 {
                return None;
            }
            continue;
        }
        let k = BigRational::from_integer(BigInt::from(k0));
        let k1 = BigRational::from_integer(BigInt::from(k0 + 1));
        let a = (&k - p0) / d;
        let b = (&k1 - p0) / d;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if lo > ta {
            ta = lo;
        }
        if hi < tb {
            tb = hi;
        }
    }
    if ta < tb {
        Some((ta, tb))
    } else {
        None
    }
}

/// The oracle's segment minimum in exact arithmetic (closed segment), None for no-data.
fn exact_segment_min(
    grid: &Grid,
    e: &Exact,
    ta: &BigRational,
    tb: &BigRational,
) -> Option<BigRational> {
    let two = BigRational::from_integer(BigInt::from(2));
    let one = BigRational::from_integer(BigInt::from(1));
    let mid = (ta + tb) / &two;
    let c0 = ifloor(&(&e.ue + &mid * &e.du));
    let r0 = ifloor(&(&e.ve + &mid * &e.dv));
    let p = corners(grid, r0, c0);
    if nodata(&p) {
        return None;
    }
    let (z00, z01, z10, z11) = (rat(p.z00), rat(p.z01), rat(p.z10), rat(p.z11));
    let (c0q, r0q) = (
        BigRational::from_integer(BigInt::from(c0)),
        BigRational::from_integer(BigInt::from(r0)),
    );
    let gap = |t: &BigRational| -> BigRational {
        let tc = &e.ue + t * &e.du - &c0q;
        let tr = &e.ve + t * &e.dv - &r0q;
        let bil = (&z00 * (&one - &tc) + &z01 * &tc) * (&one - &tr)
            + (&z10 * (&one - &tc) + &z11 * &tc) * &tr;
        &e.ze + t * &e.dz - bil + &e.bend * t * t
    };
    let mut m = gap(ta);
    let gb = gap(tb);
    if gb < m {
        m = gb;
    }
    let e3 = &z00 - &z01 - &z10 + &z11;
    let g2 = -(&e3) * &e.du * &e.dv + &e.bend;
    if g2.is_positive() {
        let tca = &e.ue + ta * &e.du - &c0q;
        let trb = &e.ve + ta * &e.dv - &r0q;
        let g1 = &e.dz
            - ((&z01 - &z00) * &e.du + (&z10 - &z00) * &e.dv + &e3 * (&tca * &e.dv + &trb * &e.du))
            + &two * &e.bend * ta;
        let s = -g1 / (&two * &g2);
        if s.is_positive() && s < tb - ta {
            let v = gap(&(ta + s));
            if v < m {
                m = v;
            }
        }
    }
    Some(m)
}

/// Exact decision on the cell's undecided patches (with neighbours); every other
/// segment is certified clear (by `segment`, or by `clear_level0` where the
/// crossings' order is uncertain).
fn blocked_exact(grid: &Grid, s: &Setup, pyr: &Pyramid, cell: &Cell, r: i64, c: i64) -> bool {
    let (mut ts, mut fl) = (Vec::new(), Vec::new());
    boundaries(cell, r, c, &mut ts, &mut fl);
    let mut patches: Vec<(i64, i64)> = Vec::new();
    for j in 0..ts.len() - 1 {
        let (ta, tb) = (ts[j], ts[j + 1]);
        let (st, r0, c0) = if fl[j] || fl[j + 1] {
            let mid = (ta + tb) / 2.0;
            (
                State::Undecided,
                (cell.ve + mid * cell.dv).floor() as i64,
                (cell.ue + mid * cell.du).floor() as i64,
            )
        } else {
            segment(cell, grid, ta, tb, j == ts.len() - 2)
        };
        if st == State::Undecided && !clear_level0(cell, pyr, ta, tb) {
            for a in -1..=1 {
                for b in -1..=1 {
                    patches.push((r0 + a, c0 + b));
                }
            }
        }
    }
    patches.sort_unstable();
    patches.dedup();
    patches.retain(|&(r0, c0)| !patch_clear(cell, grid, r0, c0));
    let e = exact_setup(s, grid.at(r as usize, c as usize), r, c);
    for (r0, c0) in patches {
        if let Some((ta, tb)) = exact_patch_segment(&e, r0, c0) {
            if let Some(m) = exact_segment_min(grid, &e, &ta, &tb) {
                if m.is_negative() {
                    return true;
                }
            }
        }
    }
    false
}

/// Exact bilinear ground height at the observer, None on no-data.
fn ground_exact(grid: &Grid, ue: &BigRational, ve: &BigRational) -> Option<BigRational> {
    let (c0, r0) = (ifloor(ue), ifloor(ve));
    let p = corners(grid, r0, c0);
    if nodata(&p) {
        return None;
    }
    let one = BigRational::from_integer(BigInt::from(1));
    let tc = ue - BigRational::from_integer(BigInt::from(c0));
    let tr = ve - BigRational::from_integer(BigInt::from(r0));
    let (z00, z01, z10, z11) = (rat(p.z00), rat(p.z01), rat(p.z10), rat(p.z11));
    Some(
        (&z00 * (&one - &tc) + &z01 * &tc) * (&one - &tr)
            + (&z10 * (&one - &tc) + &z11 * &tc) * &tr,
    )
}

/// Whether cell (r, c) is within max_distance of the observer, decided exactly:
/// (du^2 + dv^2) res^2 <= max^2 for the exact du, dv. Floats decide when the float
/// distance is clear of max by more than twice a bound on its error (the observer
/// coordinates' errors, plus a few roundings); otherwise rational arithmetic.
/// conformance/oracle.py's `within` is the same rule.
fn within(s: &Setup, r: i64, c: i64, max: f64) -> bool {
    let (du, dv) = (c as f64 - s.ue, r as f64 - s.ve);
    let d = (du * du + dv * dv).sqrt() * s.res;
    let err = s.res * (s.e_ue + s.e_ve + U * (du.abs() + dv.abs())) + 8.0 * U * d;
    if d < max - 2.0 * err {
        return true;
    }
    if d > max + 2.0 * err {
        return false;
    }
    let du = BigRational::from_integer(BigInt::from(c)) - &s.ue_x;
    let dv = BigRational::from_integer(BigInt::from(r)) - &s.ve_x;
    let res = rat(s.res);
    let m = rat(max);
    (&du * &du + &dv * &dv) * &res * &res <= &m * &m
}

// --- horizon filter ---------------------------------------------------------------
//
// A visible cell costs the walk a certified-clear segment for (nearly) every patch
// on its sightline. The horizon filter certifies the middle of the sightline at once:
// cells are processed in rings of increasing distance; before a ring, every patch
// whose nearest point is closer than the ring is inserted into an angular table that
// keeps, per sector, an upper bound on the steepest curvature-adjusted elevation
// slope (h - drop d^2 - ze*) / d of any point of any inserted patch whose angular
// span meets the sector. A sightline crosses only patches whose span contains its
// direction, and a point is above the sightline exactly when its slope beats the
// target's, (z_t + th - drop D^2 - ze*) / D. So if a lower bound on the target's
// slope beats the table's maximum over the sectors around its direction, every exact
// segment in the stretch crossing inserted patches is clear, and the walk skips it
// (walk's `mid`). Patches near the observer are left out (their slope bounds are
// useless), and the walk still does both ends: the near-observer part and the last
// few cells up to the target, which aren't inserted yet.
//
// Rigour: every float bound carries a relative margin of 1e-12 (four orders above
// the rounding error of the few operations involved) plus the observer coordinates'
// measured errors, and angles are widened by 1e-9 rad plus their error.

const NEAR: f64 = 3.0; // patches nearer than this many cells stay with the walk
const SECTORS: usize = 8; // horizon sectors per cell of raster width
// Rasters with at least this many target cells take the horizon mode: with hidden-block
// culling it is within 2% of the better mode in total over 219 real-terrain cases (lake
// windows and observers from map coordinates included) and at most 1.9x slower on any;
// the row-major walk was 2.6x slower in total. Smaller rasters take the walk.
const HORIZON_MIN: usize = 32_768;
const RING: f64 = 8.0; // ring width in cells (with culling, wider rings cost less overhead than their longer walked gaps: 8 vs 4 is 2-9% faster on all eight rbench cases)
const TURN: u64 = 1 << 40; // direction steps per turn in the cells' sort key
const REL: f64 = 1e-12;

/// A patch ready for the horizon: its nearest distance (a lower bound), its angular
/// span (widened) and an upper bound on the slope of any of its points.
struct HPatch {
    lo: f64,
    hi: f64,
    slope: f64,
    /// upper bound on the farthest distance, the span shrunk (certainly inside the
    /// true one), and a lower bound on the slope of any point of the patch
    lo_in: f64,
    hi_in: f64,
}

/// Upper bound on (h - drop d^2 - ze*) / d over d in [dl, dh], h <= hmax.
fn slope_hi(hmax: f64, ze_lo: f64, drop: f64, dl: f64, dh: f64) -> f64 {
    let dd = if drop >= 0.0 { dl } else { dh };
    let n = hmax - ze_lo - drop * dd * dd;
    let n = n + REL * (hmax.abs() + ze_lo.abs() + (drop * dd * dd).abs());
    let v = if n >= 0.0 { n / dl } else { n / dh };
    v + REL * v.abs()
}

/// Nearest distance from the observer to patch (r0, c0), before margins.
fn patch_dmin(s: &Setup, r0: i64, c0: i64) -> f64 {
    let (u0, u1, v0, v1) = (c0 as f64 - s.ue, (c0 + 1) as f64 - s.ue, r0 as f64 - s.ve, (r0 + 1) as f64 - s.ve);
    let dx = if u0 > 0.0 { u0 } else if u1 < 0.0 { -u1 } else { 0.0 };
    let dy = if v0 > 0.0 { v0 } else if v1 < 0.0 { -v1 } else { 0.0 };
    (dx * dx + dy * dy).sqrt() * s.res
}

/// The direction (atan2 of v, u) from the observer to every grid vertex (r, c), r in
/// -1..=rows, c in -1..=cols: computed once, shared by the four patches around each
/// vertex (the same formula as computing it per patch, so the same values).
struct Angles {
    data: Vec<f64>,
    w: usize,
}

impl Angles {
    fn new(s: &Setup, rows: usize, cols: usize) -> Angles {
        let (h, w) = (rows + 2, cols + 2);
        let mut data = vec![0.0f64; h * w];
        data.par_chunks_mut(w).enumerate().for_each(|(i, row)| {
            let v = (i as i64 - 1) as f64 - s.ve;
            for (j, a) in row.iter_mut().enumerate() {
                *a = v.atan2((j as i64 - 1) as f64 - s.ue);
            }
        });
        Angles { data, w }
    }

    /// The four corners of patch (r0, c0), in the order (0, 0), (1, 0), (0, 1), (1, 1)
    /// of (column offset, row offset).
    #[inline]
    fn patch(&self, r0: i64, c0: i64) -> [f64; 4] {
        let i = (r0 + 1) as usize * self.w + (c0 + 1) as usize;
        [self.data[i], self.data[i + 1], self.data[i + self.w], self.data[i + self.w + 1]]
    }
}

/// The horizon record of patch (r0, c0) (nearest distance `d0` from patch_dmin), or
/// None for a patch without data.
fn prepare_patch(grid: &Grid, s: &Setup, ang: &Angles, r0: i64, c0: i64, d0: f64) -> Option<HPatch> {
    let p = corners(grid, r0, c0);
    if nodata(&p) {
        return None;
    }
    let res = s.res;
    let e_pos = (s.e_ue + s.e_ve) * res;
    let (u0, u1, v0, v1) = (c0 as f64 - s.ue, (c0 + 1) as f64 - s.ue, r0 as f64 - s.ve, (r0 + 1) as f64 - s.ve);
    let dmin = d0 * (1.0 - REL) - 2.0 * e_pos;
    // the farthest corner: rounding is monotone, so this is the largest of the four
    let dmax = far_corner(u0, u1, v0, v1);
    // the corners' directions relative to corner (0, 0), wrapped: the patch is NEAR
    // cells away or more, so its span is far below pi
    let ca = ang.patch(r0, c0);
    let centre = ca[0];
    let (mut alo, mut ahi) = (0.0f64, 0.0f64);
    for &a in &ca[1..] {
        let mut a = a - centre;
        if a > std::f64::consts::PI {
            a -= 2.0 * std::f64::consts::PI;
        } else if a < -std::f64::consts::PI {
            a += 2.0 * std::f64::consts::PI;
        }
        alo = alo.min(a);
        ahi = ahi.max(a);
    }
    let dmax = dmax * res * (1.0 + REL) + 2.0 * e_pos;
    let widen = 1e-9 + 8.0 * e_pos / dmin;
    let hmax = p.z00.max(p.z01).max(p.z10).max(p.z11);
    Some(HPatch {
        lo: centre + alo - widen,
        hi: centre + ahi + widen,
        slope: slope_hi(hmax, s.ze - s.e_ze, s.drop, dmin, dmax),
        lo_in: centre + alo + widen,
        hi_in: centre + ahi - widen,
    })
}

/// The distance to the farthest of a patch's corners (u0..u1 by v0..v1, cells from the
/// observer): the same value as the largest of the four corner distances, since
/// squaring, adding and sqrt are monotone in rounding.
fn far_corner(u0: f64, u1: f64, v0: f64, v1: f64) -> f64 {
    let (u, v) = (u0.abs().max(u1.abs()), v0.abs().max(v1.abs()));
    (u * u + v * v).sqrt()
}

/// The ring before which something at distance d (map units) must be inserted: the
/// first ring k with d < (k RING - 1) res.
fn ring_of(d: f64, res: f64) -> usize {
    ((d / res + 1.0) / RING).floor() as usize + 1
}


/// Phase timing for VIEWSHED_DEBUG (stderr): each `lap` prints the time since the last.
struct Lap {
    on: bool,
    t: std::time::Instant,
}

impl Lap {
    fn new() -> Lap {
        Lap { on: std::env::var("VIEWSHED_DEBUG").is_ok(), t: std::time::Instant::now() }
    }

    fn lap(&mut self, name: &str) {
        if self.on {
            eprintln!("phase {name} {:.3}s", self.t.elapsed().as_secs_f64());
            self.t = std::time::Instant::now();
        }
    }
}

/// Bounds on the target's slope (z_t + th - drop D^2 - ze*) / D, its direction, and
/// the direction's error margin; None when delta makes the stretch bounds unsafe.
fn target_slope(s: &Setup, cell: &Cell, zt: f64) -> Option<(f64, f64, f64, f64, f64)> {
    let res = s.res;
    let e_pos = (s.e_ue + s.e_ve) * res;
    let dist = (cell.du * cell.du + cell.dv * cell.dv).sqrt() * res;
    if !cell.delta.is_finite() || cell.delta * dist > 0.25 * res || dist < 2.0 * res {
        return None;
    }
    let d_lo = dist * (1.0 - REL) - 2.0 * e_pos;
    let d_hi = dist * (1.0 + REL) + 2.0 * e_pos;
    let h = zt + s.th;
    let lo = {
        let dd = if s.drop >= 0.0 { d_hi } else { d_lo };
        let n = h - s.drop * dd * dd - (s.ze + s.e_ze);
        let n = n - REL * (h.abs() + (s.drop * dd * dd).abs() + s.ze.abs() + s.e_ze);
        let v = if n >= 0.0 { n / d_hi } else { n / d_lo };
        v - REL * v.abs()
    };
    let hi = {
        let dd = if s.drop >= 0.0 { d_lo } else { d_hi };
        let n = h - s.drop * dd * dd - (s.ze - s.e_ze);
        let n = n + REL * (h.abs() + (s.drop * dd * dd).abs() + s.ze.abs() + s.e_ze);
        let v = if n >= 0.0 { n / d_lo } else { n / d_hi };
        v + REL * v.abs()
    };
    let theta = cell.dv.atan2(cell.du);
    let widen = 1e-9 + 8.0 * e_pos / d_lo;
    Some((lo, hi, theta, widen, dist))
}


// --- horizon bins: analytic bounds and candidate lists ------------------------------
//
// Per angular bin: `low`, a lower bound on the horizon H(theta) of the inserted patches
// for every theta in the bin, and the candidates: inserted patches whose upper bound
// over the bin exceeds `low`. Invariant: every inserted patch meeting the bin is a
// candidate or has H <= low there. Bounds over an angular interval come from
// `bin_bounds` (closed form on the interval's centre ray plus a lateral slack), so a
// target is decided by its bin's bounds, by its candidates' bounds at its own
// direction, or (rarely) by the walk.

const REL_B: f64 = 1e-10; // margin on closed-form values, relative to the size of their terms
const CAND_CAP: usize = 48;

/// max over d in [d1, d2] of the patch polynomial's sigma = alpha d + beta + gamma / d
/// along (cth, sth), and a bound on the size of the terms that went into it.
fn centre_max(p: &Patch, s: &Setup, cth: f64, sth: f64, d1: f64, d2: f64) -> (f64, f64) {
    let (z00, z01, z10, z11) = (p.z00, p.z01, p.z10, p.z11);
    let res = s.res;
    let e3 = z00 - z01 - z10 + z11;
    let (du0, dv0) = (s.ue - p.c0, s.ve - p.r0);
    let bobs = z00 + (z01 - z00) * du0 + (z10 - z00) * dv0 + e3 * du0 * dv0;
    let alpha = e3 * cth * sth / (res * res) - s.drop;
    let beta = (((z01 - z00) + e3 * dv0) * cth + ((z10 - z00) + e3 * du0) * sth) / res;
    let gamma = bobs - s.ze;
    let d1 = d1.max(1e-300);
    let v1 = alpha * d1 + beta + gamma / d1;
    let v2 = alpha * d2 + beta + gamma / d2;
    let mut best = v1.max(v2);
    if alpha < 0.0 && gamma < 0.0 {
        let ds = (gamma / alpha).sqrt();
        if d1 < ds && ds < d2 {
            best = best.max(beta - 2.0 * (alpha * gamma).sqrt());
        }
    }
    // the polynomial is expanded around the observer: far away its terms are large and
    // cancel, so the rounding error scales with them, not with the result. Each term is
    // a height difference times a distance (each subtraction's rounding is relative to
    // its own result), so these are the magnitudes that matter, not the heights.
    let (gu, gv) = ((z01 - z00).abs(), (z10 - z00).abs());
    let e3a = e3.abs();
    let (eu, ev) = (1.0 + du0.abs(), 1.0 + dv0.abs());
    let size = (e3a / (res * res) + s.drop.abs()) * d2
        + (gu + gv + e3a * (eu + ev)) / res
        + (z00.abs() + s.ze.abs() + gu * eu + gv * ev + e3a * eu * ev) / d1;
    (best, size)
}

/// Ray-square stretch (map units) of direction (cth, sth) in patch (r0, c0).
fn stretch_d(s: &Setup, r0: f64, c0: f64, cth: f64, sth: f64) -> Option<(f64, f64)> {
    let (mut d1, mut d2) = (0.0f64, f64::INFINITY);
    for (p0, w, k0) in [(s.ue, cth, c0), (s.ve, sth, r0)] {
        if w == 0.0 {
            if !(k0 <= p0 && p0 <= k0 + 1.0) {
                return None;
            }
            continue;
        }
        let a = (k0 - p0) * s.res / w;
        let b = (k0 + 1.0 - p0) * s.res / w;
        d1 = d1.max(a.min(b));
        d2 = d2.min(a.max(b));
    }
    if d1 < d2 {
        Some((d1, d2))
    } else {
        None
    }
}

/// (lb, ub) on H_P(theta) for every theta in [ta, tb]; lb = -inf unless `covers` (the
/// patch meets every ray of the interval). Every point of the patch inside the wedge is
/// within lateral distance d tan(half) of the centre ray at the same distance d, so its
/// sigma differs from the centre ray's by at most G tan(half) / res (G: gradient bound
/// of the patch polynomial near the patch); curvature depends on d only. The observer's
/// position and eye-height errors and rounding are covered by further margins.
fn bin_bounds(p: &Patch, s: &Setup, ta: f64, tb: f64, covers: bool) -> (f64, f64) {
    let mut ca = [0.0f64; 4];
    for (i, (a_, b_)) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)].into_iter().enumerate() {
        ca[i] = (p.r0 + b_ - s.ve).atan2(p.c0 + a_ - s.ue);
    }
    bin_bounds_with(p, s, &ca, &Interval::new(ta, tb), covers)
}

/// An angular interval [ta, tb] with the trigonometry bin_bounds needs: computed once
/// per bin and reused for every patch (the same values as computing them on the spot).
#[derive(Clone, Copy)]
struct Interval {
    ta: f64,
    tb: f64,
    /// (cos, sin) of ta, tb, tm
    cs: [(f64, f64); 3],
    cos_half: f64,
    tan_half: f64,
}

impl Interval {
    fn new(ta: f64, tb: f64) -> Interval {
        let tm = 0.5 * (ta + tb);
        let half = 0.5 * (tb - ta);
        Interval {
            ta,
            tb,
            cs: [(ta.cos(), ta.sin()), (tb.cos(), tb.sin()), (tm.cos(), tm.sin())],
            cos_half: half.cos(),
            tan_half: half.tan(),
        }
    }
}

/// bin_bounds with the patch's corner directions already known.
fn bin_bounds_with(p: &Patch, s: &Setup, corner_angles: &[f64; 4], iv: &Interval, mut covers: bool) -> (f64, f64) {
    let res = s.res;
    let (ta, tb) = (iv.ta, iv.tb);
    let (cth, sth) = iv.cs[2];
    let (mut d1lo, mut d1hi, mut d2lo, mut d2hi) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    // Entry and exit distances are piecewise 1/cos of the angle to an (axis-aligned)
    // face normal: extreme at the interval's ends, where the face crossed changes (patch
    // corner directions), or at a face's normal direction (an axis direction). All of
    // those that fall inside the interval are evaluated, so the extremes are exact.
    // (cos, sin) of each direction: the interval's three from the table, the rest
    // (rarely any) computed here
    let mut dirs = [(0.0f64, 0.0f64); 16];
    dirs[..3].copy_from_slice(&iv.cs);
    let mut n = 3;
    let pi = std::f64::consts::PI;
    let axes = [-pi, -0.5 * pi, 0.0, 0.5 * pi, pi];
    for &a0 in corner_angles.iter().chain(axes.iter()) {
        for wrap in [-2.0 * pi, 0.0, 2.0 * pi] {
            if ta < a0 + wrap && a0 + wrap < tb && n < dirs.len() {
                let t = a0 + wrap;
                dirs[n] = (t.cos(), t.sin());
                n += 1;
            }
        }
    }
    let mut any = false;
    for &(ct, st) in &dirs[..n] {
        match stretch_d(s, p.r0, p.c0, ct, st) {
            Some((d1, d2)) => {
                any = true;
                d1lo = d1lo.min(d1);
                d1hi = d1hi.max(d1);
                d2lo = d2lo.min(d2);
                d2hi = d2hi.max(d2);
            }
            None => covers = false,
        }
    }
    if !any {
        return (f64::NEG_INFINITY, f64::NEG_INFINITY);
    }
    let e3 = (p.z00 - p.z01 - p.z10 + p.z11).abs();
    let g = (p.z01 - p.z00).abs().max((p.z11 - p.z10).abs())
        + (p.z10 - p.z00).abs().max((p.z11 - p.z01).abs())
        + 2.0 * e3;
    let e_pos = (s.e_ue + s.e_ve) * res; // observer position error, map units
    let pad = 1e-9 * res + 2.0 * e_pos;
    let slack = g * iv.tan_half * (1.0 + REL_B) / res;
    let d_near = d1lo * iv.cos_half - pad;
    // the observer's position error moves every point by e_pos, its eye by e_ze
    let err_obs = if d_near > 0.0 { (g * (e_pos / res) + s.e_ze) / d_near } else { f64::INFINITY };
    let (ub_v, size) = centre_max(p, s, cth, sth, d_near.max(0.0), d2hi / iv.cos_half + pad);
    let ub = ub_v + slack + err_obs + REL_B * (size + ub_v.abs() + slack);
    let mut lb = f64::NEG_INFINITY;
    if covers && d1hi < d2lo {
        let (a, b) = (d1hi / iv.cos_half + pad, d2lo * iv.cos_half - pad);
        if a < b {
            let (lb_v, size) = centre_max(p, s, cth, sth, a, b);
            lb = lb_v - slack - err_obs - REL_B * (size + lb_v.abs() + slack);
        }
    }
    (lb, ub)
}

/// bin_bounds for a hairline interval theta +- w (a target's direction and its error):
/// one stretch and one closed-form evaluation, unless a patch corner falls inside the
/// interval (then the full routine).
fn ray_bounds(p: &Patch, s: &Setup, theta: f64, w: f64, covers: bool) -> (f64, f64) {
    let (c, sn) = (theta.cos(), theta.sin());
    // a patch corner or an axis direction inside the interval: the crossing distances
    // may have an extreme inside it, so use the full routine (rare)
    for (a_, b_) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let (u, v) = (p.c0 + a_ - s.ue, p.r0 + b_ - s.ve);
        let cross = u * sn - v * c;
        if cross.abs() <= (u * u + v * v).sqrt() * (2.0 * w + 1e-12) && u * c + v * sn > 0.0 {
            return bin_bounds(p, s, theta - w, theta + w, covers);
        }
    }
    if c.abs() <= 2.0 * w + 1e-12 || sn.abs() <= 2.0 * w + 1e-12 {
        return bin_bounds(p, s, theta - w, theta + w, covers);
    }
    // otherwise the crossing distances are monotone across the interval: its two ends
    // give their exact ranges (however grazing the faces crossed)
    let (cth, sth) = (c, sn);
    let mut d1 = f64::INFINITY;
    let mut d2 = f64::NEG_INFINITY;
    let (mut d1hi, mut d2lo) = (f64::NEG_INFINITY, f64::INFINITY);
    let mut hit = 0;
    for t in [theta - w, theta, theta + w] {
        if let Some((a, b)) = stretch_d(s, p.r0, p.c0, t.cos(), t.sin()) {
            hit += 1;
            d1 = d1.min(a);
            d1hi = d1hi.max(a);
            d2 = d2.max(b);
            d2lo = d2lo.min(b);
        }
    }
    if hit == 0 {
        return (f64::NEG_INFINITY, f64::NEG_INFINITY);
    }
    let covers = covers && hit == 3;
    let res = s.res;
    let e3 = (p.z00 - p.z01 - p.z10 + p.z11).abs();
    let g = (p.z01 - p.z00).abs().max((p.z11 - p.z10).abs())
        + (p.z10 - p.z00).abs().max((p.z11 - p.z01).abs())
        + 2.0 * e3;
    let e_pos = (s.e_ue + s.e_ve) * res;
    let pad = d2 * 1e-12 + 1e-9 * res + 2.0 * e_pos;
    let slack = g * w.tan() * (1.0 + REL_B) / res;
    let d_near = d1 - pad;
    let err_obs = if d_near > 0.0 { (g * (e_pos / res) + s.e_ze) / d_near } else { f64::INFINITY };
    let (ub_v, size) = centre_max(p, s, cth, sth, d_near.max(0.0), d2 + pad);
    let ub = ub_v + slack + err_obs + REL_B * (size + ub_v.abs() + slack);
    let mut lb = f64::NEG_INFINITY;
    if covers {
        // distances every ray of the interval crosses inside the patch
        let (a, b) = (d1hi + pad, d2lo - pad);
        if a < b {
            let (lb_v, size) = centre_max(p, s, cth, sth, a, b);
            lb = lb_v - slack - err_obs - REL_B * (size + lb_v.abs() + slack);
        }
    }
    (lb, ub)
}

struct Cand {
    r0: i64,
    c0: i64,
    lo_in: f64,
    hi_in: f64,
    ub: f64,
}

struct Bins {
    low: Vec<f64>,
    cand: Vec<Vec<Cand>>,
    /// per bin: the largest upper bound of the candidates dropped when the list was
    /// full (the smallest ones go): every dropped patch has H <= this in the bin
    over: Vec<f64>,
    /// per bin: its interval [edge(i), edge(i + 1)] with trigonometry
    iv: Vec<Interval>,
    n: usize,
    scale: f64,
}

impl Bins {
    fn new(n: usize) -> Bins {
        let n = n.next_power_of_two();
        let scale = n as f64 / (2.0 * std::f64::consts::PI);
        let edge = |k: i64| k as f64 / scale - std::f64::consts::PI;
        Bins {
            low: vec![f64::NEG_INFINITY; n],
            cand: (0..n).map(|_| Vec::new()).collect(),
            over: vec![f64::NEG_INFINITY; n],
            iv: (0..n as i64).map(|k| Interval::new(edge(k), edge(k + 1))).collect(),
            n,
            scale,
        }
    }

    fn index(&self, theta: f64) -> i64 {
        ((theta + std::f64::consts::PI) * self.scale).floor() as i64
    }

    /// Insert a ring's patches, in parallel over chunks of bins: each chunk takes the
    /// patches whose span meets it (bins are independent, so no two threads touch the
    /// same bin).
    fn insert_all(&mut self, grid: &Grid, s: &Setup, ang: &Angles, patches: &[(i64, i64, HPatch)]) {
        let chunk = 256usize.min(self.n);
        let n = self.n as i64;
        let nchunks = self.n / chunk;
        // route each patch to the chunks its span meets
        let mut routes: Vec<Vec<usize>> = (0..nchunks).map(|_| Vec::new()).collect();
        for (j, (_, _, hp)) in patches.iter().enumerate() {
            let (a, b) = (self.index(hp.lo), self.index(hp.hi));
            let (ca, cb) = (a.div_euclid(chunk as i64), b.div_euclid(chunk as i64));
            for ch in ca..=cb {
                routes[ch.rem_euclid(nchunks as i64) as usize].push(j);
            }
        }
        let scale = self.scale;
        let edge = |k: i64| k as f64 / scale - std::f64::consts::PI;
        let index = |t: f64| ((t + std::f64::consts::PI) * scale).floor() as i64;
        let table = &self.iv;
        self.low
            .par_chunks_mut(chunk)
            .zip(self.cand.par_chunks_mut(chunk))
            .zip(self.over.par_chunks_mut(chunk))
            .zip(routes.par_iter())
            .enumerate()
            .for_each(|(ch, (((low, cand), over), route))| {
                let base = (ch * chunk) as i64;
                for &j in route {
                    let (r0, c0, hp) = &patches[j];
                    let p = corners(grid, *r0, *c0);
                    let (a, b) = (index(hp.lo), index(hp.hi));
                    let mut angles: Option<[f64; 4]> = None;
                    for k in a..=b {
                        let i = k.rem_euclid(n);
                        if i < base || i >= base + chunk as i64 {
                            continue;
                        }
                        let li = (i - base) as usize;
                        if hp.slope <= low[li] {
                            continue; // the crude whole-patch bound can't beat this bin
                        }
                        let ca = *angles.get_or_insert_with(|| ang.patch(*r0, *c0));
                        // a span wrapping past +-pi has k outside 0..n: its edges are
                        // shifted by 2 pi, so compute them as such (as always)
                        let iv = if k == i { table[i as usize] } else { Interval::new(edge(k), edge(k + 1)) };
                        let covers = hp.lo_in <= iv.ta && hp.hi_in >= iv.tb;
                        let (lb, ub) = bin_bounds_with(&p, s, &ca, &iv, covers);
                        if ub <= low[li] {
                            continue;
                        }
                        if lb > low[li] {
                            low[li] = lb;
                            cand[li].retain(|c| c.ub > lb);
                        }
                        cand[li].push(Cand { r0: *r0, c0: *c0, lo_in: hp.lo_in, hi_in: hp.hi_in, ub });
                        if cand[li].len() > CAND_CAP {
                            // drop the smallest upper bound, remembering it
                            let (m, _) = cand[li]
                                .iter()
                                .enumerate()
                                .min_by(|a, b| a.1.ub.total_cmp(&b.1.ub))
                                .unwrap();
                            let dropped = cand[li].swap_remove(m);
                            over[li] = over[li].max(dropped.ub);
                        }
                    }
                }
            });
    }
}

enum Verdict {
    Hidden,
    /// Every inserted patch is certainly below the sightline except these candidates
    /// (possibly none), whose own stretches the walk must check.
    Clear(Vec<(i64, i64)>),
    Unknown,
}

/// The target (slope bounds lo..hi, direction theta +- widen) against the bins.
fn bins_verdict(bins: &Bins, grid: &Grid, s: &Setup, lo: f64, hi: f64, theta: f64, widen: f64) -> Verdict {
    let n = bins.n as i64;
    let (a, b) = (bins.index(theta - widen), bins.index(theta + widen));
    let (mut lmin, mut lmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for k in a..=b {
        let i = k.rem_euclid(n) as usize;
        lmin = lmin.min(bins.low[i]);
        lmax = lmax.max(bins.low[i]).max(bins.over[i]);
    }
    if hi < lmin {
        return Verdict::Hidden;
    }
    // dominated and dropped patches have H <= lmax; candidates are checked at the
    // target's own direction
    let clear = lo > lmax;
    let mut straddle = Vec::new();
    for k in a..=b {
        let i = k.rem_euclid(n) as usize;
        for c in &bins.cand[i] {
            if c.ub < lo {
                continue; // below the target over the whole bin
            }
            let p = corners(grid, c.r0, c.c0);
            let covers = c.lo_in <= theta - widen && c.hi_in >= theta + widen;
            let (lb, ub) = ray_bounds(&p, s, theta, widen, covers);
            if lb > hi {
                return Verdict::Hidden;
            }
            if ub >= lo {
                straddle.push((c.r0, c.c0));
                if straddle.len() > 8 {
                    return Verdict::Unknown; // the walk will be quicker than more checks
                }
            }
        }
    }
    if clear && straddle.len() <= 8 {
        Verdict::Clear(straddle)
    } else {
        Verdict::Unknown
    }
}

/// The parts of [t_a, t_b] of the sightline to `cell` outside the given patches'
/// stretches (each widened by 1.5 cells along the sightline): sorted, disjoint.
fn skip_intervals(cell: &Cell, s: &Setup, t_a: f64, t_b: f64, holes: &[(i64, i64)]) -> Vec<(f64, f64)> {
    let len = (cell.du * cell.du + cell.dv * cell.dv).sqrt();
    let pad = 1.5 / len;
    let mut cut: Vec<(f64, f64)> = holes
        .iter()
        .filter_map(|&(r0, c0)| {
            let (mut ta, mut tb) = (f64::NEG_INFINITY, f64::INFINITY);
            for (p0, d, k0) in [(s.ue, cell.du, c0), (s.ve, cell.dv, r0)] {
                if d == 0.0 {
                    continue;
                }
                let x = (k0 as f64 - p0) / d;
                let y = ((k0 + 1) as f64 - p0) / d;
                ta = ta.max(x.min(y));
                tb = tb.min(x.max(y));
            }
            if ta.is_finite() && tb.is_finite() {
                Some((ta - pad, tb + pad))
            } else {
                Some((t_a, t_b)) // degenerate: don't skip anything
            }
        })
        .collect();
    cut.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out = Vec::new();
    let mut from = t_a;
    for (x, y) in cut {
        if x > from {
            out.push((from, x.min(t_b)));
        }
        from = from.max(y);
        if from >= t_b {
            break;
        }
    }
    if from < t_b {
        out.push((from, t_b));
    }
    out.retain(|&(x, y)| y > x);
    out
}

const CULL_BLOCK: usize = 4; // cells per side of a culling block

/// Hidden-block culling: is block (r0, c0) (CULL_BLOCK cells a side) certainly hidden by
/// the bins? The block owns its cells and the patches whose (0, 0) corner is one of them;
/// all of that lies in the square [c0, c0 + kb] x [r0, r0 + kb] (cell-centre coordinates;
/// rows and columns past the raster are its clamped edge), with heights at most the
/// largest of the square's cells. Every point there has slope at most slope_hi of that
/// height over the square's distance range; culled when that is below every bin's lower
/// bound over the square's angular span (bins: patches entirely nearer). A culled block's
/// cells are hidden, and its patches can be left out of the bins: each of their points is
/// below the bins' lower bound, which inserted (nearer, real) terrain attains.
fn block_culled(bins: &Bins, grid: &Grid, s: &Setup, r0: usize, c0: usize) -> bool {
    let kb = CULL_BLOCK;
    let (rows, cols) = (grid.rows, grid.cols);
    let mut hmax = f64::NEG_INFINITY;
    for r in r0..=(r0 + kb).min(rows - 1) {
        for c in c0..=(c0 + kb).min(cols - 1) {
            let v = grid.at(r, c);
            if v.is_nan() {
                return false; // blocks touching no-data stay with the cell-by-cell path
            }
            hmax = hmax.max(v);
        }
    }
    let res = s.res;
    let e_pos = (s.e_ue + s.e_ve) * res;
    let (u0, u1, v0, v1) = (c0 as f64 - s.ue, (c0 + kb) as f64 - s.ue, r0 as f64 - s.ve, (r0 + kb) as f64 - s.ve);
    let dx = if u0 > 0.0 { u0 } else if u1 < 0.0 { -u1 } else { 0.0 };
    let dy = if v0 > 0.0 { v0 } else if v1 < 0.0 { -v1 } else { 0.0 };
    let d_lo = (dx * dx + dy * dy).sqrt() * res * (1.0 - REL) - 2.0 * e_pos;
    if d_lo < 3.0 * res {
        return false;
    }
    let d_hi = far_corner(u0, u1, v0, v1) * res * (1.0 + REL) + 2.0 * e_pos;
    let smax = slope_hi(hmax + s.th.max(0.0), s.ze - s.e_ze, s.drop, d_lo, d_hi);
    // the square's angular span from its corners (it may reach past the vertex table)
    let ca = [v0.atan2(u0), v0.atan2(u1), v1.atan2(u0), v1.atan2(u1)];
    let centre = ca[0];
    let (mut alo, mut ahi) = (0.0f64, 0.0f64);
    for &a in &ca[1..] {
        let mut a = a - centre;
        if a > std::f64::consts::PI {
            a -= 2.0 * std::f64::consts::PI;
        } else if a < -std::f64::consts::PI {
            a += 2.0 * std::f64::consts::PI;
        }
        alo = alo.min(a);
        ahi = ahi.max(a);
    }
    let w = 1e-9 + 8.0 * e_pos / d_lo;
    let (a, b) = (bins.index(centre + alo - w), bins.index(centre + ahi + w));
    let n = bins.n as i64;
    if b - a >= n {
        return false;
    }
    for q in a..=b {
        if smax.partial_cmp(&bins.low[q.rem_euclid(n) as usize]) != Some(std::cmp::Ordering::Less) {
            return false;
        }
    }
    true
}

/// The deferred cells with the horizon filters, in rings of increasing distance.
fn horizon_phase(grid: &Grid, s: &Setup, pyr: &Pyramid, cells: &[(i64, i64)], seen: &mut [bool]) {
    let mut lap = Lap::new();
    let (rows, cols) = (grid.rows, grid.cols);
    let nlev = pyr.shapes.len();
    let res = s.res;
    let n_rings = (((rows * rows + cols * cols) as f64).sqrt() / RING) as usize + 4;
    let last = n_rings - 1;
    let (rows_i, cols_i) = (rows as i64, cols as i64);
    let near = NEAR * res;
    // The cells and patches are handed out lazily, by culling blocks (CULL_BLOCK cells a
    // side): each ring, after its patches are inserted, the blocks whose nearest point
    // lies in it are tested against the bins; a culled block's cells are hidden and its
    // patches never prepared; the others hand their target cells to the rings they lie
    // in and their patches to the rings before which they must be inserted. So only
    // surviving terrain is keyed, sorted and prepared.
    let mut is_target = vec![false; rows * cols];
    for &(r, c) in cells {
        is_target[r as usize * cols + c as usize] = true;
    }
    // a cell's ring and its key within the ring (direction in 2^40 steps per turn: the
    // order within a ring only affects speed, never a result)
    let cell_key = |i: usize| -> (usize, (u64, usize)) {
        let (r, c) = (i / cols, i % cols);
        let (du, dv) = (c as f64 - s.ue, r as f64 - s.ve);
        let d = (du * du + dv * dv).sqrt();
        let ring = ((d / RING).floor() as usize).min(last);
        let q = ((dv.atan2(du) + std::f64::consts::PI) * (TURN as f64 / (2.0 * std::f64::consts::PI))) as u64;
        (ring, (q.min(TURN - 1), i))
    };
    // a patch's insertion ring: the ring before which it lies entirely inside the
    // processed radius (by its farthest point); patches within NEAR stay with the walk
    let patch_ring = |r0: i64, c0: i64| -> Option<usize> {
        if patch_dmin(s, r0, c0) < near {
            return None;
        }
        let far = far_corner(c0 as f64 - s.ue, c0 as f64 + 1.0 - s.ue, r0 as f64 - s.ve, r0 as f64 + 1.0 - s.ve);
        let far = far * res * (1.0 + REL) + 2.0 * (s.e_ue + s.e_ve) * res;
        Some(ring_of(far, res).min(last))
    };
    let mut cbuck: Vec<Vec<(u64, usize)>> = (0..n_rings).map(|_| Vec::new()).collect();
    let mut pbuck: Vec<Vec<(i64, i64)>> = (0..n_rings).map(|_| Vec::new()).collect();
    // patches on the -1 row and column belong to no block
    for c0 in -1..cols_i {
        if let Some(k) = patch_ring(-1, c0) {
            pbuck[k].push((-1, c0));
        }
    }
    for r0 in 0..rows_i {
        if let Some(k) = patch_ring(r0, -1) {
            pbuck[k].push((r0, -1));
        }
    }
    // blocks by ring (of their nearest point, less a hair: no cell of a block can then
    // belong to an earlier ring)
    let kb = CULL_BLOCK;
    let mut blocks: Vec<Vec<(usize, usize)>> = (0..n_rings).map(|_| Vec::new()).collect();
    for r0 in (0..rows).step_by(kb) {
        for c0 in (0..cols).step_by(kb) {
            let (u0, u1, v0, v1) = (c0 as f64 - s.ue, (c0 + kb) as f64 - s.ue, r0 as f64 - s.ve, (r0 + kb) as f64 - s.ve);
            let dx = if u0 > 0.0 { u0 } else if u1 < 0.0 { -u1 } else { 0.0 };
            let dy = if v0 > 0.0 { v0 } else if v1 < 0.0 { -v1 } else { 0.0 };
            let d = (dx * dx + dy * dy).sqrt() - 1e-6;
            blocks[((d.max(0.0) / RING).floor() as usize).min(last)].push((r0, c0));
        }
    }
    let mut bins = Bins::new(SECTORS * rows.max(cols));
    let ang = Angles::new(s, rows, cols);
    lap.lap("h:blocks+angles");
    let (mut t_prep, mut t_ins, mut t_q) = (0.0f64, 0.0f64, 0.0f64);
    let isr = &is_target;
    for ring in 0..n_rings {
        // every patch entirely nearer than the ring's inner edge (less a cell)
        let inserted = ((ring as f64) * RING - 1.0) * res;
        let pring = std::mem::take(&mut pbuck[ring]);
        if !pring.is_empty() {
            let t0 = std::time::Instant::now();
            let prepared: Vec<(i64, i64, HPatch)> = pring
                .par_iter()
                .filter_map(|&(r0, c0)| prepare_patch(grid, s, &ang, r0, c0, patch_dmin(s, r0, c0)).map(|hp| (r0, c0, hp)))
                .collect();
            t_prep += t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            bins.insert_all(grid, s, &ang, &prepared);
            t_ins += t0.elapsed().as_secs_f64();
        }
        // this ring's blocks, against the bins so far: culled (nothing to hand out) or
        // handing out their target cells and patches
        type Handed = (Vec<(usize, (u64, usize))>, Vec<(usize, (i64, i64))>);
        let handed: Vec<Handed> = blocks[ring]
            .par_iter()
            .filter(|&&(r0, c0)| !block_culled(&bins, grid, s, r0, c0))
            .map(|&(r0, c0)| {
                let (mut cs, mut ps) = (Vec::new(), Vec::new());
                for r in r0..(r0 + kb).min(rows) {
                    for c in c0..(c0 + kb).min(cols) {
                        let i = r * cols + c;
                        if isr[i] {
                            cs.push(cell_key(i));
                        }
                        if let Some(k) = patch_ring(r as i64, c as i64) {
                            ps.push((k, (r as i64, c as i64)));
                        }
                    }
                }
                (cs, ps)
            })
            .collect();
        for (cs, ps) in handed {
            for (k, key) in cs {
                debug_assert!(k >= ring);
                cbuck[k.max(ring)].push(key);
            }
            for (k, rc) in ps {
                debug_assert!(k > ring);
                pbuck[k].push(rc);
            }
        }
        let mut cring = std::mem::take(&mut cbuck[ring]);
        if cring.is_empty() {
            continue;
        }
        if cring.len() > 4096 {
            cring.par_sort_unstable();
        } else {
            cring.sort_unstable();
        }
        let h = &bins;
        let t0 = std::time::Instant::now();
        let results: Vec<(usize, bool)> = cring
            .par_chunks(64)
            .flat_map_iter(|chunk| {
                let mut failed = vec![-1i64; nlev];
                let mut blocker: Option<(i64, i64)> = None;
                let mut out = Vec::with_capacity(chunk.len());
                for &(_, i) in chunk {
                    let (r, c) = ((i / cols) as i64, (i % cols) as i64);
                    let zt = grid.at(r as usize, c as usize);
                    let cell = make_cell(s, zt, r, c);
                    let idx = r as usize * cols + c as usize;
                    if let Some((br, bc)) = blocker {
                        if reuse_blocks(&cell, grid, br, bc) {
                            out.push((idx, false));
                            continue;
                        }
                    }
                    let mut mid = Vec::new();
                    if let Some((lo, hi, theta, widen, dist)) = target_slope(s, &cell, zt) {
                        match bins_verdict(h, grid, s, lo, hi, theta, widen) {
                            Verdict::Hidden => {
                                out.push((idx, false));
                                continue;
                            }
                            Verdict::Clear(straddle) => {
                                // t_a: beyond it every patch crossed is NEAR cells away;
                                // t_b: up to it every patch crossed lies entirely inside
                                // `inserted` (sqrt 2 cells of patch, plus rounding); the
                                // straddling candidates' own stretches stay with the walk
                                let t_a = (NEAR * res + 2.0 * res) / dist;
                                let t_b = (inserted - 2.0 * res) / dist;
                                if t_b > t_a + 0.01 {
                                    mid = skip_intervals(&cell, s, t_a, t_b, &straddle);
                                }
                            }
                            Verdict::Unknown => {}
                        }
                    }
                    let (st, r0, c0) = walk(&cell, grid, pyr, r, c, &mut failed, &mid, &mut 0);
                    let visible = match st {
                        State::Blocking => {
                            blocker = Some((r0, c0));
                            false
                        }
                        State::Undecided => !blocked_exact(grid, s, pyr, &cell, r, c),
                        _ => true,
                    };
                    out.push((idx, visible));
                }
                out.into_iter()
            })
            .collect();
        t_q += t0.elapsed().as_secs_f64();
        for (k, v) in results {
            seen[k] = v;
        }
    }
    if lap.on {
        eprintln!("phase h:prepare {t_prep:.3}s h:insert {t_ins:.3}s h:queries {t_q:.3}s");
    }
    lap.lap("h:ring-loop");
}

pub(crate) fn viewshed(grid: &Grid, x: f64, y: f64, opts: &Options) -> Result<Vec<bool>, Error> {
    if !grid.inside(x, y) {
        return Err(Error::ObserverOutside);
    }
    let mut lap = Lap::new();
    let t = grid.transform;
    let half = BigRational::new(BigInt::from(1), BigInt::from(2));
    let ue_x = (rat(x) - rat(t[0])) / rat(t[1]) - &half;
    let ve_x = (rat(y) - rat(t[3])) / rat(t[5]) - &half;
    let ground = ground_exact(grid, &ue_x, &ve_x).ok_or(Error::ObserverOnNoData)?;
    let (ue, ve) = grid.frac(x, y);
    let ze_x = &ground + rat(opts.observer_height);
    let ze = ground.to_f64().unwrap() + opts.observer_height;
    let s = Setup {
        ue,
        ve,
        ze,
        e_ue: up(&(rat(ue) - &ue_x).abs()),
        e_ve: up(&(rat(ve) - &ve_x).abs()),
        e_ze: up(&(rat(ze) - &ze_x).abs()),
        th: opts.target_height,
        drop: opts.drop_factor(),
        res: grid.res(),
        ue_x,
        ve_x,
        ze_x,
    };
    lap.lap("observer");
    let (rows, cols) = (grid.rows, grid.cols);
    let mut cells: Vec<(i64, i64)> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if grid.at(r, c).is_nan() {
                continue;
            }
            if let Some(max) = opts.max_distance {
                if !within(&s, r as i64, c as i64, max) {
                    continue;
                }
            }
            cells.push((r as i64, c as i64));
        }
    }
    lap.lap("cells");
    let pyr = Pyramid::new(grid);
    let nlev = pyr.shapes.len();
    lap.lap("pyramid");
    // Two exact ways through the cells: row-major chunks (neighbouring sightlines
    // share blockers, cache lines and pyramid blocks) for small rasters, or rings with
    // the horizon filters and hidden-block culling (HORIZON_MIN).
    let mut seen = vec![false; rows * cols];
    // VIEWSHED_MODE=rows|horizon forces a mode (the conformance tests run both)
    let forced = match std::env::var("VIEWSHED_MODE").as_deref() {
        Ok("rows") => Some(false),
        Ok("horizon") => Some(true),
        _ => None,
    };
    let use_horizon = forced.unwrap_or(cells.len() >= HORIZON_MIN);
    lap.lap("mode");
    // the horizon's sort keys pack a ring (24 bits) above a patch index (40 bits)
    let fits = (rows as u64 + 1) * (cols as u64 + 1) < (1u64 << 40)
        && ((((rows * rows + cols * cols) as f64).sqrt() / RING) as u64 + 4) < (1u64 << 24);
    if use_horizon && fits {
        horizon_phase(grid, &s, &pyr, &cells, &mut seen);
    } else {
        let decided: Vec<(usize, bool)> = cells
            .par_chunks(512)
            .flat_map_iter(|chunk| {
                let mut failed = vec![-1i64; nlev];
                let mut blocker: Option<(i64, i64)> = None;
                let mut out = Vec::with_capacity(chunk.len());
                for &(r, c) in chunk {
                    let cell = make_cell(&s, grid.at(r as usize, c as usize), r, c);
                    let idx = r as usize * cols + c as usize;
                    if let Some((br, bc)) = blocker {
                        if reuse_blocks(&cell, grid, br, bc) {
                            out.push((idx, false));
                            continue;
                        }
                    }
                    let (st, r0, c0) = walk(&cell, grid, &pyr, r, c, &mut failed, &[], &mut 0);
                    out.push((idx, match st {
                        State::Blocking => {
                            blocker = Some((r0, c0));
                            false
                        }
                        State::Undecided => !blocked_exact(grid, &s, &pyr, &cell, r, c),
                        _ => true,
                    }));
                }
                out.into_iter()
            })
            .collect();
        for (idx, v) in decided {
            seen[idx] = v;
        }
        lap.lap("rows");
    }
    Ok(seen)
}
