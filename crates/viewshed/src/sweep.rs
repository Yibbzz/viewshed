//! Radial sweep for the r.viewshed surface model, O(n log n) in practice.
//!
//! Implemented from the description in Haverkort, Toma & Zhuang, "Computing
//! visibility on terrains in external memory" (ALENEX 2007), not from GRASS's
//! GPL code.
//!
//! # Model
//!
//! The observer V is the centre of the cell containing the observer point; the
//! eye is that cell's height plus `observer_height`. Every other cell with data
//! has three samples: its centre, and the two corners that bound it as seen
//! from V (ENTER and EXIT). A corner's height is the mean of the four cells
//! meeting there, or the cell's own height if any of them is no-data or off the
//! grid. Each sample has a gradient `atan(dz / distance)`.
//!
//! A cell is *active* for a direction θ when ENTER < θ < EXIT (strict); its
//! gradient at θ is interpolated linearly in angle between ENTER and centre, or
//! centre and EXIT. A target cell (centre + `target_height`) is hidden when
//! an active cell whose centre is strictly closer to V has a strictly greater
//! gradient at the target's direction.
//!
//! # Method
//!
//! A line from V sweeps once around the circle. Each cell has ENTER, CENTER and
//! EXIT events; at equal directions EXIT comes before CENTER before ENTER.
//! Active cells live in a segment tree over integer distance rings. Each node
//! keeps the largest upper and lower bound of its active cells' slopes, so a
//! query skips subtrees that can't block and stops at the first that must.
//!
//! Trigonometry is deferred: events are ordered by an exact stand-in for the
//! angle (see [`pseudo_angle`]), the tree works on slopes `dz / distance`
//! (atan is monotone), and the real angles and gradients are computed only
//! for the cells a query actually examines, with the same formulas as the
//! model, so results are bit-for-bit those of a direct evaluation.

use std::cmp::Ordering;
use std::f64::consts::PI;

use rayon::prelude::*;

use crate::{Error, Grid, Options};

/// Directions in doubled index units (so corners are integers), y pointing north.
type Dir = (i64, i64);

/// One cell's three samples. 40 bytes; everything else is derived.
#[derive(Clone, Copy)]
struct Cell {
    slope: [f64; 3], // dz / distance at ENTER, centre, EXIT
    dr: i32,         // offset from the observer cell, rows (south +)
    dc: i32,         // and columns
    ring: u32,       // floor of the distance in cells; u32::MAX = no cell (no data or out of range)
}

impl Cell {
    const NONE: Cell = Cell { slope: [0.0; 3], dr: 0, dc: 0, ring: u32::MAX };

    fn key(&self) -> i64 {
        let (dr, dc) = (self.dr as i64, self.dc as i64);
        dr * dr + dc * dc
    }

    fn hi(&self) -> f64 {
        self.slope[0].max(self.slope[1]).max(self.slope[2])
    }

    fn lo(&self) -> f64 {
        self.slope[0].min(self.slope[1]).min(self.slope[2])
    }

    /// Exact direction of sample `slot` (0 = ENTER, 1 = centre, 2 = EXIT).
    fn dir(&self, slot: usize) -> Dir {
        let (dr, dc) = (self.dr as i64, self.dc as i64);
        let (oc, or) = match slot {
            1 => (0, 0),
            s => corners(dr, dc)[s / 2],
        };
        (2 * dc + oc, -(2 * dr + or))
    }
}

#[derive(Clone, Copy)]
struct Event {
    /// Bits of the sample direction's pseudo_angle (>= 0, so they sort like the
    /// number) with the event kind in the lowest two bits: 0 = EXIT, 1 = CENTER,
    /// 2 = ENTER. Distinct directions are many ulp apart, so this is one integer
    /// that orders by direction, then kind.
    key: u64,
    cell: u32,
}

impl Event {
    fn new(cell: &Cell, i: u32, kind: u8) -> Self {
        let bits = pseudo_angle(cell.dir(2 - kind as usize)).to_bits();
        Event { key: (bits & !3) | kind as u64, cell: i }
    }

    fn kind(&self) -> u8 {
        (self.key & 3) as u8
    }
}

fn half(d: Dir) -> u8 {
    if d.1 > 0 || (d.1 == 0 && d.0 > 0) {
        0
    } else {
        1
    }
}

/// Counter-clockwise order from east, exact.
fn cmp_dir(a: Dir, b: Dir) -> Ordering {
    half(a).cmp(&half(b)).then_with(|| {
        let cross = a.0 as i128 * b.1 as i128 - a.1 as i128 * b.0 as i128;
        0.cmp(&cross)
    })
}

/// A strictly increasing function of the counter-clockwise angle from east, in
/// [0, 4): the quadrant plus a ratio in [0, 1) ("diamond angle"). One division,
/// no trigonometry. Equal directions give bit-identical values (the same
/// rational, correctly rounded). Different directions with coordinates below n
/// differ by at least 1/(4 n^2), far above rounding while n < ~1e7, so sorting
/// by it gives the exact order.
fn pseudo_angle(d: Dir) -> f64 {
    let (x, y) = (d.0 as f64, d.1 as f64);
    if d.0 > 0 && d.1 >= 0 {
        y / (x + y)
    } else if d.0 <= 0 && d.1 > 0 {
        1.0 + (-x) / (y - x)
    } else if d.0 < 0 && d.1 <= 0 {
        2.0 + (-y) / (-x - y)
    } else {
        3.0 + x / (x - y)
    }
}

/// Largest coordinate (in half cells) for which `pseudo_angle` order is exact
/// even with the two lowest bits replaced by the event kind: distinct keys are
/// then at least 1/(4 n^2) > 6e-14 apart, over 60 ulp of a number below 4.
const PSEUDO_ANGLE_LIMIT: i64 = 2_000_000;

fn angle(d: Dir) -> f64 {
    (d.1 as f64).atan2(d.0 as f64).rem_euclid(2.0 * PI)
}

/// ENTER and EXIT corners as (d_col, d_row) offsets in half cells, row growing south.
fn corners(dr: i64, dc: i64) -> [(i64, i64); 2] {
    match (dr.signum(), dc.signum()) {
        (-1, -1) => [(1, -1), (-1, 1)],
        (-1, 0) => [(1, 1), (-1, 1)],
        (-1, 1) => [(1, 1), (-1, -1)],
        (0, 1) => [(-1, 1), (-1, -1)],
        (1, 1) => [(-1, 1), (1, -1)],
        (1, 0) => [(-1, -1), (1, -1)],
        (1, -1) => [(-1, -1), (1, 1)],
        (0, -1) => [(1, -1), (1, 1)],
        _ => unreachable!("the observer cell has no corners"),
    }
}

/// How far below (above) the target's slope a bound must be before it can
/// rule a subtree out (in). atan is monotone, but rounding in atan and in the
/// interpolation can move gradients by a few ulp; this margin is worth more
/// than 40 ulp of gradient. Very steep targets (only right next to the
/// observer) get no shortcuts and are checked cell by cell.
fn margin(st: f64) -> f64 {
    if st.abs() <= 1e6 {
        1e-14 * (1.0 + st * st)
    } else {
        f64::INFINITY
    }
}

/// Active cells grouped by integer distance ring (floor of the distance in
/// cells), with a segment tree over the rings. Only about one cell per ring
/// is active at a time, and the tree has one leaf per ring, so it stays small
/// and in cache however large the raster.
struct RingTree {
    size: usize,
    hi: Vec<f64>,
    lo: Vec<f64>,
    members: Vec<Vec<u32>>,
}

/// The target of one query, with its angle and gradient computed on first use.
struct Target {
    dir: Dir,
    key: i64,
    ring: usize,
    slope: f64,
    theta: Option<f64>,
    grad: f64,
}

impl Target {
    /// Does `cell` block it? Evaluated exactly as the model states.
    fn blocked_by(&mut self, cell: &Cell) -> bool {
        let theta = match self.theta {
            Some(t) => t,
            None => {
                let t = angle(self.dir);
                self.theta = Some(t);
                self.grad = self.slope.atan();
                t
            }
        };
        gradient_at(cell, theta) > self.grad
    }
}

impl RingTree {
    fn new(rings: usize) -> Self {
        let size = rings.next_power_of_two().max(1);
        RingTree {
            size,
            hi: vec![f64::NEG_INFINITY; 2 * size],
            lo: vec![f64::NEG_INFINITY; 2 * size],
            members: vec![Vec::new(); size],
        }
    }

    fn insert(&mut self, ring: usize, i: u32, cells: &[Cell]) {
        self.members[ring].push(i);
        self.refresh(ring, cells);
    }

    fn remove(&mut self, ring: usize, i: u32, cells: &[Cell]) {
        if let Some(p) = self.members[ring].iter().position(|&m| m == i) {
            self.members[ring].swap_remove(p);
            self.refresh(ring, cells);
        }
    }

    fn refresh(&mut self, ring: usize, cells: &[Cell]) {
        let (mut hi, mut lo) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &m in &self.members[ring] {
            hi = hi.max(cells[m as usize].hi());
            lo = lo.max(cells[m as usize].lo());
        }
        let mut k = ring + self.size;
        self.hi[k] = hi;
        self.lo[k] = lo;
        while k > 1 {
            k /= 2;
            let (hi, lo) = (self.hi[2 * k].max(self.hi[2 * k + 1]), self.lo[2 * k].max(self.lo[2 * k + 1]));
            if hi == self.hi[k] && lo == self.lo[k] {
                break; // nothing above changes
            }
            self.hi[k] = hi;
            self.lo[k] = lo;
        }
    }

    /// Does an active cell strictly closer than the target block it?
    fn blocked(&self, cells: &[Cell], t: &mut Target) -> bool {
        let m = margin(t.slope);
        let (skip_below, sure_above) = (t.slope - m, t.slope + m);
        // Rings inside the target's: all strictly closer.
        let mut stack = [(0usize, 0usize, 0usize); 128];
        let mut top = 1;
        stack[0] = (1, 0, self.size);
        while top > 0 {
            top -= 1;
            let (node, l, r) = stack[top];
            if l >= t.ring || self.hi[node] < skip_below {
                continue;
            }
            if r <= t.ring && self.lo[node] > sure_above {
                return true;
            }
            if r - l == 1 {
                if self.members[l].iter().any(|&c| t.blocked_by(&cells[c as usize])) {
                    return true;
                }
                continue;
            }
            // Visit the child with the steeper bound first: it holds the likely horizon.
            let mid = (l + r) / 2;
            let (a, b) = ((2 * node, l, mid), (2 * node + 1, mid, r));
            let (first, second) = if self.hi[2 * node] >= self.hi[2 * node + 1] { (a, b) } else { (b, a) };
            stack[top] = second;
            stack[top + 1] = first;
            top += 2;
        }
        // The target's own ring: only cells with a smaller squared distance.
        let key = t.key;
        self.members[t.ring].iter().any(|&c| cells[c as usize].key() < key && t.blocked_by(&cells[c as usize]))
    }
}

fn rel(a: f64, theta: f64) -> f64 {
    (a - theta + PI).rem_euclid(2.0 * PI) - PI
}

/// A cell's gradient at direction `theta`, interpolated in angle.
fn gradient_at(cell: &Cell, theta: f64) -> f64 {
    let a = [angle(cell.dir(0)), angle(cell.dir(1)), angle(cell.dir(2))];
    let (d0, d1, d2) = (rel(a[0], theta), rel(a[1], theta), rel(a[2], theta));
    if d1 > 0.0 {
        let (g0, g1) = (cell.slope[0].atan(), cell.slope[1].atan());
        g1 + (g0 - g1) * d1 / (d1 - d0)
    } else if d1 < 0.0 {
        let (g1, g2) = (cell.slope[1].atan(), cell.slope[2].atan());
        g1 + (g2 - g1) * (-d1) / (d2 - d1)
    } else {
        cell.slope[1].atan()
    }
}

pub(crate) fn viewshed(grid: &Grid, x: f64, y: f64, opts: &Options) -> Result<Vec<bool>, Error> {
    let (rows, cols) = (grid.rows as i64, grid.cols as i64);
    let t = &grid.transform;
    let cv = ((x - t[0]) / t[1]).floor() as i64;
    let rv = ((y - t[3]) / t[5]).floor() as i64;
    if cv < 0 || rv < 0 || cv >= cols || rv >= rows {
        return Err(Error::ObserverOutside);
    }
    let zv = grid.at(rv as usize, cv as usize);
    if zv.is_nan() {
        return Err(Error::ObserverOnNoData);
    }
    let eye = zv + opts.observer_height;
    let res = grid.res();
    let z = |r: i64, c: i64| grid.at(r as usize, c as usize);

    // Only the window within max_distance is visited.
    let reach = opts.max_distance.map(|m| (m / res).floor() as i64 + 1);
    let (r0, r1) = reach.map_or((0, rows - 1), |k| ((rv - k).max(0), (rv + k).min(rows - 1)));
    let (c0, c1) = reach.map_or((0, cols - 1), |k| ((cv - k).max(0), (cv + k).min(cols - 1)));

    let max_d = opts.max_distance;
    let make = |r: i64, c: i64| -> Cell {
        let (dr, dc) = (r - rv, c - cv);
        let zc = z(r, c);
        if (dr == 0 && dc == 0) || zc.is_nan() {
            return Cell::NONE;
        }
        let key = dr * dr + dc * dc;
        // sqrt of an exact integer: correctly rounded, and much faster than hypot
        let dist = (key as f64).sqrt() * res;
        if max_d.map_or(false, |m| dist > m) {
            return Cell::NONE;
        }
        let mut ring = (key as f64).sqrt() as i64;
        while ring * ring > key {
            ring -= 1;
        }
        while (ring + 1) * (ring + 1) <= key {
            ring += 1;
        }
        let [enter, exit] = corners(dr, dc);
        let mut cell = Cell { slope: [0.0; 3], dr: dr as i32, dc: dc as i32, ring: ring as u32 };
        cell.slope[1] = (zc - eye) / dist;
        for (j, (oc, or)) in [(0usize, enter), (2, exit)] {
            let (pc, pr) = (2 * dc + oc, 2 * dr + or);
            let (rr, cc) = (r + or.signum(), c + oc.signum());
            let h = if rr < 0 || cc < 0 || rr >= rows || cc >= cols {
                zc
            } else {
                let v = [zc, z(r, cc), z(rr, c), z(rr, cc)];
                if v.iter().any(|q| q.is_nan()) {
                    zc
                } else {
                    (v[0] + v[1] + v[2] + v[3]) / 4.0
                }
            };
            let d = ((pc * pc + pr * pr) as f64 / 4.0).sqrt() * res;
            cell.slope[j] = (h - eye) / d;
        }
        cell
    };
    // One slot per cell of the window, filled in parallel by rows.
    let width = (c1 - c0 + 1) as usize;
    let mut cells = vec![Cell::NONE; width * (r1 - r0 + 1) as usize];
    cells.par_chunks_mut(width).enumerate().for_each(|(k, row)| {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = make(r0 + k as i64, c0 + j as i64);
        }
    });
    let rings = cells.iter().filter(|c| c.ring != u32::MAX).map(|c| c.ring as usize + 1).max().unwrap_or(1);

    // Three events per window slot; empty slots get key +inf, sort last, and are cut off.
    let mut events = vec![Event { key: u64::MAX, cell: 0 }; 3 * cells.len()];
    events.par_chunks_mut(3).zip(cells.par_iter()).enumerate().for_each(|(i, (ev, c))| {
        if c.ring != u32::MAX {
            for (e, kind) in ev.iter_mut().zip([2u8, 1, 0]) {
                *e = Event::new(c, i as u32, kind);
            }
        }
    });
    let n_events = 3 * cells.iter().filter(|c| c.ring != u32::MAX).count();
    let n_half = 2 * (r1 - r0).max(c1 - c0) + 2;
    if n_half < PSEUDO_ANGLE_LIMIT {
        events.par_sort_unstable_by_key(|e| e.key);
    } else {
        events.par_sort_unstable_by(|a, b| {
            let da = cells[a.cell as usize].dir(2 - a.kind() as usize);
            let db = cells[b.cell as usize].dir(2 - b.kind() as usize);
            cmp_dir(da, db).then(a.kind().cmp(&b.kind()))
        });
    }
    events.truncate(n_events);

    // Split the circle into sectors of equal event counts. A cheap first pass
    // records which cells are active at each sector's start; then the sectors
    // are swept in parallel, each with its own tree. Same order, same answers.
    let n_sec = (rayon::current_num_threads() * 4).clamp(1, events.len().max(1));
    let bounds: Vec<usize> = (0..=n_sec).map(|k| k * events.len() / n_sec).collect();
    let mut pos = vec![u32::MAX; cells.len()];
    let mut active: Vec<u32> = Vec::new();
    let add = |active: &mut Vec<u32>, pos: &mut Vec<u32>, i: u32| {
        if pos[i as usize] == u32::MAX {
            pos[i as usize] = active.len() as u32;
            active.push(i);
        }
    };
    // Cells straddling the start direction (east of V on its row) begin active.
    for (i, c) in cells.iter().enumerate() {
        if c.ring != u32::MAX && c.dr == 0 && c.dc > 0 {
            add(&mut active, &mut pos, i as u32);
        }
    }
    let mut starts = Vec::with_capacity(n_sec);
    for k in 0..n_sec {
        starts.push(active.clone());
        for ev in &events[bounds[k]..bounds[k + 1]] {
            match ev.kind() {
                2 => add(&mut active, &mut pos, ev.cell),
                0 => {
                    let p = pos[ev.cell as usize];
                    if p != u32::MAX {
                        let last = *active.last().unwrap();
                        active.swap_remove(p as usize);
                        if last != ev.cell {
                            pos[last as usize] = p;
                        }
                        pos[ev.cell as usize] = u32::MAX;
                    }
                }
                _ => {}
            }
        }
    }

    let th = opts.target_height;
    let results: Vec<Vec<(usize, bool)>> = (0..n_sec)
        .into_par_iter()
        .map(|k| {
            let mut tree = RingTree::new(rings);
            for &i in &starts[k] {
                tree.insert(cells[i as usize].ring as usize, i, &cells);
            }
            let mut found = Vec::new();
            for ev in &events[bounds[k]..bounds[k + 1]] {
                let cell = &cells[ev.cell as usize];
                match ev.kind() {
                    2 => tree.insert(cell.ring as usize, ev.cell, &cells),
                    0 => tree.remove(cell.ring as usize, ev.cell, &cells),
                    _ => {
                        let idx = ((rv + cell.dr as i64) * cols + cv + cell.dc as i64) as usize;
                        let key = cell.key();
                        let dist = (key as f64).sqrt() * res;
                        let mut target = Target {
                            dir: cell.dir(1),
                            key,
                            ring: cell.ring as usize,
                            slope: (grid.z[idx] + th - eye) / dist,
                            theta: None,
                            grad: 0.0,
                        };
                        found.push((idx, !tree.blocked(&cells, &mut target)));
                    }
                }
            }
            found
        })
        .collect();

    let mut out = vec![false; grid.rows * grid.cols];
    out[(rv * cols + cv) as usize] = true;
    for (idx, v) in results.into_iter().flatten() {
        out[idx] = v;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_angle_orders_like_exact_angles() {
        let mut dirs = Vec::new();
        for x in -12i64..=12 {
            for y in -12i64..=12 {
                if (x, y) != (0, 0) {
                    dirs.push((x, y));
                }
            }
        }
        for &a in &dirs {
            for &b in &dirs {
                let by_key = pseudo_angle(a).total_cmp(&pseudo_angle(b));
                assert_eq!(by_key, cmp_dir(a, b), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn pseudo_angle_is_exact_far_out() {
        // nearest distinct directions at a large radius
        let n = 2_000_000i64;
        let (a, b) = ((n, n - 1), (n - 1, n - 2));
        assert_eq!(pseudo_angle(a).total_cmp(&pseudo_angle(b)), cmp_dir(a, b));
        assert_ne!(pseudo_angle(a), pseudo_angle(b));
        // equal directions give identical keys
        assert_eq!(pseudo_angle(a), pseudo_angle((2 * n, 2 * n - 2)));
        assert_eq!(pseudo_angle((n - 1, n - 1)), pseudo_angle((1, 1)));
    }
}
