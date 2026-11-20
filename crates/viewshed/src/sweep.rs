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
//! EXIT events sorted by exact angle (integer cross products, so ties are
//! exact); at equal angles EXIT comes before CENTER before ENTER. Active cells
//! live in a segment tree over integer distance rings. Each node keeps the largest
//! upper bound (max of the three gradients) and the largest lower bound (min of
//! the three) of its active cells, so a query skips subtrees that can't block
//! and stops at the first subtree that must.

use std::cmp::Ordering;
use std::f64::consts::PI;

use rayon::prelude::*;

use crate::{Error, Grid, Options};

/// Directions in doubled index units (so corners are integers), y pointing north.
type Dir = (i64, i64);

/// One cell's three samples. 64 bytes; everything else is derived.
#[derive(Clone, Copy)]
struct Cell {
    angle: [f64; 3], // ENTER, centre, EXIT
    grad: [f64; 3],
    dr: i32, // offset from the observer cell, rows (south +)
    dc: i32, // and columns
    ring: u32, // floor of the distance in cells; u32::MAX = no cell (no data or out of range)
}

impl Cell {
    const NONE: Cell = Cell { angle: [0.0; 3], grad: [0.0; 3], dr: 0, dc: 0, ring: u32::MAX };

    fn key(&self) -> i64 {
        let (dr, dc) = (self.dr as i64, self.dc as i64);
        dr * dr + dc * dc
    }

    fn hi(&self) -> f64 {
        self.grad[0].max(self.grad[1]).max(self.grad[2])
    }

    fn lo(&self) -> f64 {
        self.grad[0].min(self.grad[1]).min(self.grad[2])
    }
}

#[derive(Clone, Copy)]
struct Event {
    angle: f64,
    cell: u32,
    kind: u8, // 0 = EXIT, 1 = CENTER, 2 = ENTER; sample slot = 2 - kind
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

    /// Does an active cell strictly closer than the target (ring `ring`, squared
    /// distance `key`) block gradient `gt` at angle `theta`?
    fn blocked(&self, cells: &[Cell], ring: usize, key: i64, theta: f64, gt: f64) -> bool {
        let blocks = |m: u32| gradient_at(&cells[m as usize], theta) > gt;
        // Rings inside the target's: all strictly closer.
        let mut stack = [(0usize, 0usize, 0usize); 128];
        let mut top = 1;
        stack[0] = (1, 0, self.size);
        while top > 0 {
            top -= 1;
            let (node, l, r) = stack[top];
            if l >= ring || self.hi[node] <= gt {
                continue;
            }
            if r <= ring && self.lo[node] > gt {
                return true;
            }
            if r - l == 1 {
                if self.members[l].iter().any(|&m| blocks(m)) {
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
        self.members[ring].iter().any(|&m| cells[m as usize].key() < key && blocks(m))
    }
}

fn rel(a: f64, theta: f64) -> f64 {
    (a - theta + PI).rem_euclid(2.0 * PI) - PI
}

/// A cell's gradient at direction `theta`, interpolated in angle.
fn gradient_at(cell: &Cell, theta: f64) -> f64 {
    let (d0, d1, d2) = (rel(cell.angle[0], theta), rel(cell.angle[1], theta), rel(cell.angle[2], theta));
    let g = &cell.grad;
    if d1 > 0.0 {
        g[1] + (g[0] - g[1]) * d1 / (d1 - d0)
    } else if d1 < 0.0 {
        g[1] + (g[2] - g[1]) * (-d1) / (d2 - d1)
    } else {
        g[1]
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
        let mut cell = Cell { angle: [0.0; 3], grad: [0.0; 3], dr: dr as i32, dc: dc as i32, ring: ring as u32 };
        for (j, off) in [(0usize, Some(enter)), (1, None), (2, Some(exit))] {
            let (pc, pr) = match off {
                Some((oc, or)) => (2 * dc + oc, 2 * dr + or),
                None => (2 * dc, 2 * dr),
            };
            cell.angle[j] = angle((pc, -pr));
            let h = match off {
                None => zc,
                Some((oc, or)) => {
                    let (rr, cc) = (r + or.signum(), c + oc.signum());
                    if rr < 0 || cc < 0 || rr >= rows || cc >= cols {
                        zc
                    } else {
                        let v = [zc, z(r, cc), z(rr, c), z(rr, cc)];
                        if v.iter().any(|q| q.is_nan()) {
                            zc
                        } else {
                            (v[0] + v[1] + v[2] + v[3]) / 4.0
                        }
                    }
                }
            };
            let d = ((pc * pc + pr * pr) as f64 / 4.0).sqrt() * res;
            cell.grad[j] = ((h - eye) / d).atan();
        }
        cell
    };
    // Exact direction of a cell's sample (0 = ENTER, 1 = centre, 2 = EXIT), recomputed
    // only to break near-ties in the event order.
    let dir_of = |cell: &Cell, slot: usize| -> Dir {
        let (dr, dc) = (cell.dr as i64, cell.dc as i64);
        let (oc, or) = match slot {
            1 => (0, 0),
            s => corners(dr, dc)[s / 2],
        };
        (2 * dc + oc, -(2 * dr + or))
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

    // Sort by float angle, then fix the order within runs of nearly equal angles.
    // Two different directions within n half-cells of the observer are at least
    // 1/(2 n^2) radians apart, and atan2 is good to ~1e-15. So when 1/(2 n^2) is well
    // above the 1e-12 window, every event in a run has exactly the same direction
    // and only the kind order (EXIT, CENTER, ENTER) needs fixing; otherwise fall
    // back to exact cross products.
    // Three events per window slot; empty slots get angle +inf, sort last, and are cut off.
    let mut events = vec![Event { angle: f64::INFINITY, cell: 0, kind: 0 }; 3 * cells.len()];
    events.par_chunks_mut(3).zip(cells.par_iter()).enumerate().for_each(|(i, (ev, c))| {
        if c.ring != u32::MAX {
            for (e, kind) in ev.iter_mut().zip([2u8, 1, 0]) {
                *e = Event { angle: c.angle[2 - kind as usize], cell: i as u32, kind };
            }
        }
    });
    let n_events = 3 * cells.iter().filter(|c| c.ring != u32::MAX).count();
    // Angles are >= 0, so their bit patterns sort like the numbers. Kind order
    // within equal angles is settled by the exact pass below.
    events.par_sort_unstable_by_key(|e| e.angle.to_bits());
    events.truncate(n_events);
    const WINDOW: f64 = 1e-12;
    let n_half = 2 * (r1 - r0).max(c1 - c0) as i64 + 2;
    let same_dir_runs = 1.0 / (2.0 * (n_half as f64).powi(2)) > 1e3 * WINDOW;
    let exact = |a: &Event, b: &Event| {
        let da = dir_of(&cells[a.cell as usize], 2 - a.kind as usize);
        let db = dir_of(&cells[b.cell as usize], 2 - b.kind as usize);
        cmp_dir(da, db).then(a.kind.cmp(&b.kind))
    };
    let mut i = 0;
    while i < events.len() {
        let mut j = i + 1;
        while j < events.len() && events[j].angle - events[j - 1].angle <= WINDOW {
            j += 1;
        }
        if j - i > 1 {
            if same_dir_runs {
                events[i..j].sort_unstable_by_key(|e| e.kind);
            } else {
                events[i..j].sort_unstable_by(exact);
            }
        }
        i = j;
    }

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
            match ev.kind {
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
                match ev.kind {
                    2 => tree.insert(cell.ring as usize, ev.cell, &cells),
                    0 => tree.remove(cell.ring as usize, ev.cell, &cells),
                    _ => {
                        let idx = ((rv + cell.dr as i64) * cols + cv + cell.dc as i64) as usize;
                        let key = cell.key();
                        let dist = (key as f64).sqrt() * res;
                        let gt = ((grid.z[idx] + th - eye) / dist).atan();
                        let hidden = tree.blocked(&cells, cell.ring as usize, key, cell.angle[1], gt);
                        found.push((idx, !hidden));
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
