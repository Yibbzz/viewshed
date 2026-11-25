use std::time::Instant;
use viewshed::{fresnel_clearance, fresnel_stats, intervisible, Grid, LosOptions, Point};
fn main() {
    let n = 2001usize;
    let mut s: u64 = 7;
    let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
    let mut z = vec![0.0; n * n];
    for r in 0..n { for c in 0..n {
        let (x, y) = (r as f64, c as f64);
        z[r * n + c] = 300.0 * (x * 0.004).sin() * (y * 0.003).cos() + 60.0 * (x * 0.03 + y * 0.02).sin() + 8.0 * (x * 0.31).sin() * (y * 0.27).cos();
    }}
    for v in z.iter_mut() { *v += 2.0 * rnd(); }
    let gt = [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0];
    let g = Grid::new(&z, n, n, gt).unwrap();
    let mut pairs = Vec::new();
    for _ in 0..20000 {
        let (x, y) = (500000.0 + 100.0 + rnd() * 19800.0, 6000000.0 - 100.0 - rnd() * 19800.0);
        let (a, d) = (rnd() * std::f64::consts::TAU, 200.0 + rnd() * 5800.0);
        let (x2, y2) = ((x + d * a.cos()).clamp(500010.0, 520000.0), (y + d * a.sin()).clamp(5980010.0, 6000000.0 - 10.0));
        pairs.push((Point { x, y, height: 10.0 + 20.0 * rnd() }, Point { x: x2, y: y2, height: 10.0 + 20.0 * rnd() }));
    }
    for (label, hmin, hspan) in [("masts 10-30 m (los_bench)", 10.0, 20.0), ("masts 40-100 m", 40.0, 60.0)] {
        let mut ps = pairs.clone();
        let mut k = 0u64;
        for (a, b) in ps.iter_mut() {
            k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            a.height = hmin + hspan * ((k >> 11) as f64 / (1u64 << 53) as f64);
            k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            b.height = hmin + hspan * ((k >> 11) as f64 / (1u64 << 53) as f64);
        }
        let t = Instant::now();
        let v = intervisible(&g, &ps, &LosOptions::default()).unwrap();
        let dt = t.elapsed().as_secs_f64();
        println!("{label}: intervisible {:.1} us/pair, visible {:.1}%", dt * 1e6 / ps.len() as f64, 100.0 * v.iter().filter(|&&b| b).count() as f64 / v.len() as f64);
        for f in [1e8, 2.4e9, 6e9] {
            let _ = fresnel_stats();
            let t = Instant::now();
            let c = fresnel_clearance(&g, &ps, f, &LosOptions::default()).unwrap();
            let dt = t.elapsed().as_secs_f64();
            let (ne, np) = fresnel_stats();
            let clear = c.iter().filter(|x| x.ratio >= 0.6).count() as f64 / c.len() as f64;
            println!("{label}: fresnel {:.1} GHz {:.1} us/pair, >=0.6 clear {:.1}%, exact fallback {} links, {:.0} nodes/link", f / 1e9, dt * 1e6 / ps.len() as f64, 100.0 * clear, ne, np as f64 / ps.len() as f64);
        }
    }
}
