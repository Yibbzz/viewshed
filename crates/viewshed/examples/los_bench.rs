use std::time::Instant;
use viewshed::{intervisible, Grid, LosOptions, Point};
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
    let t = Instant::now();
    let v = intervisible(&g, &pairs, &LosOptions::default()).unwrap();
    let dt = t.elapsed().as_secs_f64();
    println!("{} pairs in {:.2} s ({:.1} us/pair), visible {:.1}%", pairs.len(), dt, dt * 1e6 / pairs.len() as f64, 100.0 * v.iter().filter(|&&b| b).count() as f64 / v.len() as f64);
}
