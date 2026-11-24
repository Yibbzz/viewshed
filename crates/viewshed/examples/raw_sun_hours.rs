//! Sun hours of a raw little-endian f64 raster, for the conformance suite.
//! Usage: raw_sun_hours <heights.f64> <rows> <cols> <x0> <res> <y0> <out.f64> <target_h>
//!        <curvature 0|1> <refraction> <earth_radius>
//!        (between <lat> <lon> <start> <end> <step> <convergence> | positions <az> <el> <hours> ...)
//!    or: raw_sun_hours samples <lat> <lon> <start> <end> <step> <convergence>
//!    or: raw_sun_hours sunpos   (reads "lat lon unix" lines on stdin, prints "az el el_true" with {:?})
use std::io::BufRead;
use std::{env, fs, time::Instant};
use viewshed::{sun_hours, sun_hours_between, sun_position, sun_samples, Grid, ShadowOptions, Sun};

fn main() {
    let a: Vec<String> = env::args().collect();
    if a[1] == "sunpos" {
        for line in std::io::stdin().lock().lines() {
            let v: Vec<f64> = line.unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
            let p = sun_position(v[0], v[1], v[2]).unwrap();
            println!("{:?} {:?} {:?}", p.azimuth_deg, p.elevation_deg, p.elevation_true_deg);
        }
        return;
    }
    if a[1] == "samples" {
        // samples <lat> <lon> <start> <end> <step> <convergence>: "az el hours" per line
        let p = |i: usize| a[i].parse::<f64>().unwrap();
        for (s, w) in sun_samples(p(2), p(3), p(4), p(5), p(6), p(7)).unwrap() {
            println!("{:?} {:?} {:?}", s.azimuth_deg, s.elevation_deg, w);
        }
        return;
    }
    let bytes = fs::read(&a[1]).expect("read heights");
    let z: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let p = |i: usize| a[i].parse::<f64>().unwrap();
    let (rows, cols) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let grid = Grid::new(&z, rows, cols, [p(4), p(5), 0.0, p(6), 0.0, -p(5)]).unwrap();
    let opts = ShadowOptions { target_height: p(8), curvature: a[9] == "1", refraction: p(10), earth_radius: p(11) };
    let t = Instant::now();
    let h = if a[12] == "between" {
        sun_hours_between(&grid, p(13), p(14), p(15), p(16), p(17), p(18), &opts).unwrap()
    } else {
        let pos: Vec<(Sun, f64)> = a[13..].chunks(3).map(|c| (Sun { azimuth_deg: c[0].parse().unwrap(), elevation_deg: c[1].parse().unwrap() }, c[2].parse().unwrap())).collect();
        sun_hours(&grid, &pos, &opts).unwrap()
    };
    println!("{}", t.elapsed().as_secs_f64());
    fs::write(&a[7], h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
