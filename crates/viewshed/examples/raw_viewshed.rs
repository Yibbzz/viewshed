//! Viewshed of a raw little-endian f64 raster, for the conformance suite.
//! Usage: raw_viewshed <heights.f64> <rows> <cols> <x0> <res> <y0> <ox> <oy> <obs_h> <out.u8>
//!        [target_h] [max_distance, negative = none] [rviewshed|bilinear]
//!        [curvature 0|1] [refraction] [earth_radius]   (curvature is off unless given)
//! Prints the computation time in seconds (excluding file I/O) on stdout.
use std::{env, fs, time::Instant};
use viewshed::{viewshed, Grid, Options, Surface};

fn main() {
    let a: Vec<String> = env::args().collect();
    let bytes = fs::read(&a[1]).expect("read heights");
    let z: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let p = |i: usize| a[i].parse::<f64>().unwrap();
    let (rows, cols) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let grid = Grid::new(&z, rows, cols, [p(4), p(5), 0.0, p(6), 0.0, -p(5)]).unwrap();
    let mut opts = Options {
        observer_height: p(9),
        curvature: false,
        ..Options::default()
    };
    if a.len() > 11 {
        opts.target_height = p(11);
    }
    if a.len() > 12 && p(12) >= 0.0 {
        opts.max_distance = Some(p(12));
    }
    if a.len() > 13 && a[13] == "bilinear" {
        opts.surface = Surface::Bilinear;
    }
    if a.len() > 16 {
        opts.curvature = a[14] == "1";
        opts.refraction = p(15);
        opts.earth_radius = p(16);
    }
    let t = Instant::now();
    let seen = viewshed(&grid, p(7), p(8), &opts).unwrap();
    println!("{}", t.elapsed().as_secs_f64());
    fs::write(&a[10], seen.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
}
