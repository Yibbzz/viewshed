//! Horizon map of a raw little-endian f64 raster, for the conformance suite.
//! Usage: raw_horizon <heights.f64> <rows> <cols> <x0> <res> <y0> <out.f64> <target_h>
//!        <curvature 0|1> <refraction> <earth_radius> (az <azimuth> | vec <dx> <dy>)
//! az: writes angles in degrees; vec: writes the horizon rise Hs. Prints the time in seconds.
use std::{env, fs, time::Instant};
use viewshed::{horizon_map, horizon_rise_vec, Grid, HorizonOptions};

fn main() {
    let a: Vec<String> = env::args().collect();
    let bytes = fs::read(&a[1]).expect("read heights");
    let z: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let p = |i: usize| a[i].parse::<f64>().unwrap();
    let (rows, cols) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let grid = Grid::new(&z, rows, cols, [p(4), p(5), 0.0, p(6), 0.0, -p(5)]).unwrap();
    let opts = HorizonOptions { target_height: p(8), curvature: a[9] == "1", refraction: p(10), earth_radius: p(11) };
    let t = Instant::now();
    let h = if a[12] == "az" { horizon_map(&grid, p(13), &opts).unwrap() } else { horizon_rise_vec(&grid, [p(13), p(14)], &opts).unwrap() };
    println!("{}", t.elapsed().as_secs_f64());
    fs::write(&a[7], h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
