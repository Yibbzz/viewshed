//! Cast-shadow map of a raw little-endian f64 raster, for the conformance suite.
//! Usage: raw_shadow <heights.f64> <rows> <cols> <x0> <res> <y0> <out.u8> <target_h>
//!        <curvature 0|1> <refraction> <earth_radius> (angles <azimuth> <elevation> | vec <dx> <dy> <dz>)
//! Prints the computation time in seconds and the sun vector used on stdout.
use std::{env, fs, time::Instant};
use viewshed::{shadow, shadow_vec, Grid, ShadowOptions, Sun};

fn main() {
    let a: Vec<String> = env::args().collect();
    let bytes = fs::read(&a[1]).expect("read heights");
    let z: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let p = |i: usize| a[i].parse::<f64>().unwrap();
    let (rows, cols) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let grid = Grid::new(&z, rows, cols, [p(4), p(5), 0.0, p(6), 0.0, -p(5)]).unwrap();
    let opts = ShadowOptions { target_height: p(8), curvature: a[9] == "1", refraction: p(10), earth_radius: p(11) };
    let t = Instant::now();
    let (lit, v) = if a[12] == "angles" {
        let sun = Sun { azimuth_deg: p(13), elevation_deg: p(14) };
        (shadow(&grid, sun, &opts).unwrap(), sun.direction(&grid).unwrap())
    } else {
        let v = [p(13), p(14), p(15)];
        (shadow_vec(&grid, v, &opts).unwrap(), v)
    };
    println!("{} {:?} {:?} {:?}", t.elapsed().as_secs_f64(), v[0], v[1], v[2]);
    fs::write(&a[7], lit.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
}
