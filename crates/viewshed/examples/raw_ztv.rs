//! ZTV of a raw little-endian f64 raster, for the conformance suite.
//! Usage: raw_ztv <heights.f64> <rows> <cols> <x0> <res> <y0> <observer_h>
//!        <max_distance, negative = none> <forward|reciprocal> <targets "x,y,h;x,y,h"> <out.u32>
//!        [curvature 0|1] [refraction] [earth_radius]   (curvature is off unless given)
//! Prints the computation time in seconds on stdout.
use std::{env, fs, time::Instant};
use viewshed::{ztv, Grid, ZtvMethod, ZtvOptions, ZtvTarget};

fn main() {
    let a: Vec<String> = env::args().collect();
    let bytes = fs::read(&a[1]).expect("read heights");
    let z: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let p = |i: usize| a[i].parse::<f64>().unwrap();
    let (rows, cols) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let grid = Grid::new(&z, rows, cols, [p(4), p(5), 0.0, p(6), 0.0, -p(5)]).unwrap();
    let targets: Vec<ZtvTarget> = a[10]
        .split(';')
        .map(|t| {
            let v: Vec<f64> = t.split(',').map(|q| q.parse().unwrap()).collect();
            ZtvTarget { x: v[0], y: v[1], height: v[2] }
        })
        .collect();
    let opts = ZtvOptions {
        observer_height: p(7),
        max_distance: if p(8) >= 0.0 { Some(p(8)) } else { None },
        method: if a[9] == "reciprocal" { ZtvMethod::Reciprocal } else { ZtvMethod::Forward },
        curvature: a.len() > 14 && a[12] == "1",
        refraction: if a.len() > 14 { p(13) } else { 0.0 },
        earth_radius: if a.len() > 14 { p(14) } else { viewshed::MEAN_EARTH_RADIUS },
    };
    let t = Instant::now();
    let counts = ztv(&grid, &targets, &opts).unwrap();
    println!("{}", t.elapsed().as_secs_f64());
    fs::write(&a[11], counts.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
