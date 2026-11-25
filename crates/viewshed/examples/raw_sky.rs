//! Sky view factor / openness from raw f64 heights (conformance tests).
//! Usage: raw_sky <heights.f64> <rows> <cols> <x0> <res> <y0> <out.f64> <kind: vs|iso|opos|oneg> <azimuths>
//!        [target_h curvature(0/1) refraction earth_radius]   (sky view factor only)
//! RAW_SKY_MAX_DISTANCE: the distance cut.
use std::{env, fs};
use viewshed::{openness, sky_view_factor, Grid, Openness, OpennessOptions, SkyOptions, SkyViewFactor};

fn main() {
    let a: Vec<String> = env::args().collect();
    let bytes = fs::read(&a[1]).expect("read heights");
    let z: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let p = |i: usize| a[i].parse::<f64>().unwrap();
    let (rows, cols) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let grid = Grid::new(&z, rows, cols, [p(4), p(5), 0.0, p(6), 0.0, -p(5)]).unwrap();
    let k: usize = a[9].parse().unwrap();
    let max_distance = env::var("RAW_SKY_MAX_DISTANCE").ok().map(|v| v.parse::<f64>().expect("max distance"));
    let out = match a[8].as_str() {
        "vs" | "iso" => {
            let mut o = SkyOptions { azimuths: k, max_distance, ..SkyOptions::default() };
            if a.len() > 13 {
                o.target_height = p(10);
                o.curvature = a[11] == "1";
                o.refraction = p(12);
                o.earth_radius = p(13);
            }
            let kind = if a[8] == "vs" { SkyViewFactor::VisibleSky } else { SkyViewFactor::Isotropic };
            sky_view_factor(&grid, kind, &o).unwrap()
        }
        "opos" | "oneg" => {
            let kind = if a[8] == "opos" { Openness::Positive } else { Openness::Negative };
            openness(&grid, kind, &OpennessOptions { azimuths: k, max_distance }).unwrap()
        }
        other => panic!("kind {other}"),
    };
    fs::write(&a[7], out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
