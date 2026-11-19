//! Viewshed of a raw little-endian f64 raster, for cross-checking with the Python package.
//! Usage: raw_viewshed <heights.f64> <rows> <cols> <x0> <res> <y0> <ox> <oy> <obs_h> <out.u8>
use std::{env, fs};
use viewshed::{viewshed, Grid, Options};

fn main() {
    let a: Vec<String> = env::args().collect();
    let bytes = fs::read(&a[1]).expect("read heights");
    let z: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    let p = |i: usize| a[i].parse::<f64>().unwrap();
    let (rows, cols) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let grid = Grid::new(&z, rows, cols, [p(4), p(5), 0.0, p(6), 0.0, -p(5)]).unwrap();
    let opts = Options { observer_height: p(9), ..Options::default() };
    let seen = viewshed(&grid, p(7), p(8), &opts).unwrap();
    fs::write(&a[10], seen.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
}
