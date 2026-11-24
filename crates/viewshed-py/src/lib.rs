//! Python bindings for the `viewshed` crate. maturin builds this into the
//! `viewshed._viewshed` extension module; `python/viewshed/__init__.py` is the
//! public API (argument checks and defaults live there).
//!
//! Every call borrows the NumPy array without copying (it must be C-contiguous
//! float64; the Python layer makes sure) and releases the GIL while it computes.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ::viewshed as vs;
use vs::{Grid, LosOptions, Options, Point, ShadowOptions, Sun, Surface, ZtvMethod, ZtvOptions, ZtvTarget};

fn value_error(e: vs::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn surface(name: &str) -> PyResult<Surface> {
    match name {
        "rviewshed" => Ok(Surface::RViewshed),
        "bilinear" => Ok(Surface::Bilinear),
        _ => Err(PyValueError::new_err(format!("unknown surface {name:?}: use 'rviewshed' or 'bilinear'"))),
    }
}

/// Visible mask (bool, the DEM's shape) from one observer.
#[pyfunction(name = "viewshed")]
#[allow(clippy::too_many_arguments)]
fn viewshed_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    x: f64,
    y: f64,
    observer_height: f64,
    target_height: f64,
    max_distance: Option<f64>,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
    surface_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let opts = Options {
        observer_height,
        target_height,
        max_distance,
        surface: surface(surface_name)?,
        curvature,
        refraction,
        earth_radius,
    };
    mask(py, dem, transform, x, y, &opts, vs::viewshed)
}

/// Cells (bool, the DEM's shape) from which the target at (x, y) is visible.
#[pyfunction(name = "reverse_viewshed")]
#[allow(clippy::too_many_arguments)]
fn reverse_viewshed_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    x: f64,
    y: f64,
    observer_height: f64,
    target_height: f64,
    max_distance: Option<f64>,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
    surface_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let opts = Options {
        observer_height,
        target_height,
        max_distance,
        surface: surface(surface_name)?,
        curvature,
        refraction,
        earth_radius,
    };
    mask(py, dem, transform, x, y, &opts, vs::reverse_viewshed)
}

type MaskFn = fn(&Grid, f64, f64, &Options) -> Result<Vec<bool>, vs::Error>;

fn mask<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    x: f64,
    y: f64,
    opts: &Options,
    f: MaskFn,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let seen = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            f(&grid, x, y, opts)
        })
        .map_err(value_error)?;
    Ok(PyArray1::from_vec(py, seen).reshape([rows, cols])?.into_any())
}

/// For every cell, how many targets (x, y, height) are visible (uint32, the DEM's shape).
#[pyfunction(name = "ztv")]
#[allow(clippy::too_many_arguments)]
fn ztv_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    targets: Vec<(f64, f64, f64)>,
    observer_height: f64,
    max_distance: Option<f64>,
    method: &str,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let method = match method {
        "forward" => ZtvMethod::Forward,
        "reciprocal" => ZtvMethod::Reciprocal,
        _ => return Err(PyValueError::new_err(format!("unknown method {method:?}: use 'forward' or 'reciprocal'"))),
    };
    let targets: Vec<ZtvTarget> = targets.into_iter().map(|(x, y, height)| ZtvTarget { x, y, height }).collect();
    let opts = ZtvOptions { observer_height, max_distance, method, curvature, refraction, earth_radius };
    let counts = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            vs::ztv(&grid, &targets, &opts)
        })
        .map_err(value_error)?;
    Ok(PyArray1::from_vec(py, counts).reshape([rows, cols])?.into_any())
}

/// Lit mask (bool, the DEM's shape) for a sun at (azimuth, elevation) degrees.
#[pyfunction(name = "shadow")]
#[allow(clippy::too_many_arguments)]
fn shadow_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    azimuth: f64,
    elevation: f64,
    target_height: f64,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let sun = Sun { azimuth_deg: azimuth, elevation_deg: elevation };
    let opts = ShadowOptions { target_height, curvature, refraction, earth_radius };
    let lit = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            vs::shadow(&grid, sun, &opts)
        })
        .map_err(value_error)?;
    Ok(PyArray1::from_vec(py, lit).reshape([rows, cols])?.into_any())
}

/// Hours of direct sun per cell (float64, the DEM's shape; NaN on no-data) for sun
/// positions (grid azimuth, elevation, weight in hours).
#[pyfunction(name = "sun_hours")]
#[allow(clippy::too_many_arguments)]
fn sun_hours_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    positions: Vec<(f64, f64, f64)>,
    target_height: f64,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let suns: Vec<(Sun, f64)> =
        positions.into_iter().map(|(a, e, w)| (Sun { azimuth_deg: a, elevation_deg: e }, w)).collect();
    let opts = ShadowOptions { target_height, curvature, refraction, earth_radius };
    let hours = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            vs::sun_hours(&grid, &suns, &opts)
        })
        .map_err(value_error)?;
    Ok(PyArray1::from_vec(py, hours).reshape([rows, cols])?.into_any())
}

/// (azimuth from true north, apparent elevation, true elevation) in degrees.
#[pyfunction(name = "sun_position")]
fn sun_position_py(lat: f64, lon: f64, unix_seconds: f64) -> PyResult<(f64, f64, f64)> {
    let p = vs::sun_position(lat, lon, unix_seconds).map_err(value_error)?;
    Ok((p.azimuth_deg, p.elevation_deg, p.elevation_true_deg))
}

/// Midpoint samples between two times: (grid azimuth, apparent elevation, weight in hours),
/// sun-down samples left out.
#[pyfunction(name = "sun_samples")]
fn sun_samples_py(lat: f64, lon: f64, start: f64, end: f64, step: f64, convergence: f64) -> PyResult<Vec<(f64, f64, f64)>> {
    let s = vs::sun_samples(lat, lon, start, end, step, convergence).map_err(value_error)?;
    Ok(s.into_iter().map(|(sun, w)| (sun.azimuth_deg, sun.elevation_deg, w)).collect())
}

type PyPair = ((f64, f64, f64), (f64, f64, f64));

/// Intervisibility (bool per pair) of point pairs ((x, y, h), (x, y, h)), exact.
#[pyfunction(name = "intervisible")]
fn intervisible_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    pairs: Vec<PyPair>,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let pts: Vec<(Point, Point)> = pairs
        .into_iter()
        .map(|((ax, ay, ah), (bx, by, bh))| (Point { x: ax, y: ay, height: ah }, Point { x: bx, y: by, height: bh }))
        .collect();
    let opts = LosOptions { curvature, refraction, earth_radius };
    let seen = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            vs::intervisible(&grid, &pts, &opts)
        })
        .map_err(value_error)?;
    Ok(PyArray1::from_vec(py, seen).into_any())
}

/// How many of the observers (x, y, eye height) see each cell (uint32, the DEM's shape).
#[pyfunction(name = "cumulative")]
#[allow(clippy::too_many_arguments)]
fn cumulative_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    observers: Vec<(f64, f64, f64)>,
    target_height: f64,
    max_distance: Option<f64>,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
    surface_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let obs: Vec<Point> = observers.into_iter().map(|(x, y, height)| Point { x, y, height }).collect();
    let opts = Options { target_height, max_distance, surface: surface(surface_name)?, curvature, refraction, earth_radius, ..Options::default() };
    let counts = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            vs::cumulative(&grid, &obs, &opts)
        })
        .map_err(value_error)?;
    Ok(PyArray1::from_vec(py, counts).reshape([rows, cols])?.into_any())
}

/// Fraction of the every-stride-th-cell observers that see each cell (float64; NaN on no-data),
/// and the number of observers.
#[pyfunction(name = "visibility_index")]
#[allow(clippy::too_many_arguments)]
fn visibility_index_py<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    transform: [f64; 6],
    stride: usize,
    observer_height: f64,
    target_height: f64,
    max_distance: Option<f64>,
    curvature: bool,
    refraction: f64,
    earth_radius: f64,
    surface_name: &str,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let opts = Options { observer_height, target_height, max_distance, surface: surface(surface_name)?, curvature, refraction, earth_radius };
    let (frac, k) = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            vs::visibility_index(&grid, stride, &opts)
        })
        .map_err(value_error)?;
    Ok((PyArray1::from_vec(py, frac).reshape([rows, cols])?.into_any(), k))
}

#[pymodule]
fn _viewshed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(viewshed_py, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_viewshed_py, m)?)?;
    m.add_function(wrap_pyfunction!(ztv_py, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_py, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_py, m)?)?;
    m.add_function(wrap_pyfunction!(intervisible_py, m)?)?;
    m.add_function(wrap_pyfunction!(sun_hours_py, m)?)?;
    m.add_function(wrap_pyfunction!(sun_position_py, m)?)?;
    m.add_function(wrap_pyfunction!(sun_samples_py, m)?)?;
    m.add("MEAN_EARTH_RADIUS", vs::MEAN_EARTH_RADIUS)?;
    Ok(())
}
