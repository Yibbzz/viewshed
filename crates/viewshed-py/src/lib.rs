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
use vs::{Grid, Options, ShadowOptions, Sun, Surface, ZtvMethod, ZtvOptions, ZtvTarget};

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
    let (rows, cols) = (dem.shape()[0], dem.shape()[1]);
    let z = dem.as_slice()?;
    let opts = Options {
        observer_height,
        target_height,
        max_distance,
        surface: surface(surface_name)?,
        curvature,
        refraction,
        earth_radius,
    };
    let seen = py
        .detach(|| {
            let grid = Grid::new(z, rows, cols, transform)?;
            vs::viewshed(&grid, x, y, &opts)
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

#[pymodule]
fn _viewshed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(viewshed_py, m)?)?;
    m.add_function(wrap_pyfunction!(ztv_py, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_py, m)?)?;
    m.add("MEAN_EARTH_RADIUS", vs::MEAN_EARTH_RADIUS)?;
    Ok(())
}
