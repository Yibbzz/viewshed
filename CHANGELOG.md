# Changelog

All notable changes to `viewshed` (the Rust crate and the Python package, which share a version).

## Unreleased

### Added
- **Horizon maps.** `viewshed.horizon_map(dem, transform, azimuth, target_height=0.0, ...)` in Python, `viewshed::horizon_map(grid, azimuth_deg, &HorizonOptions)` in Rust: the horizon's elevation angle in degrees for every cell, exact to within 1e-9 in the tangent (certified per cell). −90 with only no-data ahead, NaN on no-data, not clamped at 0. Consistent with `shadow`. Checked against a new exact (algebraic) horizon oracle on tie-heavy and random rasters, sliver-prone directions, no-data and curvature.
- **Exact point-to-point line of sight.** `viewshed.intervisible(dem, transform, a, b, ...)` in Python (one pair, or arrays of points, one-to-many broadcast), `viewshed::intervisible(grid, &[(Point, Point)], &LosOptions)` in Rust. Exact on the bilinear surface for endpoints anywhere, with curvature; order-independent; equal to the bilinear viewshed for cell-centre targets. Checked against a new exact oracle (`oracle.line_of_sight`) on tie-heavy rasters with endpoints on cell centres, cell corners and grid lines. About 50 µs per link (2,001 × 2,001, links up to 6 km, 4 cores).
- `Error::PointOutside`, `PointOnNoData` and `InvalidPointHeight`, each naming the pair.

### Deprecated
- `viewshed::line_of_sight` (Rust): it samples the line (4 points per cell, with a tolerance), so it can miss thin blockers. Use `intervisible`.

## 0.3.0 (2026-11-24)

### Added
- **Sun hours.** `viewshed.sun_hours(dem, transform, start, end, step=15 min, lat, lon, crs=None, grid_convergence=None, ...)` and `viewshed.sun_position(lat, lon, time)` in Python; `sun_hours_between`, `sun_hours`, `sun_samples` and `sun_position` in Rust. Hours of direct sun per cell: exact shadow maps summed over midpoint time samples, with the sun's position from the NOAA/Meeus equations (within about 0.02° of NREL SPA). Grid north vs true north via `grid_convergence`, or from the raster's `crs` with pyproj in Python. Times must be timezone-aware. New `Error` variants `InvalidLocation`, `InvalidTime` and `InvalidWeight` (`Error` is `#[non_exhaustive]`, so this isn't breaking).
- **Reverse viewshed.** `viewshed.reverse_viewshed(dem, transform, target, target_height=0.0, observer_height=1.6, ...)` in Python, `viewshed::reverse_viewshed` in Rust. Returns the cells from which a target at (x, y) can be seen: for every cell, the same answer as a viewshed from that cell.
  - `surface="bilinear"`: one exact viewshed from the target with the heights swapped (sightlines, the blocking rule and the curvature drop are symmetric). Checked against the exact oracle run forwards from every cell, on tie-heavy rasters with holes and curvature, with the target on and off a cell centre.
  - `surface="rviewshed"` (default): that model isn't symmetric, so every cell looks at the target in turn, as `ztv` does. Exact for the model, but slower (about 5× an ordinary viewshed at 2,001 × 2,001).

### Changed
- Shadow maps about 2× faster (2,001 × 2,001, 4 cores: 0.6–1.1 s → 0.3–0.5 s), identical output: a cell's first stretch towards the sun is certified lit without walking it wherever the terrain there is no steeper than the sun (a per-patch slope bound), and the per-patch bounds are computed once.

### Fixed
- `surface="bilinear"`: an observer exactly on a cell centre or on a line between cell centres was rejected ("no height data under the observer") when the patch to its lower right had no data, even though a neighbouring patch with data defines the ground there. Any patch containing the point now gives its height (they agree along shared edges). Same fix in the reference oracle (version `bilinear-4`).

## 0.2.0 (2026-11-24)

### Added
- **Exact cast-shadow maps.** `viewshed.shadow(dem, transform, azimuth, elevation, target_height=0.0, curvature=True, refraction=0.13)` in Python, `viewshed::shadow` in Rust. Returns every cell as lit or in shadow for a sun at the given azimuth (degrees clockwise from north) and elevation.
  - Exact on the bilinear surface, the same one as `surface="bilinear"` viewsheds. A cell is in shadow when its ray towards the sun passes strictly below the surface along a stretch of positive length. Rays that leave the raster are lit. No-data cells are `False` and never cast shadow. Earth curvature and refraction work as they do for viewsheds.
  - Exact for the sun direction as computed from the two angles in floating point. Cardinal azimuths are exact.
  - About 1 s for 4 million cells (2,001 × 2,001, 4 cores: 0.6–0.7 s with the sun 15–45° high, 1.1 s at 5°).
  - Approximate tools differ from the exact map on rough terrain: GRASS `r.sunmask` on 1.6–4.9% of cells (and takes minutes); a fast sampled sweep (Corripio 2003, about 50–100× faster) on 0.5–2.5%.
  - New conformance oracle (exact rational arithmetic plus a fast float pass), tested on tie-heavy and random rasters × 16 sun positions.

### Changed
- Exact bilinear viewsheds: in sector mode, the exact pass now decides only the patches the walk left undecided, instead of walking the whole sightline again in floats. Results are bit-identical, and cases that hit the exact fallback run up to 25% faster.
- CI actions updated to Node 24.

### Breaking (Rust only)
- `viewshed::Error` is `#[non_exhaustive]` and has new variants `InvalidSun`, `InvalidTargetHeight` and `UnsupportedRefraction`. A `match` on `Error` needs a `_ =>` arm. Python is unaffected: invalid arguments still raise `ValueError`.

## 0.1.0 (2026-11-24)

First release on PyPI and crates.io.

- `viewshed.viewshed`: single-observer viewsheds on the GRASS `r.viewshed` surface model (matches GRASS 8.4 on every cell, with and without earth curvature and refraction), or `surface="bilinear"` for exact line of sight on the bilinear surface.
- `viewshed.ztv`: wind-farm zones of theoretical visibility (counts of visible targets per cell), `method="forward"` (the definition) or `"reciprocal"` (one viewshed per target).
- Earth curvature and refraction on by default; `observer_height`, `target_height` and `max_distance` options.
- Rust core on all cores; abi3 wheels for Linux (x86_64, aarch64), macOS (Intel, Apple) and Windows, Python 3.9+; no GDAL, GRASS or compiler needed.
- Conformance suite: exact bilinear oracle, independent r.viewshed-model oracle, GRASS comparisons.

[0.3.0]: https://github.com/Yibbzz/viewshed/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Yibbzz/viewshed/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Yibbzz/viewshed/releases/tag/v0.1.0
