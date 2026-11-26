# viewshed

**Fast, exact line-of-sight viewsheds** for Python and Rust.

Most fast viewshed tools approximate line of sight; GDAL's, for example, wrongly hid 1–5% of visible ground in our tests. The exact ones are slow to call or hard to install. `viewshed` aims to be both exact and fast: a Rust core, NumPy in and out, `pip install` on any platform with no GDAL, GRASS or compiler, and first-class support for many observers (cumulative viewsheds, visibility index, wind-turbine ZTV).

## Status

0.4.0 (adds horizon maps, exact point-to-point line of sight, cumulative viewsheds and a visibility index). See the [repository](https://github.com/Yibbzz/viewshed) for the plan and the conformance suite that checks every result.

- **Exact, and checked.** The default surface model is GRASS `r.viewshed`'s, and results match GRASS 8.4 on every cell, with and without earth curvature and refraction. Two independent checkers back this up: an exact bilinear line-of-sight oracle (closed form, float decisions certified by rigorous rounding-error bounds, ties in rational arithmetic), and an r.viewshed-model oracle written from the method's description. The `"bilinear"` surface is exact line of sight on the bilinear surface and matches that oracle on every cell (see [Bilinear surface](#bilinear-surface)). Accuracy against real geometry is measured on analytic scenes and continuous surfaces.
- **Fast.** A radial sweep in Rust on all cores: a 1,000-cell radius (4 million cells) takes about 1 s and 100 MB on a 4-core laptop, against 8–9 s for `r.viewshed`.
- **Wind-farm ZTV.** Counts how many turbine tips every cell can see, either by the definition (every cell an observer) or by the usual reciprocal shortcut (one viewshed per tip, about 5× faster and within a few hundredths of a percent of the definition). On an 8 km square at 10 m with 8 turbines, the definition takes about 7 s on a 4-core laptop.
- **Many observers**: cumulative viewsheds and a visibility index, exact (see [Many observers](#many-observers)).
- **Point-to-point line of sight**, exact, for many pairs at once (radio links; see [Line of sight between points](#line-of-sight-between-points)).
- **Reverse viewshed**: from which cells a target (a mast, a roof) can be seen, exact for every cell (see [Reverse viewshed](#reverse-viewshed)).
- **Exact shadow maps**: every cell lit or in shadow for a given sun position, on the bilinear surface (see [Shadow maps](#shadow-maps)).
- **Horizon maps**: for every cell, the exact elevation angle of the skyline in a given direction (see [Horizon maps](#horizon-maps)).
- **Sun hours**: hours of direct sun per cell over a day, a season or a year, from exact shadow maps and the sun's computed position (see [Sun hours](#sun-hours)).
- **Earth curvature and refraction**, on by default, as in GRASS `r.viewshed -c -r`.

| Package | Install | Code |
| --- | --- | --- |
| Python | `pip install viewshed` (abi3 wheels for Python ≥ 3.9) | `python/` |
| Rust | `cargo add viewshed` | `crates/viewshed/` |

## Rust

```rust
use viewshed::{reverse_viewshed, shadow, viewshed, ztv, Grid, Options, ShadowOptions, Sun, ZtvOptions, ZtvTarget};

let z = vec![0.0; 41 * 41];
let grid = Grid::new(&z, 41, 41, [0.0, 1.0, 0.0, 41.0, 0.0, -1.0]).unwrap();
let seen = viewshed(&grid, 20.5, 20.5, &Options::default()).unwrap();          // Vec<bool>
let tips = [ZtvTarget { x: 10.5, y: 30.5, height: 150.0 }];
let counts = ztv(&grid, &tips, &ZtvOptions::default()).unwrap();                // Vec<u32>
let mast = Options { target_height: 30.0, ..Options::default() };
let seen_from = reverse_viewshed(&grid, 20.5, 35.5, &mast).unwrap();          // Vec<bool>
let sun = Sun { azimuth_deg: 135.0, elevation_deg: 30.0 };
let lit = shadow(&grid, sun, &ShadowOptions::default()).unwrap();               // Vec<bool>
```

## Surface model

The default is the model GRASS `r.viewshed` uses, so its results can be checked against GRASS. The observer stands at the centre of its cell. Every cell closer than the target is reduced to three height samples: its centre, and the two corners that bound it as seen from the observer, with each corner averaged from the four cells meeting there. A target (its cell centre plus `target_height`) is hidden when a closer cell's gradient, interpolated by angle at the target's direction, is steeper. With curvature on, every point is lowered by d²/(2R)·(1 − refraction) at horizontal distance d. No-data (NaN) never blocks and is never visible.

## Bilinear surface

`Surface::Bilinear` (Python: `surface="bilinear"`) treats the DEM as heights at cell centres joined by bilinear interpolation, puts the observer at its exact point (not snapped to a cell centre) and decides line of sight to every cell exactly on that surface. Each decision is made in floating point only when rigorous rounding-error bounds prove it; the rare undecided sightlines are settled in rational arithmetic. So the output is identical, cell for cell, to the exact oracle in [`conformance/`](https://github.com/Yibbzz/viewshed/tree/main/conformance). No-data (NaN) never blocks and is never visible; a patch (the square between four cell centres) with a no-data corner has no surface. A sightline is blocked only where it passes strictly below the surface along a stretch of positive length, so touching a single grid vertex between two patches without surface (possible only on exact lattice directions at the edge of no-data areas) doesn't block it.

Large rasters (32,768 cells or more) are computed outward from the observer in rings, split into 64 angular sectors that run in parallel. Each sector keeps an analytic horizon of the terrain inserted so far. A cell certainly below that horizon is hidden without a sightline walk, whole 4×4 blocks of such cells are skipped together, and a cell certainly above it only has its last few cells walked. Small rasters walk each sightline with a max-height pyramid.

Speed on 2,001 × 2,001 windows of real terrain (wall time, best of three, 4 cores of an i5-8500T; observer 1.6 m above the ground, curvature and refraction on):

| Terrain | Observer | Visible | Exact bilinear | r.viewshed-model sweep (approximate) |
| --- | --- | --- | --- | --- |
| Alps (Copernicus 30 m) | window centre | 0.6% | 0.21 s | 0.43 s |
| Alps | highest point | 11% | 0.42 s | 0.55 s |
| Iowa (USGS 3DEP 10 m) | window centre | 1% | 0.22 s | 0.43 s |
| Iowa | highest point | 6% | 0.34 s | 0.50 s |
| Kansas (USGS 3DEP 10 m) | window centre | 14% | 0.96 s | 0.53 s |
| Kansas | highest point | 37% | 0.89 s | 0.62 s |
| Wellington (1 m LiDAR) | window centre | 17% | 0.55 s | 0.51 s |
| Wellington | highest point | 48% | 0.85 s | 0.65 s |

The exact bilinear result is faster than this library's approximate r.viewshed-model sweep (same runs) on the Alps and Iowa windows, about level on the Wellington centre, and 1.3–1.8× slower where much of the terrain is visible (Kansas, the Wellington high point).

## Many observers

`cumulative` counts, for every cell, how many of a list of observers can see it (points along a walking route, a set of viewpoints, masts of different heights); `visibility_index` puts an observer on every n-th cell and gives each cell the fraction of them that can see it, a map of how exposed each place is.

```rust
use viewshed::{cumulative, visibility_index, Options, Point};
let route = [Point { x: 10.5, y: 30.5, height: 1.6 }, Point { x: 20.5, y: 20.5, height: 10.0 }];
let counts = cumulative(&grid, &route, &Options::default()).unwrap();                        // Vec<u32>
let opts = Options { max_distance: Some(3000.0), ..Options::default() };
let (vi, n_observers) = visibility_index(&grid, 20, &opts).unwrap();                         // Vec<f64>
```

Both are exactly the sum of one viewshed per observer, on either surface model, spread over all cores, so they are exact in the same sense as `viewshed`, and the cost is one viewshed per observer. On 1,001 × 1,001 cells at 10 m with a 3 km range (4 cores, on a busy machine), 100 observers take about 10 s with the default surface and 23 s with `"bilinear"`; a visibility index with an observer on every 20th cell (2,500 observers) about 4 minutes with the default surface. Cells beyond `max_distance` of an observer count as not seen by it, so the index is lower near the raster's edge.

## Line of sight between points

`intervisible` decides exactly whether pairs of points can see each other: radio masts, sensors, any two points anywhere in the raster, each at a height above the ground.

```rust
use viewshed::{intervisible, LosOptions, Point};
let mast = Point { x: 20.5, y: 20.5, height: 30.0 };
let site = Point { x: 5.5, y: 30.5, height: 2.0 };
let ok = intervisible(&grid, &[(mast, site)], &LosOptions::default()).unwrap();   // Vec<bool>
```

The model is the bilinear surface (as `surface="bilinear"`): two points see each other when the segment between them is nowhere strictly below the surface along a stretch of positive length, with earth curvature and refraction as for viewsheds. The answer doesn't depend on which point is first, and for targets on cell centres it equals the bilinear viewshed. Most of each sightline is cleared by a bound (a bilinear patch never rises above its highest corner); stretches close to the ground are decided per patch in floating point with rigorous error margins, and the rare ones within the margin in exact rational arithmetic. On a 2,001 × 2,001 raster at 10 m, 20,000 links of up to 6 km take about 1 s on a 4-core laptop (about 50 µs per link).

## Fresnel clearance

`fresnel_clearance` gives the worst clearance of a radio link through the first Fresnel zone.

```rust
use viewshed::{fresnel_clearance, LosOptions, Point};
let a = Point { x: 100.0, y: 200.0, height: 10.0 };
let b = Point { x: 300.0, y: 200.0, height: 10.0 };
let c = fresnel_clearance(&grid, &[(a, b)], 2.4e9, &LosOptions::default()).unwrap()[0];
println!("ratio={} t={}", c.ratio, c.t);
```

The link ends are `(x, y, height above the surface)` in metres. The result is the minimum, along the link, of the vertical clearance between the sightline and the terrain divided by the first-zone radius `sqrt(lambda d1 d2 / D)`. A ratio of 1 means the whole first zone is clear, 0.6 is the usual planning target, 0 is grazing, and a negative ratio means the terrain blocks the line itself. `t` is the position of the worst point from `a` to `b`.

The calculation is exact on the bilinear surface, including curvature and refraction, and is certified to within `1e-9 + 1e-12 |ratio|` of the exact rational-arithmetic value. Links run in parallel; on the mixed 2,001 × 2,001 benchmark terrain they take about 16–19 microseconds per pair on 4 cores.

## Reverse viewshed

`reverse_viewshed` turns the question round: from which cells can a target at (x, y), `opts.target_height` above the ground, be seen by an observer `opts.observer_height` above each cell?

```rust
use viewshed::{reverse_viewshed, Options, Surface};
let opts = Options { target_height: 30.0, observer_height: 1.6, surface: Surface::Bilinear, ..Options::default() };
let seen_from = reverse_viewshed(&grid, x, y, &opts).unwrap();                  // Vec<bool>
```

A cell is `true` when a viewshed from that cell would show the target, for every cell. On the bilinear surface that costs one exact viewshed: the sightline is the same segment both ways, the blocking rule (strictly below the surface along a stretch of positive length) doesn't depend on direction, and the curvature drop below the chord at a point is the same seen from either end, so it is the viewshed from the target with the two heights swapped (checked against the exact oracle run forwards from every cell). The target stays at its exact point; observers stand at cell centres. The `Surface::RViewshed` model isn't symmetric (only cells nearer the observer block), so there every cell looks at the target's cell centre in turn, as `ztv` does with `ZtvMethod::Forward`: exact for the model but slower (2,001 × 2,001, 4 cores: 6.7 s, against 2.3 s for the bilinear reverse viewshed and 1.4 s for an ordinary r.viewshed-model viewshed). Taking the viewshed from the target instead differed on 0.2–1.1% of cells in our tests (41 × 41 rough rasters).

## Shadow maps

`shadow` marks every cell lit or in shadow for a sun at a given azimuth (degrees clockwise from north) and elevation (degrees above the horizon), exactly, on the same bilinear surface as the exact viewshed.

```rust
use viewshed::{shadow, ShadowOptions, Sun};
let lit = shadow(&grid, Sun { azimuth_deg: 135.0, elevation_deg: 30.0 }, &ShadowOptions::default()).unwrap(); // Vec<bool>
```

A cell is in shadow when its ray towards the sun passes strictly below the surface along a stretch of positive length; rays that leave the raster are lit, and no-data cells are `False` and never cast shadow. The result is exact for the sun's direction vector as computed from the two angles in floating point (cardinal azimuths are exact): at 45°, for example, the ray rises by `tan(45°)` = 0.9999999999999999 per unit, not 1, which can move a shadow's edge by a cell on terrain built to tie. Most cells are certified lit by a running maximum along strips in the sun's direction; the rest are decided on their own ray with rigorous rounding-error bounds, and true ties in exact rational arithmetic. Earth curvature and refraction work as for viewsheds (refraction above 1 is rejected).

An exact shadow map takes about 0.5 s for 4 million cells (2,001 × 2,001 windows as above, 4 cores of an i5-8500T: 0.3–0.5 s with the sun 5–45° high). Approximate methods are faster or slower depending on the method, and differ from the exact map on a few percent of cells: a fast sampled sweep (Corripio 2003) is about 30–50 times faster and, on rough terrain at 2,001 × 2,001, differs on 0.5–1% of cells with the sun 15–30° high (missed shadow) and 1.5–2.5% at 5°; on 501 × 501 windows of rough terrain with the sun 5–30° high, GRASS `r.sunmask` differs on 1.6–4.9% of cells, mostly extra shadow within two cells of the shadow edge, and takes minutes.

## Horizon maps

`horizon_map` gives, for every cell, the elevation angle (degrees) of the highest terrain ahead in one direction: the input for sky view factor, terrain openness or a quick look at where the sun clears the skyline.

```rust
use viewshed::{horizon_map, HorizonOptions};
let h = horizon_map(&grid, 135.0, &HorizonOptions::default()).unwrap();          // Vec<f64>, degrees
```

Same surface as `shadow`: heights at cell centres, bilinear between them, and the surface ends half a cell beyond the outer cell centres. Each value is the highest elevation angle of any point ahead on a stretch of positive length (no-data never counts), with earth curvature and refraction as for viewsheds. Cells with only no-data ahead get −90, no-data cells NaN. Negative horizons stay negative: GRASS `r.horizon` clamps at 0 (`np.maximum(h, 0)` to compare). A cell is shadowed by a sun at that azimuth exactly when the sun is below its horizon, so horizon maps and `shadow` agree.

Exact to within 10⁻⁹ in the tangent of the angle (certified for every cell), up to the final arctangent. Strips along the direction hold a tree of upper convex hulls of patch corners (with each strip's sideways tilt taken out), so most of the terrain ahead is ruled out without being looked at; the rest is evaluated in closed form with rigorous error bounds, and the rare cells that can't be bounded tightly enough are settled with their stretches decided in exact rational arithmetic. Speed depends on the terrain: one direction over 1,001 × 1,001 cells takes about 1.5 s on mountains (Alps, 30 m) and rolling farmland (Iowa, 10 m), 3–5 s on flat plains (Kansas, 10 m) and under 1 s on a smooth tilted plane (4 cores). The slowest case we know of is artificial: a tilted plane with centimetre noise, about 13 s, where many distant patches come within a hair of the horizon and can't be ruled out early. GRASS `r.horizon` agrees in grid-aligned directions, but in directions off the grid axes on rough terrain it is more than 0.5° off on 9–30% of cells (it samples along the ray), with 0.5–3.7% of cells flipping between lit and shadowed. It is also slower in wall time (3.5–8.8 s single-threaded against 0.8–2.0 s on 4 cores, 1,001 × 1,001, one direction), and about as fast per core. To compare, put the raster on its zone's central meridian or rotate the azimuth: `r.horizon` turns its directions to true north by the projection's grid convergence, while `horizon_map` uses grid north.

## Sun hours

`sun_hours_between` adds up exact shadow maps over a time range: hours of direct sun per cell (`f64`, NaN on no-data). `sun_position` gives the sun's azimuth and elevation at a place and time; `sun_samples` and `sun_hours` take the sampling, or any weighted set of sun positions, into your own hands.

```rust
use viewshed::{sun_hours_between, sun_position, ShadowOptions};
// 2026-06-21 00:00 to 2026-06-22 00:00 UTC (seconds since 1970), 15-minute steps, at 51.5 N 0.13 W
let hours = sun_hours_between(&grid, 51.5, -0.13, 1_782_000_000.0, 1_782_086_400.0, 900.0, 0.0, &ShadowOptions::default()).unwrap();
let sun = sun_position(51.5, -0.13, 1_782_043_200.0).unwrap();                  // azimuth_deg, elevation_deg, ...
```

Times are seconds since 1970-01-01 UTC. Each time step (default 15 minutes) uses the sun's position at the middle of the step and counts the whole step if the cell is lit; steps with the sun below the horizon count for nothing, and a last shorter step is sampled at its own middle. Each shadow map is exact for the sun position the package computes; the total over time is as fine as the step. The sun's position comes from the NOAA solar calculator equations (after Meeus), within about 0.02° of NREL's SPA for 1950–2050, and is computed once per step for the whole raster (across 50 km the sun's azimuth varies by about 0.3°). Two different refractions are involved: the sun's apparent elevation includes astronomical refraction (sunlight bending near the horizon), while `refraction` is the terrestrial coefficient used for the sightline over the curved earth, as for viewsheds.

Azimuths in the raster are measured from grid north, which differs from true north by the meridian convergence (a few degrees in UTM away from the central meridian). Pass it as `grid_convergence` (grid azimuth = true azimuth − convergence). (The Python package can compute it from the raster's CRS with pyproj.)

A day at 15-minute steps on 2,001 × 2,001 cells takes 15–23 s on 4 cores (about 0.3–0.4 s per daylight step); a year at hourly steps, about half an hour.

How much exactness matters here: over a midsummer day, a fast sampled sweep (Corripio 2003) gave too few sun hours on average (0.01–0.09 h per cell), with 7.6% of cells on a 1 m lidar DEM and 2.6% on 30 m Alpine terrain off by an hour or more. That is 3–6 times the effect of realistic, spatially correlated DEM error on the same terrain; on flat 10 m farmland, DEM error matters more.

## Licence

MIT OR Apache-2.0, at your option.
