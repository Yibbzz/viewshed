# Project Brief: sightshed, a Fast, Exact Viewshed Library for Python

Nov 19, 2026 · @Nathan

## Overview and goal

Build a free, open-source Python library, written in Rust, that computes viewsheds with **true line of sight**. It should be as fast as the fast approximate tools and as accurate as the exact ones. It installs with `pip install sightshed` on any platform, with no GDAL, GRASS or compiler needed, and takes NumPy arrays in and gives NumPy arrays out.

**Problem.** Python users pick between fast and wrong, and right but awkward:

- **GDAL's viewshed** (`gdal_viewshed`, `gdal.ViewshedGenerate`) is fast and everywhere, but it uses an approximation (the Wang et al. reference plane) instead of line of sight, so errors build up with distance. Measured in PhotoFootprint's tests against strict line of sight, it **wrongly hid 1–5% of visible ground**, always erring towards hidden.
- **GRASS `r.viewshed`** is exact: a sweep-line algorithm by Haverkort, Toma and Zhuang. It needs a GRASS session, though, about **1.4 s of start-up per call**, and that rules it out for many observers or interactive use.
- **xarray-spatial `viewshed`** ports `r.viewshed` to Numba, but it pulls in xarray, Numba and Dask, and its out-of-core fallback is a different, approximate model. Its own docs give differing figures for how far that fallback disagrees, up to about 20% of cells on rough terrain in one version.

**Who it's for.**
1. **Landscape and visual impact (LVIA) practitioners** who need zones of theoretical visibility (ZTV) for wind turbines, pylons and buildings. Many observers or many targets, where speed and correctness both matter, and where the main tools today are commercial.
2. **Developers of GIS apps and plugins**, such as PhotoFootprint, who need a small, fast, exact viewshed call with no heavy dependencies.
3. **Researchers and analysts** doing visibility indexes, intervisibility and siting studies in notebooks and pipelines.

**What makes it different.**

- Exact line of sight against a clearly defined terrain model, tested against GRASS `r.viewshed` and a brute-force reference.
- One compiled package with no dependencies beyond NumPy, and prebuilt wheels for Windows, macOS and Linux that work on every Python ≥ 3.9 (abi3).
- **Many observers** as a first-class feature: cumulative counts, visibility index and ZTV, spread across all CPU cores.
- Practical limits apps need: azimuth and elevation limits (a camera's field of view), minimum and maximum distance, observer and target heights, earth curvature and refraction, and a choice of cell rule (centre visible vs any part visible).

**Working name:** `sightshed` (free on PyPI as of Nov 19, 2026; `viewshed` is also free but too generic).

## Prior art and reuse

Implement from the published papers. Don't copy GPL code (GRASS) into a permissively licensed library.

| Source | Use for | Notes |
| --- | --- | --- |
| Haverkort, Toma & Zhuang, "Computing visibility on terrains in external memory" (ALENEX 2007), basis of [GRASS r.viewshed](https://manpages.debian.org/testing/grass-doc/r.viewshed.1grass.en.html) | Core single-observer algorithm | Rotating sweep line over "events", with a balanced tree of active cells keyed by distance. O(n log n), versus O(n^1.5) for checking each cell's line of sight. GRASS is GPL, so implement from the paper. |
| [GRASS r.viewshed](https://manpages.debian.org/testing/grass-doc/r.viewshed.1grass.en.html) | **Test oracle** | Already used this way in PhotoFootprint (`tests/test_grass_crosscheck.py`), which agrees with it on over 99% of cells. Runs on GRASS 7.8 to 8.4 via `grass --tmp-location`. |
| [xarray-spatial viewshed](https://xarray-spatial.readthedocs.io/en/stable/reference/_autosummary/xrspatial.viewshed.viewshed.html) | Benchmark, API ideas | Numba port of the same algorithm. The main competitor to benchmark against. |
| [GDAL viewshed](https://gdal.org/en/stable/programs/gdal_raster_viewshed.html) | Benchmark, option names | Fast but approximate. GDAL 3.10 added cumulative mode and parallel jobs, and 3.12 added start/end angle and minimum distance options, so match its option names where sensible. |
| [WhiteboxTools VisibilityIndex](https://jblindsay.github.io/ghrg/Whitebox/Help/VisibilityIndex.html) | Benchmark for many observers | Its docs call it "very computationally intensive" and suggest skipping rows and columns to make it finish. |
| PhotoFootprint `terrain/visibility.py` | Reference behaviour for sectors, cell rules and target offset | NumPy radial sweep: exact enough, but slower (0.2 s for one observer over a 250 m radius at 1 m cells). |
| [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs) + [rust-numpy](https://github.com/PyO3/rust-numpy) + [Rayon](https://github.com/rayon-rs/rayon) | Bindings, wheels, parallelism | Standard stack for abi3 wheels with NumPy input and output that release the GIL. |

## Scope

| Phase | Includes |
| --- | --- |
| 0: Conformance suite and benchmark | No library code yet. Build the conformance suite (see below), then benchmark GDAL, `r.viewshed`, xarray-spatial, WhiteboxTools and the PhotoFootprint sweep for accuracy (against a brute-force line-of-sight reference) and speed, on synthetic surfaces and a real 1 m Environment Agency LiDAR DSM. Include a many-observer case. **Go/no-go:** go ahead only if the gap is real. |
| 1: MVP | Exact single-observer viewshed. Options: observer height, target height, min/max distance, azimuth limits, elevation limits, curvature and refraction, no-data. Outputs: visible mask, plus optionally the elevation angle and the height above the surface a target would need to be visible. abi3 wheels for all major platforms on PyPI. |
| 2: Many observers | Cumulative count over a list of observers. Visibility index (every cell, or a stride). ZTV: from how many observers a target of a given height is visible. Spread over all cores. Bounded memory. |
| 3: Integrations | Optional helpers for rasterio/GDAL files (`sightshed[io]`), a QGIS Processing provider (pure-Python plugin that uses the library when installed), and use as an optional speed-up in PhotoFootprint. |

**Out of scope (for now):** GPU, 3D meshes and voxel visibility, rasters too big for memory (Phase 2 tiles a window per observer instead), and Fresnel or radio propagation.

## Functional requirements

**1. Terrain model and the definition of "visible"**

- State the terrain model exactly and test against it: elevations at cell centres, linearly interpolated along the line of sight. Match `r.viewshed`'s model by default, so that tool can check our answers.
- A target cell is visible when the straight line from the observer's eye (surface + observer height) to the target point (surface + target height) passes nowhere below the terrain. No tolerance.
- Cell rule: `centre` (default, matches `r.viewshed`) or `any` (some part of the cell is visible, like PhotoFootprint's sweep).

**2. Single-observer API (Phase 1)**

```python
import sightshed

mask = sightshed.viewshed(
    dem,                      # 2-D float32/float64 NumPy array, NaN = no data
    transform,                # GDAL-style geotransform (x0, dx, 0, y0, 0, dy)
    observer=(x, y),
    observer_height=1.6, target_height=0.0,
    max_distance=500.0, min_distance=0.0,
    azimuth=None,             # e.g. (330, 30): clockwise from grid north, wraps
    elevation=None,           # e.g. (-30, 20) degrees
    curvature=True, refraction=0.13,
    cell_rule="centre",
    output="mask",            # or "angle" or "height_needed"
)
```

- Release the GIL during computation. Return results the same shape as the input, or a window when `crop=True`.
- Clear errors for an observer outside the raster or on no-data, non-square cells, or rotated rasters.

**3. Many observers (Phase 2)**

```python
counts = sightshed.cumulative(dem, transform, observers)        # observers: (n, 2) or (n, 3) array
index = sightshed.visibility_index(dem, transform, stride=1, max_distance=..., observer_height=...)
ztv = sightshed.ztv(dem, transform, targets, target_heights, observer_height=1.6)
```

- Parallel across observers, deterministic results, progress callback, cancellation.
- ZTV works the other way round: for each cell as an observer, count how many targets (such as turbine tips) are visible.

**4. Packaging**

- Rust library crate (`sightshed-core`, also on crates.io) plus a Python crate via PyO3.
- Wheels: manylinux x86_64 and aarch64; macOS x86_64 and arm64; Windows x64. abi3 for Python 3.9 and later, so one wheel per platform covers every Python that QGIS ships.
- No runtime dependencies beyond NumPy.

## Technical approach

- **Algorithm:** a radial sweep in the style of Haverkort, Toma and Zhuang. Events are each cell's enter, centre and exit angles, sorted by angle. A balanced tree of "active" cells is keyed by distance and augmented with maximum gradient, so each visibility query is O(log n). Implement the simpler O(n^1.5) per-cell line-of-sight version first, as the reference and fallback.
- **Azimuth limits** restrict the sweep's starting events. This is a big saving for camera fields of view.
- **Many observers:** one sweep per observer on its own window (max distance), in parallel with Rayon, summing into thread-local count arrays.
- **Numerics:** f64 for geometry, with tie-breaking set out explicitly, plus tests for flat terrain and exact grazing.

## Performance and accuracy targets

These are provisional and get calibrated in Phase 0.

- **Accuracy:** at least 99.9% cell agreement with `r.viewshed` (`centre` rule, same heights) on the synthetic set and the LiDAR tile, and 100% agreement with the brute-force reference on small rasters.
- **Single observer:** at least 5× faster than `r.viewshed` (including its start-up) and no slower than GDAL single-threaded, for a 1,000-cell radius.
- **Many observers:** 1,000 observers × 500-cell radius in under 60 s on an 8-core laptop.
- **Install:** `pip install sightshed` gives a working import on Windows, macOS and Linux with no compiler.

## Milestones and acceptance criteria

1. **M0: Conformance suite and benchmark.** Oracles for each surface model (with the models `r.viewshed` and GDAL use confirmed from their papers and source), analytic scenes, adapters for every existing tool, and the CI report running on every push. Then a report with accuracy and speed tables for all existing tools, and a go/no-go recommendation with numbers.
2. **M1: Reference implementation.** The brute-force line-of-sight version in Rust with Python bindings, all property tests passing, and wheels building in CI.
3. **M2: Fast sweep.** The sweep-line algorithm matches the reference 100% on small rasters and `r.viewshed` ≥ 99.9% on the test set, and meets the single-observer speed target.
4. **M3: Options.** Azimuth and elevation limits, min distance, curvature and refraction, the `any` cell rule, and the `angle` and `height_needed` outputs, each with tests.
5. **M4: Many observers.** Cumulative, visibility index and ZTV; meets the many-observer target.
6. **M5: Release.** Docs site, examples (camera field of view, wind turbine ZTV), version 0.1 on PyPI and crates.io.
7. **M6: Integrations.** `sightshed[io]` helpers, a QGIS Processing provider plugin, and the optional speed-up in PhotoFootprint.

## Conformance suite: one CI that compares every viewshed tool

Its own Python package in the repo (`conformance/`), built **first**: Phase 0's benchmark runs on it, and every later milestone is judged by it.

**Two kinds of baseline.** "Accurate" only means something relative to a stated surface model: a DEM gives heights at cell centres, and each tool decides differently what lies between them. So:

1. **Model-exact oracles** (is the algorithm correct for the surface it claims?). For each surface model (bilinear, triangulated, flat-topped cells, and whatever `r.viewshed` and GDAL actually assume, confirmed from their papers and source in M0), compute line of sight exactly:
   - walk the sightline cell by cell (Amanatides–Woo traversal, so no cell is skipped);
   - in each cell, solve the sightline-minus-surface gap in closed form (linear or quadratic);
   - use exact rational arithmetic for near-ties.

   The output per cell is the **clearance**, the smallest height of the sightline above the surface, not just visible or hidden.
2. **Analytic scenes** (how close is it to the real geometry?). Scenes defined as continuous shapes (box buildings, a cone hill, a ridge, a thin wall, a slope), with true visibility from closed-form geometry. Rasterise at 4, 2, 1 and 0.5 m and measure each tool's error as cells get finer. Good tools converge; an algorithm-level flaw shows up as error that doesn't shrink.

**Tools compared on every run:** GDAL (all modes), GRASS `r.viewshed`, xarray-spatial, WhiteboxTools, PhotoFootprint's NumPy sweep, and `sightshed` (each cell rule). Each tool gets a small adapter: the same inputs and heights, its output resampled onto the oracle's grid.

**Metrics:**
- wrongly hidden and wrongly visible, split into clearance bands (cells under 1 cm reported as ambiguous and set aside; 1 cm–10 cm; 10 cm–1 m; over 1 m);
- shadow-edge cells (within one cell of a boundary) vs interior cells;
- convergence with resolution on analytic scenes;
- wall-clock time and peak memory.

**Datasets:**
- small synthetic rasters, oracle on every cell;
- analytic scenes at four resolutions;
- real 1 m EA LiDAR DSM tiles (urban, wooded, hilly), oracle on a fixed random sample plus every cell near a shadow edge.

Oracle results are cached by a hash of their inputs, so CI only recomputes when inputs change.

**CI:** one pinned Docker image (GDAL, GRASS, WhiteboxTools, Rust toolchain). Every push produces a report (Markdown tables plus charts) as a build artefact and job summary. Only `sightshed`'s own targets fail the build: 100% agreement with its declared model's oracle on small rasters, and no accuracy or speed regression beyond set limits. Other tools are tracked, not gated.

**Limits, stated in the report:** everything is relative to the DEM. Real-world ground truth (surveyed visibility from known points, such as photos) is a possible later dataset, not part of the baseline.

## Testing

- **Conformance suite** (above): the main accuracy gate.
- **Brute-force reference:** a slow, exact per-cell line of sight in pure Python/NumPy, used as ground truth on small rasters.
- **GRASS cross-check:** reuse PhotoFootprint's `r.viewshed` harness; skipped where GRASS is missing, run in CI using the QGIS/GRASS Docker images.
- **Property tests (Hypothesis):** everything visible on a flat plane; analytic shadow behind a single wall; raising the observer never hides a cell; adding an obstacle never reveals one; rotating or flipping the raster rotates the result.
- **Benchmarks:** Criterion (Rust) and pytest-benchmark (Python), tracked in CI so speed regressions are visible.

## Licence

**MIT OR Apache-2.0**, the Rust convention, so LVIA consultancies and commercial tools can use it, which is how a library like this gets adopted. Implement from the papers; never copy GPL code from GRASS. GPL plugins such as PhotoFootprint can still use it.

## Using it from QGIS (PhotoFootprint and other plugins)

- The QGIS plugin repository asks plugins **not to bundle binaries**, so the compiled wheel can't ship inside a plugin zip.
- Instead, `sightshed` is an **optional dependency**: the user (or the qpip plugin, from a requirements file) pip-installs it into QGIS's own Python. The plugin checks for it at import and **falls back to its NumPy sweep** if it's missing. Results must match, so share the same test cases.
- abi3 wheels for Python 3.9+ cover the Python versions QGIS ships on Windows, macOS and Linux.

## Open questions for the owner

- [ ] Name: `sightshed`, or something else?
- [ ] Licence: MIT OR Apache-2.0 (recommended), or GPL to match the GIS ecosystem?
- [ ] Which comes first after the MVP, the LVIA many-observer features or app integration (PhotoFootprint)?
- [ ] A real test area: an EA 1 m LiDAR DSM tile, ideally with buildings, trees and hills.
- [ ] GPU support later, or never?
