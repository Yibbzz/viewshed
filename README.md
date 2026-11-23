# viewshed

**Fast, exact line-of-sight viewsheds** for Python and Rust.

Most fast viewshed tools approximate line of sight; GDAL's, for example, wrongly hid 1–5% of visible ground in our tests. The exact ones are slow to call or hard to install. `viewshed` aims to be both exact and fast: a Rust core, NumPy in and out, `pip install` on any platform with no GDAL, GRASS or compiler, and first-class support for many observers (cumulative viewsheds, visibility index, wind-turbine ZTV).

## Status

Pre-release (0.0.x, not yet on PyPI or crates.io in this form). See [BRIEF.md](BRIEF.md) for the plan and [`conformance/`](conformance/) for how every result is checked.

- **Exact, and checked.** The default surface model is GRASS `r.viewshed`'s, and results match GRASS 8.4 on every cell, with and without earth curvature and refraction. Two independent checkers back this up: an exact bilinear line-of-sight oracle (closed form, float decisions certified by rigorous rounding-error bounds, ties in rational arithmetic), and an r.viewshed-model oracle written from the method's description. The `"bilinear"` surface is exact line of sight on the bilinear surface and matches that oracle on every cell (see [Bilinear surface](#bilinear-surface)). Accuracy against real geometry is measured on analytic scenes and continuous surfaces.
- **Fast.** A radial sweep in Rust on all cores: a 1,000-cell radius (4 million cells) takes about 1 s and 100 MB on a 4-core laptop, against 8–9 s for `r.viewshed`.
- **Wind-farm ZTV.** Counts how many turbine tips every cell can see, either by the definition (every cell an observer) or by the usual reciprocal shortcut (one viewshed per tip, about 5× faster and within a few hundredths of a percent of the definition). On an 8 km square at 10 m with 8 turbines, the definition takes about 7 s on a 4-core laptop.
- **Earth curvature and refraction**, on by default, as in GRASS `r.viewshed -c -r`.

| Package | Install | Code |
| --- | --- | --- |
| Python | `pip install viewshed` (abi3 wheels for Python ≥ 3.9) | [`python/`](python/) |
| Rust | `cargo add viewshed` | [`crates/viewshed/`](crates/viewshed/) |

## Python

```python
import numpy as np
import viewshed

dem = np.zeros((41, 41)); dem[15, :] = 5.0          # flat ground with a 5 m wall
gt = (0.0, 1.0, 0.0, 41.0, 0.0, -1.0)               # GDAL-style geotransform
seen = viewshed.viewshed(dem, gt, observer=(20.5, 20.5), observer_height=1.6)   # bool array

# zone of theoretical visibility: how many blade tips each cell can see from 2 m
tips = [(10.5, 30.5, 150.0), (30.5, 30.5, 150.0)]
counts = viewshed.ztv(dem, gt, tips, observer_height=2.0)                      # uint32 array
```

Options: `observer_height`, `target_height`, `max_distance`, `curvature` (default on), `refraction` (0.13), `earth_radius` (mean earth radius), `surface` (`"rviewshed"`, or `"bilinear"`: exact line of sight on the bilinear surface, observer at its exact point). The computation runs in Rust with the GIL released. `viewshed.reference` is the original pure-NumPy implementation, kept as an independent check.

## Surface model

The default is the model GRASS `r.viewshed` uses, so its results can be checked against GRASS. The observer stands at the centre of its cell. Every cell closer than the target is reduced to three height samples: its centre, and the two corners that bound it as seen from the observer, with each corner averaged from the four cells meeting there. A target (its cell centre plus `target_height`) is hidden when a closer cell's gradient, interpolated by angle at the target's direction, is steeper. With curvature on, every point is lowered by d²/(2R)·(1 − refraction) at horizontal distance d. No-data (NaN) never blocks and is never visible.

## Bilinear surface

`surface="bilinear"` treats the DEM as heights at cell centres joined by bilinear interpolation, puts the observer at its exact point (not snapped to a cell centre) and decides line of sight to every cell exactly on that surface. Each decision is made in floating point only when rigorous rounding-error bounds prove it; the rare undecided sightlines are settled in rational arithmetic. So the output is identical, cell for cell, to the exact oracle in [`conformance/`](conformance/).

Large rasters (32,768 cells or more) are computed outward from the observer in rings, split into 64 angular sectors that run in parallel. Each sector keeps an analytic horizon of the terrain inserted so far. A cell certainly below that horizon is hidden without a sightline walk, whole 4×4 blocks of such cells are skipped together, and a cell certainly above it only has its last few cells walked. Small rasters walk each sightline with a max-height pyramid.

Speed on 2,001 × 2,001 windows of real terrain (wall time, best of three, 4 cores of an i5-8500T; observer 1.6 m above the ground, curvature and refraction on):

| Terrain | Observer | Visible | Time |
| --- | --- | --- | --- |
| Alps (Copernicus 30 m) | window centre | 0.6% | 0.22 s |
| Alps | highest point | 11% | 0.59 s |
| Iowa (USGS 3DEP 10 m) | window centre | 1% | 0.22 s |
| Iowa | highest point | 6% | 0.40 s |
| Kansas (USGS 3DEP 10 m) | window centre | 14% | 1.16 s |
| Kansas | highest point | 37% | 1.08 s |
| Wellington (1 m LiDAR) | window centre | 17% | 0.69 s |
| Wellington | highest point | 48% | 1.16 s |

In the same runs, the exact bilinear result was as fast as or faster than this library's approximate r.viewshed-model sweep on the Alps and Iowa windows, and 1.3–2.7× slower on Kansas and Wellington, where more of the terrain is visible.

## Releasing

The version is written in four places (`crates/viewshed/Cargo.toml`, `crates/viewshed-py/Cargo.toml`, `python/pyproject.toml`, `python/viewshed/__init__.py`); `python scripts/check_versions.py` checks they agree. Pushing a tag `vX.Y.Z` that matches runs the `wheels` workflow: it builds abi3 wheels for Linux (x86_64, aarch64), macOS (Intel, Apple) and Windows, installs and tests each one on Python 3.9 and 3.13, then publishes to PyPI (trusted publishing, `pypi` environment) and crates.io (`CARGO_REGISTRY_TOKEN` in the `crates-io` environment). Nothing is published if the versions and tag disagree or any test fails.

## Licence

MIT OR Apache-2.0, at your option.
