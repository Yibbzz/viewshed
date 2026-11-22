# viewshed

**Fast, exact line-of-sight viewsheds** for Python and Rust.

Most fast viewshed tools approximate line of sight; GDAL's, for example, wrongly hid 1–5% of visible ground in our tests. The exact ones are slow to call or hard to install. `viewshed` aims to be both exact and fast: a Rust core, NumPy in and out, `pip install` on any platform with no GDAL, GRASS or compiler, and first-class support for many observers (cumulative viewsheds, visibility index, wind-turbine ZTV).

## Status

Pre-release (0.0.x, not yet on PyPI or crates.io in this form). See the project brief for the plan and the conformance suite for how every result is checked.

- **Exact, and checked.** The default surface model is GRASS `r.viewshed`'s, and results match GRASS 8.4 on every cell, with and without earth curvature and refraction. Two independent checkers back this up: an exact bilinear line-of-sight oracle (closed form, float decisions certified by rigorous rounding-error bounds, ties in rational arithmetic), and an r.viewshed-model oracle written from the method's description. The `"bilinear"` surface is exact line of sight on the bilinear surface and matches that oracle on every cell, at about the speed of the r.viewshed sweep. Accuracy against real geometry is measured on analytic scenes and continuous surfaces.
- **Fast.** A radial sweep in Rust on all cores: a 1,000-cell radius (4 million cells) takes about 1 s and 100 MB on a 4-core laptop, against 8–9 s for `r.viewshed`.
- **Wind-farm ZTV.** Counts how many turbine tips every cell can see, either by the definition (every cell an observer) or by the usual reciprocal shortcut (one viewshed per tip, about 5× faster and within a few hundredths of a percent of the definition). On an 8 km square at 10 m with 8 turbines, the definition takes about 7 s on a 4-core laptop.
- **Earth curvature and refraction**, on by default, as in GRASS `r.viewshed -c -r`.

| Package | Install | Code |
| --- | --- | --- |
| Python | `pip install viewshed` (abi3 wheels for Python ≥ 3.9) | `python/` |
| Rust | `cargo add viewshed` | `crates/viewshed/` |

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

Heights at cell centres, bilinearly interpolated between them. A cell is visible when the straight line from the observer's eye to the point `target_height` above the cell centre stays on or above the surface. With curvature on, every point is lowered by d²/(2R)·(1 − refraction) at horizontal distance d. No-data (NaN) never blocks and is never visible.

## Licence

MIT OR Apache-2.0, at your option.
