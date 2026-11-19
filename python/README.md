# viewshed

**Fast, exact line-of-sight viewsheds** for Python and Rust.

Most fast viewshed tools approximate line of sight; GDAL's, for example, wrongly hid 1–5% of visible ground in our tests. The exact ones are slow to call or hard to install. `viewshed` aims to be both exact and fast: a Rust core, NumPy in and out, `pip install` on any platform with no GDAL, GRASS or compiler, and first-class support for many observers (cumulative viewsheds, visibility index, wind-turbine ZTV).

## Status

**0.0.x is the reference implementation:** a simple, correct line-of-sight check to every cell. It's slow and meant for small rasters and testing. The fast sweep-line algorithm will keep the same API and results. See the project brief for the plan.

| Package | Install | Code |
| --- | --- | --- |
| Python | `pip install viewshed` | `python/` (NumPy) |
| Rust | `cargo add viewshed` | `crates/viewshed/` |

## Python

```python
import numpy as np
import viewshed

dem = np.zeros((41, 41)); dem[15, :] = 5.0          # flat ground with a 5 m wall
gt = (0.0, 1.0, 0.0, 41.0, 0.0, -1.0)               # GDAL-style geotransform
seen = viewshed.viewshed(dem, gt, observer=(20.5, 20.5), observer_height=1.6)
```

## Surface model

Heights at cell centres, bilinearly interpolated between them. A cell is visible when the straight line from the observer's eye to the point `target_height` above the cell centre stays on or above the surface. No-data (NaN) never blocks and is never visible.

## Licence

MIT OR Apache-2.0, at your option.
