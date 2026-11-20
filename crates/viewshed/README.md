# viewshed

**Fast, exact line-of-sight viewsheds** for Python and Rust.

Most fast viewshed tools approximate line of sight; GDAL's, for example, wrongly hid 1–5% of visible ground in our tests. The exact ones are slow to call or hard to install. `viewshed` aims to be both exact and fast: a Rust core, NumPy in and out, `pip install` on any platform with no GDAL, GRASS or compiler, and first-class support for many observers (cumulative viewsheds, visibility index, wind-turbine ZTV).

## Status

**Rust:** the default is a fast radial sweep over GRASS `r.viewshed`'s surface model, using all cores. It matches GRASS cell for cell. A 1,000-cell radius (4 million cells) takes about 1.9 s on a 4-core laptop, against 8–9 s for `r.viewshed`. The slow bilinear reference from 0.0.1 is still available as `Surface::Bilinear`.

**Python:** still the pure-NumPy bilinear reference from 0.0.1. The Rust core will be exposed to Python through PyO3 bindings next, with the same API.

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

The default is the model GRASS `r.viewshed` uses, so its results can be checked against GRASS. The observer stands at the centre of its cell. Every cell closer than the target is reduced to three height samples: its centre, and the two corners that bound it as seen from the observer, with each corner averaged from the four cells meeting there. A target (its cell centre plus `target_height`) is hidden when a closer cell's gradient, interpolated by angle at the target's direction, is steeper. No-data (NaN) never blocks and is never visible.

## Licence

MIT OR Apache-2.0, at your option.
