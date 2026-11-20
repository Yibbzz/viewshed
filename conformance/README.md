# viewshed conformance suite

Compares viewshed tools against exact baselines (see "Conformance suite" in
[`../BRIEF.md`](../BRIEF.md)). Not published; it's the yardstick for the library.

Run it with the Python environment that has the package built in it, so the
bindings are checked too:

```bash
cd python && uv venv --system-site-packages .venv && uv pip install --python .venv/bin/python maturin pytest "numpy<2"
.venv/bin/maturin develop --release && cd ../conformance
PY=../python/.venv/bin/python
$PY -m pytest tests                    # the suite's own tests
$PY -m conformance.run --quick         # ~1 min
$PY -m conformance.run --out report.md
$PY -m conformance.bench --grass       # single-observer speed, 1,000-cell radius
$PY -m conformance.ztv_bench           # wind-farm ZTV, every cell an observer
$PY -m conformance.ztv_bench --cells 401 --relief 3 --truth      # vs true visibility
$PY -m conformance.ztv_bench --cells 401 --res 50 --relief 3 --curvature --truth
```

## Wind-farm ZTV (`ztv_bench.py`)

A synthetic landscape with turbines on a ridge, every cell an observer. It
times both ZTV methods in the crate, *forward* (every cell looks at every blade
tip: the definition) and *reciprocal* (one viewshed from each tip, the usual
shortcut), and reports where they disagree. With `--truth` the landscape is a
smooth mathematical surface, and both are compared with true visibility on it
(each sightline sampled every 5 m in NumPy, so allow a few minutes).

## Baselines

- **`oracle.py`, exact bilinear oracle.** Line of sight on the surface model
  `viewshed` declares (cell-centre heights, bilinear between them). The sightline
  is split at every patch boundary, the gap to the surface in each patch is a
  quadratic minimised in closed form, and near-ties are redone in exact rational
  arithmetic. Gives visibility and clearance per cell. Results are cached in
  `.cache/` by a hash of the inputs.
- **`oracle_rviewshed.py`, the r.viewshed model.** What GRASS `r.viewshed`
  actually computes, written from a description of the method (not GRASS's GPL
  code): each closer cell is reduced to three gradient samples (the two corners
  bounding it as seen from the observer, heights averaged from the four cells
  meeting there, and its centre), interpolated linearly in angle at the target's
  direction. It is not a ray-vs-bilinear-surface test, though the manual suggests
  so. It matches GRASS 8.4 on 100% of cells on every synthetic raster
  (`tests/test_oracle_rviewshed.py`), and differs from the bilinear model on up
  to 1.5% of cells on rough terrain.
- **`scenes.py`, analytic scenes.** Boxes, ridges, cones and thin walls on flat
  ground, true visibility from closed-form geometry, rasterised at 4, 2, 1 and 0.5 m.

## Tools (`adapters.py`)

Each adapter gets the same DEM, observer (always a cell centre), observer and
target heights, and returns a mask on the input grid. Tools that aren't installed
are skipped: the Rust crate (`viewshed-rs-sweep`, the default r.viewshed-model
sweep, and `viewshed-rs-bilinear`), the Python 0.0.1 reference, GDAL in all four
modes (curvature off), GRASS `r.viewshed` (one session per run; start-up reported
separately) and the PhotoFootprint sweep (`PHOTOFOOTPRINT=/path/to/repo`).

## Golden outputs (`golden.py`, `golden.json`)

Fixed scenes (single viewshed, ZTV forward and reciprocal, flat and curved,
no-data holes) with SHA-256 hashes of their inputs and exact outputs.
`tests/test_golden.py` fails if any output changes by a single cell, so a speed
change can't alter results unnoticed. Record new hashes with
`python -m conformance.golden --update` only after a change has been verified
cell for cell against the oracles.

## Earth curvature and refraction

Both oracles take `curvature`, `refraction` and `earth_radius`. In the bilinear
oracle the d²/(2R) drop keeps each patch's gap quadratic, so it stays exact.
`tests/test_curvature.py` checks the bilinear oracle against the analytic horizon
on flat ground, the r.viewshed oracle against GRASS `r.viewshed -c -r` on every
cell (GRASS uses its location's ellipsoid semi-major axis; with that radius the
match is exact over thousands of curvature-decided cells), and the Rust sweep
against the oracle on random rasters with tiny earth radii (so curvature decides
many cells). The d²/(2R) drop is within 1 mm of a true sphere out to 35 km.

## Not done yet

- xarray-spatial and WhiteboxTools adapters.
- An oracle for GDAL's model (Wang et al. reference planes), plus flat-topped and
  triangulated cells.
- Possible r.viewshed quirks to test: its search tree is keyed by distance alone,
  so cells at exactly equal distance may interfere; and the `-e` output may count
  the observer height twice. Neither showed up on the synthetic rasters.
- Real 1 m EA LiDAR DSM tiles (oracle on a sample plus shadow-edge cells).
- Peak memory, charts, the pinned Docker image and the CI job with gates.
- A faster (vectorised or compiled) bilinear oracle: about 0.3 ms per cell in pure Python.

`tests/test_rust_sweep.py` requires the Rust sweep to equal `oracle_rviewshed`
on every cell of every dataset, scene and 40 random rasters (no-data, integer
heights with many exact ties, distance limits).
