"""The Rust crate's exact bilinear mode must equal the bilinear oracle on every cell.

The oracle is oracle_fast.fastest(): oracle_numba (bit-identical to oracle.py,
tests/test_oracle_fast.py) when numba is installed, else oracle.py itself."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import datasets, oracle_fast, scenes  # noqa: E402
from conformance.adapters import Job, Rust  # noqa: E402

RUST = Rust("bilinear")
ORACLE = oracle_fast.fastest()
pytestmark = pytest.mark.skipif(not RUST.available(), reason="Rust toolchain not available")


def random_jobs(n_jobs=40, seed=11):
    rng = np.random.default_rng(seed)
    for k in range(n_jobs):
        n = int(rng.integers(5, 40))
        dem = rng.normal(0, rng.uniform(0.1, 5), (n, n)).cumsum(0)
        if k % 3 == 0:
            dem[rng.random((n, n)) < 0.1] = np.nan
        if k % 5 == 0:
            dem = np.round(dem)        # integer heights: many exact ties
        res = [1.0, 2.0, 0.5, 0.3][k % 4]
        gt = (1000.0 * (k % 3), res, 0.0, 5000.0 + n * res, 0.0, -res)
        c0, r0 = rng.integers(0, n, 2)
        if np.isnan(dem[r0, c0]):
            dem[r0, c0] = 0.0
        # observer on a cell centre, a corner or anywhere in the cell
        fx, fy = [(0.5, 0.5), (0.0, 0.0), tuple(rng.random(2))][k % 3]
        obs = (gt[0] + (c0 + fx) * res, gt[3] - (r0 + fy) * res)
        md = None if k % 4 else float(rng.uniform(2, n)) * res
        curve = bool(k % 2)
        yield Job(dem, gt, obs, float(rng.uniform(0, 5)), float(rng.choice([0.0, 0.0, 1.0])), md,
                  curve, 0.13, [6371008.8, 30.0 * res][(k // 2) % 2])


def near_lattice_jobs(n_jobs=36, seed=23):
    """Observers a rounding error off a cell centre or a grid line (UTM-scale origins,
    as real georeferencing gives), hydro-flattened plateaus (exact ties over many
    patches) and integer heights: sightlines that run along, or pass within ~1e-11
    of, grid lines and lattice points for their whole length."""
    rng = np.random.default_rng(seed)
    for k in range(n_jobs):
        n = int(rng.integers(20, 64))
        dem = rng.normal(0, rng.uniform(0.05, 1.0), (n, n)).cumsum(0).cumsum(1) * 0.05 + 550.0
        for _ in range(int(rng.integers(1, 4))):  # flat lakes
            r0, c0 = rng.integers(0, n - 4, 2)
            h, w = rng.integers(3, n // 2, 2)
            dem[r0:r0 + h, c0:c0 + w] = np.float32(dem[r0:r0 + h, c0:c0 + w].min())
        if k % 4 == 1:
            dem = np.round(dem)
        if k % 6 == 5:
            dem[rng.random((n, n)) < 0.05] = np.nan
        res = [10.0, 1.0, 30.0, 0.5][k % 4]
        x0 = float(rng.uniform(1e5, 9e5)) + 0.69978725986
        y0 = float(rng.uniform(1e6, 9e6)) + 0.1998547595
        gt = (x0, res, 0.0, y0, 0.0, -res)
        c0, r0 = (int(v) for v in rng.integers(2, n - 2, 2))
        if np.isnan(dem[r0, c0]):
            dem[r0, c0] = 550.0
        # cell centre, corner or edge midpoint, computed in floats from the origin
        fx, fy = [(0.5, 0.5), (0.0, 0.0), (0.5, 0.0), (0.0, 0.5)][k % 4]
        obs = (x0 + (c0 + fx) * res, y0 - (r0 + fy) * res)
        if k % 3 == 2:  # a hair off as well
            obs = (obs[0] + float(rng.choice([-1, 1])) * 1e-9 * res, obs[1])
        th = [0.0, 0.0, 0.5][k % 3]
        yield Job(dem, gt, obs, float(rng.choice([1.6, 10.0])), th, None,
                  bool(k % 2), 0.13, 6371008.8)


def near_blocker_jobs(n_jobs=24, seed=29):
    """Short walls 1-3 cells from the observer, taller than the eye, on gentle terrain:
    targets behind them are hidden by nothing else, so the stretch of every sightline
    nearest the observer (terrain the horizon structures leave to the walk, or bound
    crudely) decides them."""
    rng = np.random.default_rng(seed)
    for k in range(n_jobs):
        n = int(rng.integers(30, 64))
        dem = rng.normal(0, 0.2, (n, n)).cumsum(0).cumsum(1) * 0.02 + 100.0
        res = [1.0, 10.0, 2.0][k % 3]
        x0, y0 = 500000.0 + 0.37, 4000000.0 + 0.61
        gt = (x0, res, 0.0, y0, 0.0, -res)
        c0, r0 = (int(v) for v in rng.integers(8, n - 8, 2))
        eye = float(rng.choice([1.6, 2.0]))
        # one to three wall stubs at 1-3 cells, a few cells long, eye-high or more
        for _ in range(int(rng.integers(1, 4))):
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(1.0, 3.0)
            wr, wc = int(round(r0 + dist * np.sin(ang))), int(round(c0 + dist * np.cos(ang)))
            if (wr, wc) == (r0, c0):
                continue
            h = dem[r0, c0] + eye + float(rng.uniform(-0.3, 3.0))
            for t in range(-2, 3):
                rr = wr + int(round(t * np.cos(ang)))
                cc = wc - int(round(t * np.sin(ang)))
                if 0 <= rr < n and 0 <= cc < n and (rr, cc) != (r0, c0):
                    dem[rr, cc] = h
        fx, fy = [(0.5, 0.5), (0.37, 0.61), (0.0, 0.5)][k % 3]
        obs = (x0 + (c0 + fx) * res, y0 - (r0 + fy) * res)
        yield Job(dem, gt, obs, eye, [0.0, 0.0, 1.0][k % 3], None, bool(k % 2), 0.13, 6371008.8)


def beside_nodata_jobs(n_jobs=24, seed=31):
    """Observers exactly on a grid vertex (a cell centre) or on a patch edge, where some
    of the patches containing the point have no surface and others do: the ground there
    comes from any patch with a surface (they agree along shared edges)."""
    rng = np.random.default_rng(seed)
    for k in range(n_jobs):
        n = int(rng.integers(8, 30))
        dem = rng.normal(0, rng.uniform(0.2, 3), (n, n)).cumsum(0)
        if k % 2:
            dem = np.round(dem)
        res = [1.0, 0.5, 2.0][k % 3]
        gt = (0.0, res, 0.0, n * res, 0.0, -res)
        r0, c0 = (int(v) for v in rng.integers(1, n - 2, 2))
        dem[r0, c0] = dem[r0, c0 + 1] = dem[r0 + 1, c0] = 1.0
        kind = k % 3     # vertex, vertical edge, horizontal edge (in centre coordinates)
        # knock out the lower-right patch (the one a floor() lookup picks), and at a
        # vertex sometimes one more, never all the patches containing the point
        dem[rng.random((n, n)) < 0.03] = np.nan
        dem[r0:r0 + 2, c0:c0 + 2] = 1.0
        dem[r0 - 1:r0 + 2, c0 - 1] = np.where(np.isnan(dem[r0 - 1:r0 + 2, c0 - 1]), 2.0,
                                              dem[r0 - 1:r0 + 2, c0 - 1])
        dem[r0 - 1, c0:c0 + 2] = np.where(np.isnan(dem[r0 - 1, c0:c0 + 2]), 2.0, dem[r0 - 1, c0:c0 + 2])
        dem[r0 + 1, c0 + 1] = np.nan
        if kind == 0 and k % 2:
            dem[r0 - 1, c0 + 1] = np.nan
        fx, fy = [(0.5, 0.5), (0.5, 0.5 + float(rng.uniform(0.05, 0.95))),
                  (0.5 + float(rng.uniform(0.05, 0.95)), 0.5)][kind]
        obs = (gt[0] + (c0 + fx) * res, gt[3] - (r0 + fy) * res)
        yield Job(dem, gt, obs, float(rng.uniform(0, 4)), [0.0, 1.0][k % 2],
                  None if k % 4 else 6.0 * res, bool(k % 2), 0.13, [6371008.8, 30.0 * res][k % 2])


def fixed_jobs():
    for f in datasets.DATASETS.values():
        for th in datasets.TARGET_HEIGHTS:
            yield Job(f(), datasets.GT, datasets.OBSERVER, 1.6, th)
    for sc in scenes.SCENES:
        dem, gt, obs = sc.raster(2.0)
        yield Job(dem, gt, obs, 1.6, 0.0)


def _oracle(job):
    kw = {"clearance": False} if ORACLE.__name__.endswith("oracle_numba") else {}
    return ORACLE.viewshed(job.dem, job.gt, job.observer, job.observer_height, job.target_height,
                           job.max_distance, **job.curve_kwargs(), **kw)[0]


@pytest.mark.parametrize("mode", ["rows", "horizon", "sector", "sector4"])
@pytest.mark.parametrize("job", [*fixed_jobs(), *random_jobs(), *near_lattice_jobs(), *near_blocker_jobs(),
                                 *beside_nodata_jobs()])
def test_bilinear_equals_oracle(job, mode, monkeypatch):
    # the crate picks a mode per raster (row-major walk, or rings with the horizon
    # filters); both are exact, and both are checked here on every job
    # "sector4": the sector-parallel mode with 4 sectors, so that culling blocks fit inside
    # sectors (and are exercised) even on these small rasters
    monkeypatch.setenv("VIEWSHED_MODE", "sector" if mode == "sector4" else mode)
    if mode.startswith("sector"):
        monkeypatch.setenv("VIEWSHED_CHECK", "1")  # the skips' invariants, asserted
    if mode == "sector4":
        monkeypatch.setenv("VIEWSHED_SECN", "4")
    try:
        theirs = _oracle(job)
    except ValueError:                    # observer on no-data
        return
    ours = RUST.run(job)
    assert (ours == theirs).all(), f"differs at {np.argwhere(ours != theirs)[:5].tolist()}"


def test_observer_beside_nodata_has_ground():
    """Every beside_nodata job has ground under the observer (in the oracle, so the
    equality test above really compares viewsheds rather than skipping)."""
    from conformance import oracle
    for job in beside_nodata_jobs():
        assert oracle._ground_exact(job.dem, job.gt, *job.observer) is not None
