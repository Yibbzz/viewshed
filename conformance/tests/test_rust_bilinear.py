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


@pytest.mark.parametrize("job", [*fixed_jobs(), *random_jobs()])
def test_bilinear_equals_oracle(job):
    try:
        theirs = _oracle(job)
    except ValueError:                    # observer on no-data
        return
    ours = RUST.run(job)
    assert (ours == theirs).all(), f"differs at {np.argwhere(ours != theirs)[:5].tolist()}"
