"""The Rust sweep must equal the r.viewshed-model oracle on every cell."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import datasets, oracle_rviewshed as orv, scenes  # noqa: E402
from conformance.adapters import Job, Rust  # noqa: E402

RUST = Rust("rviewshed")
pytestmark = pytest.mark.skipif(not RUST.available(), reason="Rust toolchain not available")


def random_jobs(n_jobs=40, seed=7):
    rng = np.random.default_rng(seed)
    for k in range(n_jobs):
        n = int(rng.integers(5, 40))
        dem = rng.normal(0, rng.uniform(0.1, 5), (n, n)).cumsum(0)
        if k % 3 == 0:
            dem[rng.random((n, n)) < 0.1] = np.nan
        if k % 5 == 0:
            dem = np.round(dem)        # integer heights: many exact ties
        r0, c0 = rng.integers(0, n, 2)
        if np.isnan(dem[r0, c0]):
            dem[r0, c0] = 0.0
        md = None if k % 4 else float(rng.uniform(2, n))
        yield Job(dem, (0.0, 1.0, 0.0, float(n), 0.0, -1.0), (c0 + 0.5, n - r0 - 0.5),
                  float(rng.uniform(0, 5)), float(rng.choice([0.0, 0.0, 1.0])), md)


def fixed_jobs():
    for f in datasets.DATASETS.values():
        for th in datasets.TARGET_HEIGHTS:
            yield Job(f(), datasets.GT, datasets.OBSERVER, 1.6, th)
    for sc in scenes.SCENES:
        dem, gt, obs = sc.raster(2.0)
        yield Job(dem, gt, obs, 1.6, 0.0)


@pytest.mark.parametrize("job", [*fixed_jobs(), *random_jobs()])
def test_sweep_equals_oracle(job):
    ours = RUST.run(job)
    theirs = orv.viewshed(job.dem, job.gt, job.observer, job.observer_height,
                          job.target_height, job.max_distance)
    assert (ours == theirs).all()
