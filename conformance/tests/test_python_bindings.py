"""The Python package must give the r.viewshed-model oracle's answer on every
cell (flat earth and with curvature), and the same ZTV as the Rust crate."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import oracle_rviewshed as orv  # noqa: E402
from conformance.adapters import PythonBindings  # noqa: E402

PY = PythonBindings()
pytestmark = pytest.mark.skipif(not PY.available(),
                                reason="viewshed bindings not built (maturin develop in python/)")

from test_curvature import curved_jobs  # noqa: E402
from test_rust_sweep import fixed_jobs, random_jobs  # noqa: E402


@pytest.mark.parametrize("job", [*fixed_jobs(), *random_jobs(20, seed=3), *curved_jobs(12, seed=4)])
def test_bindings_equal_oracle(job):
    theirs = orv.viewshed(job.dem, job.gt, job.observer, job.observer_height, job.target_height,
                          job.max_distance, **job.curve_kwargs())
    assert (PY.run(job) == theirs).all()


def test_ztv_forward_equals_viewshed_from_every_cell():
    import viewshed
    rng = np.random.default_rng(8)
    n = 15
    z = rng.normal(0, 2, (n, n)).cumsum(1)
    gt = (0.0, 1.0, 0.0, float(n), 0.0, -1.0)
    targets = [(3.5, 11.5, 6.0), (10.5, 2.5, 0.0)]
    kw = dict(curvature=True, refraction=0.13, earth_radius=40.0)
    counts = viewshed.ztv(z, gt, targets, observer_height=1.5, **kw)
    for r in range(n):
        for c in range(n):
            expect = 0
            for tx, ty, h in targets:
                seen = viewshed.viewshed(z, gt, (c + 0.5, n - r - 0.5), observer_height=1.5,
                                         target_height=h, **kw)
                expect += seen[int(n - ty), int(tx)]
            assert counts[r, c] == expect
