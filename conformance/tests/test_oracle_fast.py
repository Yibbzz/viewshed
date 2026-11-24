"""oracle_numba against oracle.py. The full pass must be bit-identical: same
visibility, same clearance (every bit, NaN where oracle.py has NaN), same count of
cells sent to exact arithmetic. The visibility-only pass (clearance=False) must give
the same visibility.

The cases lean on ties: integer heights, observers on cell centres, corners and
edges (sightlines through lattice points), plateaus that graze, no-data, curvature
with tiny earth radii, max_distance, and non-unit cells."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformance import oracle  # noqa: E402

numba_oracle = pytest.importorskip("conformance.oracle_numba")   # numba is optional


from cases import CASES  # noqa: E402,F401  (shared, numba-free)


@pytest.mark.parametrize("name,z,gt,obs,oh,th,md,curve", CASES, ids=[c[0] for c in CASES])
def test_bit_identical_to_reference(name, z, gt, obs, oh, th, md, curve):
    try:
        want = oracle.viewshed(z, gt, obs, oh, th, md, **curve)
    except ValueError as e:                       # observer on no-data: both must refuse
        with pytest.raises(ValueError, match=str(e)):
            numba_oracle.viewshed(z, gt, obs, oh, th, md, **curve)
        return
    got = numba_oracle.viewshed(z, gt, obs, oh, th, md, **curve)
    assert (got[0] == want[0]).all(), f"visibility differs at {np.argwhere(got[0] != want[0])[:5]}"
    a, b = got[1].view(np.uint64), want[1].view(np.uint64)
    assert (a == b).all(), f"clearance bits differ at {np.argwhere(a != b)[:5]}"
    assert got[2] == want[2]
    vis, clr, _ = numba_oracle.viewshed(z, gt, obs, oh, th, md, clearance=False, **curve)
    assert clr is None
    assert (vis == want[0]).all(), f"visibility-only differs at {np.argwhere(vis != want[0])[:5]}"
