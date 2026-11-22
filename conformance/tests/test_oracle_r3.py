"""The gridlines (R3) oracle: every cell visible in the bilinear model is visible in
R3 (R3 checks a subset of the same surface), and the compiled oracle_r3_numba
equals the exact oracle_r3 on every cell of the tie-heavy cases."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformance import oracle, oracle_r3  # noqa: E402
from test_oracle_fast import CASES  # noqa: E402


@pytest.mark.parametrize("name,z,gt,obs,oh,th,md,curve", CASES, ids=[c[0] for c in CASES])
def test_r3_sees_at_least_what_bilinear_sees(name, z, gt, obs, oh, th, md, curve):
    try:
        bil = oracle.viewshed(z, gt, obs, oh, th, md, **curve)[0]
    except ValueError:
        return
    r3 = oracle_r3.viewshed(z, gt, obs, oh, th, md, **curve)
    assert not (bil & ~r3).any(), f"bilinear-visible, R3-hidden at {np.argwhere(bil & ~r3)[:5]}"


@pytest.mark.parametrize("name,z,gt,obs,oh,th,md,curve", CASES, ids=[c[0] for c in CASES])
def test_compiled_r3_equals_exact_r3(name, z, gt, obs, oh, th, md, curve):
    fast = pytest.importorskip("conformance.oracle_r3_numba")
    try:
        want = oracle_r3.viewshed(z, gt, obs, oh, th, md, **curve)
    except ValueError:
        return
    got, _ = fast.viewshed(z, gt, obs, oh, th, md, **curve)
    assert (got == want).all(), f"differs at {np.argwhere(got != want)[:5]}"
