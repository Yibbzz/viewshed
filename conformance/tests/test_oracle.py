import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from conformance import oracle  # noqa: E402
from viewshed import reference  # noqa: E402

GT = (0.0, 1.0, 0.0, 31.0, 0.0, -1.0)
OBS = (15.5, 15.5)


def test_flat_plane_all_visible_without_exact_fallback():
    vis, clr, n_exact = oracle.viewshed(np.full((31, 31), 7.0), GT, OBS)
    assert vis.all() and n_exact == 0


def test_plane_is_all_visible_from_above():
    yy, xx = np.mgrid[0:31, 0:31]
    vis, _, _ = oracle.viewshed(0.3 * xx - 0.2 * yy, GT, OBS)
    assert vis.all()


def test_wall_shadow_and_clearance_sign():
    z = np.zeros((31, 31)); z[10, :] = 5.0
    vis, clr, _ = oracle.viewshed(z, GT, OBS)
    assert vis[11, 15] and vis[10, 15] and vis[25, 15]
    assert not vis[3, 15] and clr[3, 15] < 0
    assert clr[25, 15] > 0


def test_exact_grazing_tie_counts_as_visible():
    # eye at 2 m over a 2 m ridge row, flat 2 m plateau beyond: the sightline grazes exactly
    z = np.zeros((31, 31)); z[:12, :] = 2.0
    vis, _, n_exact = oracle.viewshed(z, GT, OBS, observer_height=2.0)
    assert vis[5, 15] and n_exact > 0


def test_nodata_never_blocks_and_is_never_visible():
    z = np.zeros((31, 31)); z[10, :] = np.nan
    vis, _, _ = oracle.viewshed(z, GT, OBS)
    assert not vis[10, 15] and vis[3, 15]


def test_reference_implementation_mostly_agrees():
    rng = np.random.default_rng(1)
    z = np.cumsum(np.cumsum(rng.normal(0, 0.3, (31, 31)), 0), 1) / 5
    ours, _, _ = oracle.viewshed(z, GT, OBS)
    ref = reference.viewshed(z, GT, OBS)
    assert 0.1 < ours.mean() < 0.95
    assert (ours == ref).mean() > 0.98   # the NumPy reference samples every 1/4 cell
