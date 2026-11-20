import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance.metrics import compare, edge_cells  # noqa: E402
from conformance.scenes import Box, Cone, Ridge, Scene, _blocked_cone, _blocked_convex  # noqa: E402

EYE = np.array([0.0, 0.0, 1.6])


def seg_to(*pts):
    return np.array(pts, float)


def test_box_blocks_through_but_not_over_or_on_top():
    b = Box(10, 12, -5, 5, 4)
    pts = seg_to([20, 0, 0], [20, 0, 50], [11, 0, 4], [20, 20, 0])
    assert _blocked_convex(b, EYE, pts).tolist() == [True, False, True, False]
    # the roof is hidden from 1.6 m (the near wall is in the way), but visible from 10 m
    assert not _blocked_convex(b, np.array([0, 0, 10.0]), seg_to([11, 0, 4]))[0]


def test_ridge_and_cone_shadow():
    r = Ridge(10, 4, -50, 50, 5)
    assert _blocked_convex(r, EYE, seg_to([20, 0, 0], [10, 0, 5], [8, 0, 3]))\
        .tolist() == [True, False, False]
    c = Cone(10, 0, 4, 5)
    assert _blocked_cone(c, EYE, seg_to([20, 0, 0], [10, 0, 5], [20, 10, 0])).tolist() == \
        [True, False, False]


def test_scene_raster_centres_observer():
    s = Scene("t", [Box(10, 12, -5, 5, 4)])
    for res in (4.0, 2.0, 1.0, 0.5):
        dem, gt, obs = s.raster(res)
        n = dem.shape[0]
        assert n % 2 == 1
        assert gt[0] + (n // 2 + 0.5) * res == 0.0 and gt[3] - (n // 2 + 0.5) * res == 0.0


def test_truth_on_open_ground_is_all_visible():
    assert Scene("open", []).truth(2.0).all()


def test_metrics_counts():
    truth = np.zeros((5, 5), bool); truth[:, :2] = True
    mask = truth.copy(); mask[0, 0] = False; mask[4, 4] = True
    zone = np.ones_like(truth)
    m = compare(truth, mask, zone)
    assert m["cells"] == 25 and abs(m["wrongly_hidden"] - 0.04) < 1e-12
    assert abs(m["wrongly_visible"] - 0.04) < 1e-12
    assert edge_cells(truth, zone)[:, 1:3].all() and not edge_cells(truth, zone)[:, 4].any()
