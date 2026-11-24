"""The public API: types, shapes, defaults and errors. Accuracy is checked
cell by cell against the oracles and GRASS in ../conformance."""

import threading

import numpy as np
import pytest

import viewshed

GT = (0.0, 1.0, 0.0, 41.0, 0.0, -1.0)


def wall():
    z = np.zeros((41, 41))
    z[15, :] = 5.0
    return z


def test_returns_a_bool_mask_shaped_like_the_dem():
    seen = viewshed.viewshed(wall(), GT, (20.5, 20.5))
    assert seen.dtype == bool and seen.shape == (41, 41)
    assert seen[16, 20] and seen[15, 20] and seen[30, 20] and not seen[5, 20]


def test_float32_and_non_contiguous_input_give_the_same_answer():
    z = wall()
    base = viewshed.viewshed(z, GT, (20.5, 20.5))
    assert (viewshed.viewshed(z.astype(np.float32), GT, (20.5, 20.5)) == base).all()
    big = np.zeros((41, 82))
    big[:, ::2] = z
    assert (viewshed.viewshed(big[:, ::2], GT, (20.5, 20.5)) == base).all()
    assert (viewshed.viewshed(z.tolist(), list(GT), [20.5, 20.5]) == base).all()


def test_options():
    z = wall()
    assert viewshed.viewshed(z, GT, (20.5, 20.5), observer_height=50.0)[5, 20]
    # a 10 m target 15 m away is hidden by the 5 m wall 5 m away (sightline 4.4 m there);
    # a 20 m one is not (7.7 m)
    assert not viewshed.viewshed(z, GT, (20.5, 20.5), target_height=10.0)[5, 20]
    assert viewshed.viewshed(z, GT, (20.5, 20.5), target_height=20.0)[5, 20]
    near = viewshed.viewshed(z, GT, (20.5, 20.5), max_distance=10.0)
    assert near[20, 25] and not near[20, 35]
    ref = viewshed.viewshed(z, GT, (20.5, 20.5), surface="bilinear", curvature=False)
    assert ref[16, 20] and not ref[5, 20]


def test_curvature_sets_the_horizon_on_flat_ground():
    k, h = 0.13, 1.6
    radius = (1 - k) * 10.5 ** 2 / (2 * h)        # horizon at 10.5 cells
    seen = viewshed.viewshed(np.zeros((41, 41)), GT, (20.5, 20.5), observer_height=h,
                             refraction=k, earth_radius=radius)
    assert seen[20, 29] and not seen[20, 32]
    assert viewshed.viewshed(np.zeros((41, 41)), GT, (20.5, 20.5), earth_radius=radius,
                             curvature=False).all()


def test_ztv_counts_and_methods():
    z = wall()
    targets = [(20.5, 20.5, 30.0), (20.5, 3.5, 2.0)]
    fwd = viewshed.ztv(z, GT, targets)
    rec = viewshed.ztv(z, GT, targets, method="reciprocal")
    assert fwd.dtype == np.uint32 and fwd.shape == (41, 41)
    assert fwd.max() == 2 and (fwd <= 2).all()
    assert (fwd != rec).mean() < 0.05


def test_nodata():
    z = wall()
    z[0, 0] = np.nan
    assert not viewshed.viewshed(z, GT, (20.5, 20.5))[0, 0]
    with pytest.raises(ValueError, match="no height data"):
        viewshed.viewshed(z, GT, (0.5, 40.5))


@pytest.mark.parametrize("kwargs, match", [
    (dict(transform=(0, 1, 0.1, 41, 0, -1)), "rotated"),
    (dict(transform=(0, 1, 0, 41, 0, -2)), "square"),
    (dict(observer=(100.0, 0.0)), "outside"),
    (dict(observer_height=-1.0), "observer_height"),
    (dict(max_distance=0.0), "max_distance"),
    (dict(surface="nope"), "surface"),
    (dict(dem=np.zeros(5)), "2-D"),
])
def test_errors(kwargs, match):
    args = dict(dem=np.zeros((41, 41)), transform=GT, observer=(20.5, 20.5))
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        viewshed.viewshed(**args)


def test_ztv_errors():
    with pytest.raises(ValueError, match="method"):
        viewshed.ztv(np.zeros((9, 9)), (0, 1, 0, 9, 0, -1), [(4.5, 4.5, 1.0)], method="nope")
    with pytest.raises(ValueError, match="outside"):
        viewshed.ztv(np.zeros((9, 9)), (0, 1, 0, 9, 0, -1), [(40.5, 4.5, 1.0)])


def test_runs_in_parallel_threads():
    # the GIL is released, so threads can call it at the same time
    z = np.random.default_rng(0).normal(0, 3, (201, 201)).cumsum(0)
    gt = (0.0, 1.0, 0.0, 201.0, 0.0, -1.0)
    expected = viewshed.viewshed(z, gt, (100.5, 100.5))
    results = [None] * 4

    def work(i):
        results[i] = viewshed.viewshed(z, gt, (100.5, 100.5))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all((r == expected).all() for r in results)


def test_shadow_of_a_wall():
    # flat ground, a 5 m wall in column 20, sun due east at 40 degrees (tan = 0.839): a cell
    # c < 20 looks east along its row and its ray is 0.839 (20 - c) m up at the wall's top
    # (5 m): shadowed for c in 15..19 (4.2 m at c = 15), lit at c = 14 (5.03 m) and beyond
    # the wall. (At exactly 45 degrees c = 15 is shadowed too: tan(45 deg) is 0.9999999999999999
    # in floats, and shadows are exact for the float sun vector.)
    z = np.zeros((41, 41))
    z[:, 20] = 5.0
    lit = viewshed.shadow(z, GT, azimuth=90.0, elevation=40.0, curvature=False)
    assert lit.dtype == bool and lit.shape == (41, 41)
    row = lit[20]
    assert not row[15:20].any()
    assert row[:15].all() and row[20:].all()
    # sun due west: the mirror image
    lit_w = viewshed.shadow(z, GT, azimuth=270.0, elevation=40.0, curvature=False)
    assert (lit_w[20, ::-1] == row).all()


def test_shadow_no_data_and_errors():
    z = np.zeros((11, 11))
    z[5, 5] = np.nan
    lit = viewshed.shadow(z, GT, 180.0, 30.0)
    assert not lit[5, 5] and lit[0, 0]
    for elevation in (0.0, 90.0, -5.0, float("nan")):
        with pytest.raises(ValueError):
            viewshed.shadow(z, GT, 180.0, elevation)
    with pytest.raises(ValueError):
        viewshed.shadow(z, GT, 180.0, 30.0, target_height=-1.0)
    with pytest.raises(ValueError):
        viewshed.shadow(z, GT, 180.0, 30.0, refraction=1.5)


def test_reverse_viewshed_equals_a_viewshed_from_every_cell():
    rng = np.random.default_rng(5)
    n = 15
    z = rng.normal(0, 2, (n, n)).cumsum(1)
    z[4, 9] = np.nan
    gt = (0.0, 1.0, 0.0, float(n), 0.0, -1.0)
    tr, tc = 6, 3
    kw = dict(curvature=True, refraction=0.13, earth_radius=40.0, max_distance=11.0)
    for surface in ("rviewshed", "bilinear"):
        rev = viewshed.reverse_viewshed(z, gt, (tc + 0.5, n - tr - 0.5), target_height=4.0,
                                        observer_height=1.5, surface=surface, **kw)
        assert rev.dtype == bool and rev.shape == z.shape and not rev[4, 9]
        compared = 0
        for r in range(n):
            for c in range(n):
                try:
                    seen = viewshed.viewshed(z, gt, (c + 0.5, n - r - 0.5), observer_height=1.5,
                                             target_height=4.0, surface=surface, **kw)
                except ValueError:   # no-data, or (bilinear) a centre beside a no-data patch
                    continue
                assert rev[r, c] == seen[tr, tc], (surface, r, c)
                compared += 1
        assert compared > 200 and 0 < rev.sum() < compared


def test_reverse_viewshed_errors_name_the_target():
    z = wall()
    with pytest.raises(ValueError, match="the target is outside the raster"):
        viewshed.reverse_viewshed(z, GT, (100.0, 0.0))
    z[20, 20] = np.nan
    with pytest.raises(ValueError, match="no height data under the target"):
        viewshed.reverse_viewshed(z, GT, (20.5, 20.5))
    with pytest.raises(ValueError, match="target_height"):
        viewshed.reverse_viewshed(z, GT, (5.5, 5.5), target_height=-1.0)
    with pytest.raises(ValueError, match="surface"):
        viewshed.reverse_viewshed(z, GT, (5.5, 5.5), surface="nope")
