"""The public API: types, shapes, defaults and errors. Accuracy is checked
cell by cell against the oracles and GRASS in ../conformance."""

import math
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
                except ValueError:   # only a no-data cell has no ground under its centre
                    assert np.isnan(z[r, c]) and not rev[r, c]
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


from datetime import datetime, timedelta, timezone  # noqa: E402

UTC = timezone.utc


def test_sun_position_is_plausible_and_needs_a_time_zone():
    # London, midsummer solar noon (~12:59 UTC +- a few minutes): sun due south, ~62 deg up
    p = viewshed.sun_position(51.5, -0.13, datetime(2026, 6, 21, 12, 3, tzinfo=UTC))
    assert abs(p.azimuth - 180.0) < 3.0 and abs(p.elevation - 62.0) < 0.5
    assert p.elevation > p.elevation_true                     # refraction lifts the sun
    assert viewshed.sun_position(51.5, -0.13, datetime(2026, 6, 21, 12, 3, tzinfo=UTC).timestamp()) == p
    with pytest.raises(ValueError, match="time zone"):
        viewshed.sun_position(51.5, -0.13, datetime(2026, 6, 21, 12, 3))
    with pytest.raises(ValueError):
        viewshed.sun_position(91.0, 0.0, 0.0)


def _day():
    return datetime(2026, 6, 21, 0, 0, tzinfo=UTC), datetime(2026, 6, 22, 0, 0, tzinfo=UTC)


def test_sun_hours_flat_ground_gets_the_whole_day_and_a_wall_shades_its_north():
    z = np.zeros((41, 41))
    start, end = _day()
    h = viewshed.sun_hours(z, GT, start, end, step=timedelta(minutes=30), lat=51.5, lon=-0.13)
    assert h.dtype == np.float64 and h.shape == z.shape
    day = h[20, 20]
    assert 16.0 <= day <= 17.5 and (h == day).all()         # ~16.6 h of daylight in London
    z[20, 10:31] = 30.0                                       # an east-west wall, 30 m high
    z[0, 0] = np.nan
    h = viewshed.sun_hours(z, GT, start, end, step=timedelta(minutes=30), lat=51.5, lon=-0.13)
    assert np.isnan(h[0, 0])
    # row 0 is north: row 18 is just north of the wall, row 22 just south of it. In a
    # London midsummer the sun rises in the north-east and sets in the north-west, so
    # the south side loses the early and late sun, the north side the midday sun
    assert h[18, 20] < day - 4.0 and h[22, 20] < day - 4.0
    midday = viewshed.sun_hours(z, GT, datetime(2026, 6, 21, 10, tzinfo=UTC),
                                datetime(2026, 6, 21, 16, tzinfo=UTC), step=1800, lat=51.5, lon=-0.13)
    assert midday[18, 20] == 0.0 and midday[22, 20] == 6.0 and midday[20, 5] == 6.0


def test_sun_hours_is_the_sum_of_shadow_maps_at_its_samples():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 3, (25, 25)).cumsum(0)
    gt = (500000.0, 2.0, 0.0, 5650000.0, 0.0, -2.0)
    start, end = datetime(2026, 3, 1, 6, tzinfo=UTC), datetime(2026, 3, 1, 18, tzinfo=UTC)
    h = viewshed.sun_hours(z, gt, start, end, step=3600, lat=51.0, lon=-3.0, grid_convergence=1.5,
                           target_height=0.5)
    want = np.zeros(z.shape)
    for i in range(12):
        t = start + timedelta(minutes=30 + 60 * i)
        p = viewshed.sun_position(51.0, -3.0, t)
        if p.elevation > 0:
            want += viewshed.shadow(z, gt, (p.azimuth - 1.5) % 360.0, p.elevation, target_height=0.5)
    assert (h == want).all()


def test_sun_hours_grid_convergence_from_crs():
    pytest.importorskip("pyproj")
    from viewshed import _centre_and_convergence
    # UTM 30N, raster centred ~3 deg east of the central meridian (-3 deg) at 51 N:
    # convergence ~ (lon - lon0) sin(lat) ~ +2.3 deg (grid north east of true north)
    z = np.zeros((10, 10))
    gt = (709000.0, 10.0, 0.0, 5655000.0, 0.0, -10.0)
    lat, lon, conv = _centre_and_convergence(z, gt, "EPSG:32630")
    assert abs(lat - 51.0) < 0.1 and abs(lon - 0.0) < 0.1
    assert conv == pytest.approx((lon + 3.0) * math.sin(math.radians(lat)), abs=0.02)
    start, end = _day()
    a = viewshed.sun_hours(z, gt, start, end, step=3600, crs="EPSG:32630")
    b = viewshed.sun_hours(z, gt, start, end, step=3600, lat=lat, lon=lon, grid_convergence=conv)
    assert (a == b).all()


def test_sun_hours_needs_a_place():
    with pytest.raises(ValueError, match="lat and lon"):
        viewshed.sun_hours(np.zeros((5, 5)), GT, *_day())


def test_intervisible_shapes_broadcasting_and_errors():
    z = wall()
    mast = (20.5, 20.5, 1.6)
    # the wall is row 15 (y = 25.5), north of the mast; larger y is further north
    assert viewshed.intervisible(z, GT, mast, (20.5, 5.5, 0.0)) is True        # south: open
    assert viewshed.intervisible(z, GT, mast, (20.5, 35.5, 0.0)) is False      # behind the wall
    many = [(20.5, 5.5, 0.0), (20.5, 35.5, 0.0), (20.5, 35.5, 20.0)]   # 20 m clears the wall
    got = viewshed.intervisible(z, GT, mast, many)                              # one to many
    assert got.dtype == bool and got.tolist() == [True, False, True]
    assert viewshed.intervisible(z, GT, many, mast).tolist() == got.tolist()   # order-free
    # the same as the bilinear viewshed for cell-centre targets
    seen = viewshed.viewshed(z, GT, (20.5, 20.5), observer_height=1.6, surface="bilinear")
    cells = [(c + 0.5, 41 - r - 0.5, 0.0) for r in range(0, 41, 3) for c in range(0, 41, 3)]
    assert viewshed.intervisible(z, GT, mast, cells).tolist() == \
        [bool(seen[r, c]) for r in range(0, 41, 3) for c in range(0, 41, 3)]
    with pytest.raises(ValueError, match="pair 1: a point is outside"):
        viewshed.intervisible(z, GT, mast, [(1.5, 1.5, 0.0), (99.0, 1.0, 0.0)])
    with pytest.raises(ValueError, match="pair 0: heights"):
        viewshed.intervisible(z, GT, mast, (1.5, 1.5, -1.0))
    with pytest.raises(ValueError, match="3 points and b has 2"):
        viewshed.intervisible(z, GT, many, many[:2])


def test_horizon_map_geometry_edges_and_shadow_consistency():
    z = wall()                                                  # 5 m wall along row 15
    h = viewshed.horizon_map(z, GT, azimuth=0.0, curvature=False)   # looking north
    assert h.dtype == np.float64 and h.shape == z.shape
    assert abs(h[20, 20] - 45.0) < 1e-9                         # 5 m up, 5 cells away
    assert (np.abs(h[0, :]) < 1e-12).all()                      # top row: the half-cell rim, flat
    assert abs(h[5, 20]) < 1e-12                                # beyond the wall: flat ahead
    zt = z.copy()
    zt[0, :] = np.nan                                           # only no-data ahead of row 1
    assert (viewshed.horizon_map(zt, GT, 0.0)[1, :] == -90.0).all()
    south = viewshed.horizon_map(z, GT, azimuth=180.0, curvature=False)
    assert abs(south[10, 20] - 45.0) < 1e-9                     # the wall seen from the north
    zn = z.copy()
    zn[3, 3] = np.nan
    assert np.isnan(viewshed.horizon_map(zn, GT, 0.0)[3, 3])
    zc = np.zeros((41, 41))
    curved = viewshed.horizon_map(zc, (0.0, 100.0, 0.0, 4100.0, 0.0, -100.0), 90.0)
    assert (curved[:, :-1] == 0.0).all()                        # flat: the level limit beside the cell, curved or not
    zc[20, 40] = 10.0                                           # a 10 m post 4 km east of (20, 0)
    post = viewshed.horizon_map(zc, (0.0, 100.0, 0.0, 4100.0, 0.0, -100.0), 90.0)
    drop = 4000.0 ** 2 * (1 - 0.13) / (2 * 6371008.8)           # the earth curves away
    assert abs(post[20, 0] - np.degrees(np.arctan((10.0 - drop) / 4000.0))) < 1e-9
    # shadow(az, el) is lit exactly when el is above the horizon (away from ties)
    rng = np.random.default_rng(4)
    zr = rng.normal(0, 2, (40, 40)).cumsum(0)
    for az, el in [(135.0, 20.0), (250.0, 8.0), (10.0, 35.0)]:
        hor = viewshed.horizon_map(zr, GT, az)
        lit = viewshed.shadow(zr, GT, az, el)
        clear = np.abs(hor - el) > 1e-6
        assert (lit[clear] == (el > hor[clear])).all()
        assert 0 < lit.sum() < lit.size
    with pytest.raises(ValueError):
        viewshed.horizon_map(z, GT, 0.0, target_height=-1.0)
