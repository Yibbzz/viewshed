"""The same cases as the Rust crate's tests, so both implementations agree."""

import numpy as np
import pytest

import viewshed

GT = (0.0, 1.0, 0.0, 41.0, 0.0, -1.0)   # 41 x 41, 1 m cells


def test_flat_ground_is_all_visible():
    assert viewshed.viewshed(np.full((41, 41), 50.0), GT, (20.5, 20.5)).all()


def wall():
    z = np.zeros((41, 41))
    z[15, :] = 5.0          # a 5 m wall along row 15, north of the observer
    return z


def test_wall_hides_ground_behind_it():
    seen = viewshed.viewshed(wall(), GT, (20.5, 20.5))
    assert seen[16, 20] and seen[15, 20] and seen[30, 20]
    assert not seen[5, 20]


def test_taller_observer_sees_over_the_wall():
    assert viewshed.viewshed(wall(), GT, (20.5, 20.5), observer_height=50.0)[5, 20]


def test_max_distance_and_nodata():
    z = np.zeros((41, 41))
    z[0, 0] = np.nan
    seen = viewshed.viewshed(z, GT, (20.5, 20.5), max_distance=10.0)
    assert seen[20, 25] and not seen[20, 35] and not seen[0, 0]


def test_bad_input_is_an_error():
    with pytest.raises(ValueError, match="rotated"):
        viewshed.viewshed(np.zeros((41, 41)), (0, 1, 0.1, 41, 0, -1), (20.5, 20.5))
    with pytest.raises(ValueError, match="outside"):
        viewshed.viewshed(np.zeros((41, 41)), GT, (100.0, 0.0))


def test_bilinear_surface_is_exact_on_a_plane():
    xs = np.arange(41) + 0.5
    ys = 41 - (np.arange(41) + 0.5)
    gx, gy = np.meshgrid(xs, ys)
    z = 3 + 0.2 * gx - 0.1 * gy
    px, py = np.array([10.3, 20.7]), np.array([30.2, 5.5])
    assert np.allclose(viewshed.surface_height(z, GT, px, py), 3 + 0.2 * px - 0.1 * py)
