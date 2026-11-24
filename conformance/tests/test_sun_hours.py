"""Sun-hours oracle: flat terrain gets the sampled day length; a wall delays the sun on its
west side by the shadow's time; weights add; no-data is NaN; samples follow the midpoint rule."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import shadow_oracle as so  # noqa: E402
from conformance import sun_hours as sh  # noqa: E402
from conformance.sunpos import sun_position  # noqa: E402

GT = (500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0)
DAY = 1719921600.0          # 2024-07-02 12:00 UTC
LAT, LON = 47.0, 8.0


def test_flat_gets_the_sampled_day_length():
    z = np.zeros((6, 7))
    step = 1800.0
    h = sh.sun_hours(z, GT, LAT, LON, DAY - 43200, DAY + 43200, step)
    up = sum(1 for i in range(48) if sun_position(LAT, LON, DAY - 43200 + (i + 0.5) * step)[1] > 0)
    assert np.all(h == up * 0.5)
    assert 15.0 <= up * 0.5 <= 16.5            # Swiss midsummer day length


def test_wall_shades_west_side_in_the_morning_only():
    z = np.zeros((9, 31)); z[:, 15] = 40.0
    h = sh.sun_hours(z, GT, LAT, LON, DAY - 43200, DAY + 43200, 1800.0)
    flat = sh.sun_hours(np.zeros((9, 31)), GT, LAT, LON, DAY - 43200, DAY + 43200, 1800.0)[0, 0]
    assert h[4, 14] < flat and h[4, 16] < flat          # west side loses the morning, east side the evening
    assert h[4, 15] == flat                              # the crest


def test_core_is_a_weighted_sum_of_shadow_maps():
    rng = np.random.default_rng(2)
    z = rng.integers(0, 4, (8, 9)).astype(float); z[2, 3] = np.nan
    pos = [(135.0, 20.0, 0.25), (200.0, 35.0, 1.0), (300.0, 8.0, 0.5), (10.0, 90.0, 2.0)]
    h = sh.sun_hours_at(z, GT, pos)
    want = np.zeros(z.shape)
    for az, el, w in pos[:3]:
        want += np.where(so.shadow(z, GT, az, el)[0], w, 0.0)
    want += np.where(np.isnan(z), 0.0, 2.0)
    want[np.isnan(z)] = np.nan
    assert np.array_equal(h, want, equal_nan=True)


def test_midpoint_samples_and_night():
    s = sh.samples(LAT, LON, DAY, DAY + 3600, 900.0)
    assert len(s) == 4 and all(w == 0.25 for *_, w in s)
    assert sh.samples(LAT, LON, DAY + 12 * 3600 - 900, DAY + 12 * 3600, 300.0) == []   # local midnight
    az0 = sun_position(LAT, LON, DAY + 450.0)[0]
    assert math.isclose(sh.samples(LAT, LON, DAY, DAY + 900, 900.0, grid_convergence=1.5)[0][0], az0 - 1.5)
