"""Earth curvature and refraction: oracles against analytic answers and GRASS,
and the Rust sweep against the oracle (bit-for-bit model equality)."""

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import oracle, oracle_rviewshed as orv  # noqa: E402
from conformance.adapters import AIRY_A, Grass, Job, Rust  # noqa: E402

N = 41
GT = (0.0, 1.0, 0.0, float(N), 0.0, -1.0)
OBS = (N / 2, N / 2)
EYE = 1.6
# A tiny "earth" so curvature matters on a 41-cell raster: horizon sqrt(h / f) = 10.5 cells.
REFR = 0.13
RADIUS = (1 - REFR) * 10.5 ** 2 / (2 * EYE)


def horizon_case():
    d = np.hypot(*np.meshgrid(np.arange(N) + 0.5 - N / 2, np.arange(N) + 0.5 - N / 2))
    return np.zeros((N, N)), d


def test_bilinear_oracle_matches_the_analytic_horizon():
    # Flat ground: the gap along the sightline is (1 - t)(h - f D^2 t), so ground
    # at distance D is visible exactly when D <= sqrt(h / f).
    z, d = horizon_case()
    vis, _, _ = oracle.viewshed(z, GT, OBS, EYE, 0.0, curvature=True, refraction=REFR,
                                earth_radius=RADIUS)
    assert (vis == (d <= 10.5)).all()


def test_rviewshed_oracle_horizon_is_within_a_cell():
    z, d = horizon_case()
    vis = orv.viewshed(z, GT, OBS, EYE, 0.0, curvature=True, refraction=REFR, earth_radius=RADIUS)
    assert vis[d < 9.5].all() and not vis[d > 11.5].any()


def test_curvature_off_is_unchanged():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, (N, N)).cumsum(0)
    a = orv.viewshed(z, GT, OBS, EYE, 0.0)
    b = orv.viewshed(z, GT, OBS, EYE, 0.0, curvature=False, refraction=0.5, earth_radius=1.0)
    assert (a == b).all()


def curved_jobs(n_jobs=24, seed=11):
    rng = np.random.default_rng(seed)
    for k in range(n_jobs):
        n = int(rng.integers(7, 40))
        dem = rng.normal(0, rng.uniform(0.05, 3), (n, n)).cumsum(0)
        if k % 3 == 0:
            dem[rng.random((n, n)) < 0.1] = np.nan
        if k % 4 == 0:
            dem = np.round(dem)
        r0, c0 = rng.integers(0, n, 2)
        if np.isnan(dem[r0, c0]):
            dem[r0, c0] = 0.0
        yield Job(dem, (0.0, 1.0, 0.0, float(n), 0.0, -1.0), (c0 + 0.5, n - r0 - 0.5),
                  float(rng.uniform(0, 5)), float(rng.choice([0.0, 1.0])),
                  None if k % 5 else float(rng.uniform(3, n)), True,
                  float(rng.choice([0.0, 0.13, 1 / 7])), float(rng.uniform(5, 200)))


RUST = Rust("rviewshed")


@pytest.mark.skipif(not RUST.available(), reason="Rust toolchain not available")
@pytest.mark.parametrize("job", list(curved_jobs()))
def test_sweep_equals_oracle_with_curvature(job):
    theirs = orv.viewshed(job.dem, job.gt, job.observer, job.observer_height, job.target_height,
                          job.max_distance, **job.curve_kwargs())
    assert (RUST.run(job) == theirs).all()


@pytest.mark.skipif(shutil.which("grass") is None, reason="GRASS GIS not installed")
def test_oracle_matches_grass_with_curvature_and_refraction():
    # 500 m cells over near-flat ground, so the earth's bulge decides thousands of cells.
    # GRASS uses its location's ellipsoid semi-major axis (Airy 1830 for EPSG:27700).
    rng = np.random.default_rng(5)
    n, res = 121, 500.0
    gt = (0.0, res, 0.0, n * res, 0.0, -res)
    jobs = [Job(np.round(rng.normal(0, amp, (n, n)), 1), gt, (n * res / 2, n * res / 2), eye, th,
                None, True, 1 / 7, AIRY_A)
            for amp, eye, th in [(0.5, 30.0, 0.0), (2.0, 60.0, 0.0), (1.0, 40.0, 10.0)]]
    for job, (grass, _) in zip(jobs, Grass().run_many(jobs)):
        ours = orv.viewshed(job.dem, job.gt, job.observer, job.observer_height,
                            job.target_height, **job.curve_kwargs())
        flat = orv.viewshed(job.dem, job.gt, job.observer, job.observer_height, job.target_height)
        assert (flat != grass).sum() > 500          # curvature really matters here
        assert (ours == grass).all()


def test_parabolic_drop_is_within_a_millimetre_of_a_sphere_at_35_km():
    # The d^2 / 2R drop approximates R - sqrt(R^2 - d^2); the gap is about d^4 / (8 R^3).
    for d in (1e3, 1e4, 3.5e4):
        exact = 6371008.8 - math.sqrt(6371008.8 ** 2 - d * d)
        assert abs(exact - d * d / (2 * 6371008.8)) < 1e-3
