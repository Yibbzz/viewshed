import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import datasets, oracle_rviewshed as orv  # noqa: E402
from conformance.adapters import Grass, Job  # noqa: E402

GT = (0.0, 1.0, 0.0, 31.0, 0.0, -1.0)
OBS = (15.5, 15.5)


def test_flat_is_visible_and_wall_casts_a_shadow():
    assert orv.viewshed(np.zeros((31, 31)), GT, OBS).all()
    z = np.zeros((31, 31)); z[10, :] = 5.0
    v = orv.viewshed(z, GT, OBS)
    assert v[10, 15] and v[25, 15] and not v[3, 15]


def test_raising_the_observer_never_hides_a_cell():
    z = datasets.rough()
    low = orv.viewshed(z, datasets.GT, datasets.OBSERVER, 1.6)
    high = orv.viewshed(z, datasets.GT, datasets.OBSERVER, 5.0)
    assert (high | ~low).all()


@pytest.mark.skipif(shutil.which("grass") is None, reason="GRASS GIS not installed")
def test_matches_grass_exactly():
    names = ["hills", "urban", "nodata_holes"]
    jobs = [Job(datasets.DATASETS[n](), datasets.GT, datasets.OBSERVER, 1.6, th)
            for n in names for th in (0.0, 1.5)]
    for job, (grass, _) in zip(jobs, Grass().run_many(jobs)):
        ours = orv.viewshed(job.dem, job.gt, job.observer, 1.6, job.target_height)
        valid = ~np.isnan(job.dem)
        assert (ours == grass)[valid].all()
