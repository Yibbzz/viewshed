"""Single-observer speed: python -m conformance.bench [--radius 1000] [--grass]

A smooth synthetic DSM of (2 * radius + 1)^2 cells, 1 m cells, observer at the
centre, max_distance = radius. Prints seconds per tool and checks the Rust
sweep against GRASS cell for cell when GRASS runs.
"""

import argparse
import time

import numpy as np

from . import exact
from .adapters import Gdal, Grass, Job, Rust


def surface(n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, n))
    for _ in range(6):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    c = np.arange(n) + 0.5
    x, y = np.meshgrid(c, c)
    return z / exact.std(z) * 3 + 20 * np.sin(x / 150) * np.cos(y / 200)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=1000)
    ap.add_argument("--grass", action="store_true", help="also time GRASS (slow)")
    a = ap.parse_args(argv)
    n = 2 * a.radius + 1
    job = Job(surface(n), (0.0, 1.0, 0.0, float(n), 0.0, -1.0), (n / 2, n / 2), 1.6, 0.0,
              float(a.radius))
    rs = Rust("rviewshed")
    rs.available()
    ours, t = rs.run_many([job])[0]
    print(f"{n} x {n} cells, radius {a.radius}, {ours.mean():.1%} visible")
    print(f"viewshed-rs-sweep  {t:7.3f} s")
    gd = Gdal("Edge")
    t0 = time.perf_counter()
    gd.run(job)
    print(f"gdal-edge          {time.perf_counter() - t0:7.3f} s")
    if a.grass:
        g = Grass()
        t0 = time.perf_counter()
        (theirs, tg), = g.run_many([job])
        total = time.perf_counter() - t0
        print(f"grass-r.viewshed   {tg:7.3f} s compute, {total:.3f} s with import and start-up")
        print(f"cells differing from GRASS: {int((theirs != ours).sum())}")


if __name__ == "__main__":
    main()
