"""Wind-farm ZTV benchmark: python -m conformance.ztv_bench [--cells 801] [--res 10]

A synthetic landscape (rolling hills, valleys, a ridge) with a wind farm of
turbines on the ridge, blade tips `--tip` m above ground. Every cell is an
observer at `--eye` m. Times the forward method (every cell looks at every
tip: the definition) and the reciprocal shortcut (one viewshed from each tip),
and reports how far they agree. Curvature and refraction are off (not yet
implemented), which matters beyond a few km.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .adapters import REPO, Rust

BIN = REPO / "target/release/examples/raw_ztv"


RELIEF = 1.0   # scales the hills (not the ridge); set by --relief


def smooth_surface(x, y, mid):
    """The landscape without its roughness: a continuous function, so true
    visibility can be computed on it directly."""
    hills = 60 * np.sin(x / 1300) * np.cos(y / 1700) + 35 * np.sin((x + 2 * y) / 900)
    hills += 25 * np.sin(x / 310) * np.sin(y / 270)                 # smaller hills and valleys
    return RELIEF * hills + 90 * np.exp(-((x - mid) / 900) ** 2) + 100   # a ridge for the farm


def landscape(n, res, seed=3, rough=True):
    rng = np.random.default_rng(seed)
    c = (np.arange(n) + 0.5) * res
    x, y = np.meshgrid(c, c)
    z = smooth_surface(x, y, x.mean())
    if not rough:
        return z
    noise = rng.normal(0, 1, (n, n))
    for _ in range(8):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                 + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5
    return z + 6 * noise / noise.std()


def true_ztv(n, res, targets, eye, max_d, step=5.0):
    """Count of tips visible from every cell centre on the continuous surface,
    sampling each sightline at most every `step` m. Map y here is north-up
    like the raster: row r has centre y = (n - r - 0.5) * res."""
    c = (np.arange(n) + 0.5) * res
    gx, gy = np.meshgrid(c, (n - np.arange(n) - 0.5) * res)
    mid = n * res / 2
    # smooth_surface is defined on the raster's (column, row-from-top) grid used in landscape()
    surf = lambda x, y: smooth_surface(x, n * res - y, mid)   # noqa: E731
    counts = np.zeros((n, n), np.uint32)
    for tx, ty, h in targets:
        ox, oy = gx.ravel(), gy.ravel()
        z0 = surf(ox, oy) + eye
        z1 = surf(np.array(tx), np.array(ty)) + h
        dist = np.hypot(tx - ox, ty - oy)
        ok = np.ones(ox.shape, bool) if max_d is None else dist <= max_d
        steps = int(np.ceil(dist.max() / step))
        for k in range(1, steps):
            t = k / steps
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                break
            px = ox[idx] + t * (tx - ox[idx])
            py = oy[idx] + t * (ty - oy[idx])
            line = z0[idx] + t * (z1 - z0[idx])
            ok[idx[surf(px, py) > line]] = False
        counts += ok.reshape(n, n)
    return counts


def run(dem, gt, eye, method, targets, max_d):
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "z.f64"), Path(d, "o.u32")
        np.ascontiguousarray(dem, "<f8").tofile(src)
        rows, cols = dem.shape
        spec = ";".join(f"{x!r},{y!r},{h!r}" for x, y, h in targets)
        p = subprocess.run([str(BIN), str(src), str(rows), str(cols), repr(gt[0]), repr(gt[1]),
                            repr(gt[3]), repr(eye), repr(-1.0 if max_d is None else max_d),
                            method, spec, str(dst)], check=True, capture_output=True, text=True)
        return np.fromfile(dst, "<u4").reshape(rows, cols), float(p.stdout)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=801, help="raster width and height in cells")
    ap.add_argument("--res", type=float, default=10.0, help="cell size, m")
    ap.add_argument("--turbines", type=int, default=8)
    ap.add_argument("--tip", type=float, default=150.0, help="blade tip height, m")
    ap.add_argument("--eye", type=float, default=2.0, help="observer eye height, m")
    ap.add_argument("--max-distance", type=float, default=None)
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--relief", type=float, default=1.0, help="scale the hills by this")
    ap.add_argument("--truth", action="store_true",
                    help="smooth landscape, compare both methods with true visibility on it")
    a = ap.parse_args(argv)
    global RELIEF
    RELIEF = a.relief
    Rust("rviewshed").available()          # builds the examples
    n, res = a.cells, a.res
    dem = landscape(n, res, rough=not a.truth)
    gt = (0.0, res, 0.0, n * res, 0.0, -res)
    mid = n * res / 2
    targets = [(mid + ((i % 2) - 0.5) * 120 + res / 2,
                mid + (i - (a.turbines - 1) / 2) * 450 + res / 2, a.tip) for i in range(a.turbines)]
    targets = [(np.floor(x / res) * res + res / 2, np.floor(y / res) * res + res / 2, h)
               for x, y, h in targets]           # tips at cell centres, as the model places them
    print(f"{n} x {n} cells at {res:g} m ({n * res / 1000:g} km square), {a.turbines} turbines, "
          f"tip {a.tip:g} m, eye {a.eye:g} m, every cell an observer")
    rec, t_rec = run(dem, gt, a.eye, "reciprocal", targets, a.max_distance)
    print(f"reciprocal (one viewshed per turbine): {t_rec:7.2f} s")
    if a.skip_forward:
        return
    fwd, t_fwd = run(dem, gt, a.eye, "forward", targets, a.max_distance)
    print(f"forward (every cell looks at every tip): {t_fwd:7.2f} s")
    valid = ~np.isnan(dem)
    diff = (fwd != rec) & valid
    print(f"cells seeing at least one turbine: forward {np.mean(fwd[valid] > 0):.1%}, "
          f"reciprocal {np.mean(rec[valid] > 0):.1%}")
    print(f"cells whose count differs: {diff.sum()} of {valid.sum()} ({diff.mean():.3%}); "
          f"reciprocal sees more at {(rec > fwd).sum()}, fewer at {(rec < fwd).sum()}")
    any_f, any_r = fwd > 0, rec > 0
    print(f"cells that differ on 'any turbine visible': {(any_f != any_r).sum()} "
          f"({np.mean(any_f != any_r):.3%})")
    if a.truth:
        import time
        t0 = time.perf_counter()
        truth = true_ztv(n, res, targets, a.eye, a.max_distance)
        print(f"true ZTV on the continuous surface ({time.perf_counter() - t0:.0f} s in NumPy): "
              f"{np.mean(truth > 0):.1%} of cells see a turbine")
        for name, got in (("forward", fwd), ("reciprocal", rec)):
            wrong = got != truth
            print(f"  {name:10s} count wrong at {wrong.mean():.3%} of cells "
                  f"(too many {np.mean(got > truth):.3%}, too few {np.mean(got < truth):.3%}); "
                  f"'any visible' wrong at {np.mean((got > 0) != (truth > 0)):.3%}")


if __name__ == "__main__":
    main()
