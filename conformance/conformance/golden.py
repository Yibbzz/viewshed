"""Golden outputs: fixed scenes whose exact results are recorded as SHA-256
hashes in golden.json. A speed change must leave every hash unchanged, so no
cell can change without the tests failing.

    python -m conformance.golden            # check (also run by tests/test_golden.py)
    python -m conformance.golden --update   # record, only after a change is verified
                                            # cell for cell against the oracles

The input DEM's hash is recorded too, so a change in how a scene is generated
(for example a new NumPy random stream) shows up as "input changed", not as a
wrong result.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import ztv_bench
from .adapters import EXE, REPO, num, Job, Rust
from .bench import surface

GOLDEN = Path(__file__).resolve().parents[1] / "golden.json"


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _ztv_scene(n, res, relief, holes):
    ztv_bench.RELIEF = relief
    dem = ztv_bench.landscape(n, res, rough=True)
    if holes:
        rng = np.random.default_rng(1)
        dem[rng.random(dem.shape) < 0.03] = np.nan
        dem[100:130, 40:90] = np.nan
    mid = n * res / 2
    tg = [(np.floor((mid + ((i % 2) - 0.5) * 120) / res) * res + res / 2,
           np.floor((mid + (i - 3.5) * 450) / res) * res + res / 2, 150.0 if i % 3 else 0.0)
          for i in range(8)]
    ok = lambda t: (0 <= int(n - t[1] / res) < n and 0 <= int(t[0] / res) < n  # noqa: E731
                    and not np.isnan(dem[int(n - t[1] / res), int(t[0] / res)]))
    return dem, (0.0, res, 0.0, n * res, 0.0, -res), [t for t in tg if ok(t)]


def _ztv(dem, gt, targets, eye, method, curvature):
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "z.f64"), Path(d, "o.u32")
        np.ascontiguousarray(dem, "<f8").tofile(src)
        spec = ";".join(f"{num(x)},{num(y)},{num(h)}" for x, y, h in targets)
        args = [str(REPO / f"target/release/examples/raw_ztv{EXE}"), str(src), *map(str, dem.shape),
                num(gt[0]), num(gt[1]), num(gt[3]), num(eye), "-1", method, spec, str(dst)]
        if curvature:
            args += ["1", "0.13", "6371008.8"]
        subprocess.run(args, check=True, capture_output=True)
        return np.fromfile(dst, "<u4")


def cases():
    """(name, input array, function computing the output)."""
    rust = Rust("rviewshed")
    rust.available()
    z = surface(2001, seed=41)   # seed 41 puts the observer on a rise: ~1.5-4% visible
    gt = (0.0, 1.0, 0.0, 2001.0, 0.0, -1.0)
    yield ("viewshed r1000 flat", z,
           lambda: rust.run(Job(z, gt, (1000.5, 1000.5), 1.6, 0.0, 1000.0)))
    yield ("viewshed r1000 target 1.5 curved", z,
           lambda: rust.run(Job(z, gt, (1000.5, 1000.5), 1.6, 1.5, 1000.0, True, 0.13, 6371008.8)))
    yield ("viewshed full raster curved", z,
           lambda: rust.run(Job(z, gt, (300.5, 1700.5), 10.0, 0.0, None, True, 1 / 7, 6377563.396)))
    for name, (n, res, relief, holes, curve, eye) in {
        "flat10": (401, 10.0, 1.0, False, False, 2.0),
        "rough50curv": (401, 50.0, 3.0, False, True, 2.0),
        "holes": (301, 10.0, 3.0, True, True, 1.6),
    }.items():
        dem, g, tg = _ztv_scene(n, res, relief, holes)
        for method in ("forward", "reciprocal"):
            yield (f"ztv {method} {name}", dem,
                   lambda dem=dem, g=g, tg=tg, eye=eye, m=method, c=curve: _ztv(dem, g, tg, eye, m, c))


def compute():
    return {name: {"input": _sha(inp), "output": _sha(fn())} for name, inp, fn in cases()}


def check():
    want = json.loads(GOLDEN.read_text())
    got = compute()
    problems = []
    for name, rec in want.items():
        if name not in got:
            problems.append(f"{name}: case missing")
        elif got[name]["input"] != rec["input"]:
            problems.append(f"{name}: input changed (scene generation differs; not a result change)")
        elif got[name]["output"] != rec["output"]:
            problems.append(f"{name}: OUTPUT CHANGED")
    return problems, sorted(set(got) - set(want))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args(argv)
    if a.update:
        GOLDEN.write_text(json.dumps(compute(), indent=2) + "\n")
        print(f"wrote {GOLDEN}")
        return
    problems, new = check()
    for p in problems:
        print(p)
    for n in new:
        print(f"{n}: not recorded yet (run --update once verified)")
    print("all golden outputs identical" if not problems else f"{len(problems)} problem(s)")


if __name__ == "__main__":
    main()
