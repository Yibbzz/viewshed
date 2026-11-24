"""The crate's sun_position / sun_hours / sun_hours_between against the oracles. The solar position
agrees with conformance/sunpos.py to 1e-9 deg (same float operations; not always bit-identical:
the compiler may merge sin/cos into one sincos call, an ulp apart). Sun hours equal the exact-shadow
oracle summed over the crate's own sample positions, on every cell (exact for those floats)."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conformance import sun_hours as sh  # noqa: E402
from conformance.sunpos import sun_position  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_BIN = None


def _bin():
    global _BIN
    if _BIN is None:
        cargo = str(Path.home() / ".cargo/bin/cargo") if (Path.home() / ".cargo/bin/cargo").exists() else shutil.which("cargo")
        if not cargo:
            return None
        r = subprocess.run([cargo, "build", "--release", "-q", "--examples"], cwd=REPO, capture_output=True, text=True)
        b = REPO / "target/release/examples/raw_sun_hours"
        _BIN = b if r.returncode == 0 and b.exists() else False
    return _BIN or None


pytestmark = pytest.mark.skipif(_bin() is None, reason="Rust toolchain not available")


def num(x):
    return repr(float(x))


def test_sun_position_matches_oracle():
    ref = json.loads((Path(__file__).parent / "spa_reference.json").read_text())["points"]
    rng = np.random.default_rng(9)
    pts = [(p["lat"], p["lon"], p["t"]) for p in ref]
    pts += [(float(rng.uniform(-90, 90)), float(rng.uniform(-400, 400)), float(rng.uniform(-3e9, 4e9))) for _ in range(2000)]
    pts += [(90.0, 0.0, 0.0), (-90.0, 0.0, 1e9), (0.0, 0.0, -86400.5), (45.0, 180.0, 1719921600.0)]
    inp = "".join(f"{num(a)} {num(b)} {num(c)}\n" for a, b, c in pts)
    out = subprocess.run([str(_bin()), "sunpos"], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")
    worst, ulp_off = 0.0, 0
    for (lat, lon, t), line in zip(pts, out):
        got = tuple(float(v) for v in line.split())
        want = sun_position(lat, lon, t)
        d = max(abs(g - w) for g, w in zip(got, want))
        if d > 180:       # azimuth wrap at 0/360
            d = abs(d - 360)
        worst = max(worst, d)
        ulp_off += got != want
    assert worst < 1e-9, worst
    assert ulp_off < len(pts) // 100, ulp_off


def rust_samples(lat, lon, start, end, step, conv):
    out = subprocess.run([str(_bin()), "samples", *map(num, (lat, lon, start, end, step, conv))], capture_output=True, text=True, check=True).stdout
    return [tuple(float(v) for v in line.split()) for line in out.strip().split("\n") if line]


def run(z, gt, tail, th=0.0, curv=False):
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "z.f64"), Path(d, "o.f64")
        np.ascontiguousarray(z, "<f8").tofile(src)
        rows, cols = z.shape
        subprocess.run([str(_bin()), str(src), str(rows), str(cols), num(gt[0]), num(gt[1]), num(gt[3]), str(dst),
                        num(th), "1" if curv else "0", "0.13", "6371008.8", *tail], check=True, capture_output=True,
                       env={**os.environ, "VIEWSHED_SHADOW_CHECK": "1"})
        return np.fromfile(dst, "<f8").reshape(rows, cols)


GT = (500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0)
DAY = 1719921600.0          # 2024-07-02 12:00 UTC


def rasters():
    rng = np.random.default_rng(4)
    z1 = rng.normal(0, 8, (14, 17)).cumsum(0)
    z2 = rng.integers(0, 5, (12, 12)).astype(float) * 10.0
    z2[rng.random(z2.shape) < 0.08] = np.nan
    z3 = np.zeros((9, 21)); z3[:, 10] = 40.0
    return [("rough", z1), ("ties", z2), ("wall", z3)]


@pytest.mark.parametrize("name,z", rasters(), ids=lambda v: v if isinstance(v, str) else "")
def test_sun_hours_between_equals_oracle(name, z):
    for lat, lon, conv, th, curv in ((47.0, 8.0, 0.0, 0.0, False), (-33.9, 151.2, 1.7, 0.5, True), (69.6, 18.9, -2.3, 0.0, False)):
        got = run(z, GT, ["between", num(lat), num(lon), num(DAY - 43200), num(DAY + 43200), "3600", num(conv)], th, curv)
        pos = rust_samples(lat, lon, DAY - 43200, DAY + 43200, 3600.0, conv)
        py = sh.samples(lat, lon, DAY - 43200, DAY + 43200, 3600.0, conv)
        assert len(pos) == len(py) and all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 and a[2] == b[2] for a, b in zip(pos, py))
        want = sh.sun_hours_at(z, GT, pos, target_height=th, curvature=curv)
        assert np.array_equal(got, want, equal_nan=True), (name, lat, np.argwhere(got != want)[:4].tolist())


def test_positions_core_zenith_night_weights():
    z = rasters()[1][1]
    pos = [(135.0, 20.0, 0.25), (200.0, 35.0, 1.0), (10.0, 90.0, 2.0), (300.0, -3.0, 5.0), (0.0, 45.0, 0.0)]
    got = run(z, GT, ["positions", *[num(v) for p in pos for v in p]])
    want = sh.sun_hours_at(z, GT, [p for p in pos if p[1] > 0])
    assert np.array_equal(got, want, equal_nan=True)


def test_samples_partial_step_and_errors_match():
    d = 1718928000.0
    for args in ((89.9, 0.0, d + 21600, d + 64800, 1500.0, 0.0), (47.0, 8.0, d, d + 86400 - 7, 3600.0, 1.2), (-60.0, 20.0, d, d + 100.0, 3600.0, 0.0)):
        rs = rust_samples(*args)
        py = sh.samples(*args)
        assert len(rs) == len(py) and all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 and a[2] == b[2] for a, b in zip(rs, py)), args
    for bad in ((91.0, 0.0, d, d + 10, 1.0, 0.0), (0.0, 0.0, d + 10, d, 1.0, 0.0), (0.0, 0.0, d, d + 10, 0.0, 0.0)):
        p = subprocess.run([str(_bin()), "samples", *map(num, bad)], capture_output=True, text=True)
        assert p.returncode != 0
