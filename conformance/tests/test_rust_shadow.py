"""The crate's exact cast-shadow maps (viewshed::shadow) must equal the shadow oracle on every cell.

Runs examples/raw_shadow with VIEWSHED_SHADOW_CHECK (strip / a-cut membership asserted). Direction
given as a vector (lattice-aligned rays) or as azimuth/elevation (incl. the cardinals, whose
vectors are exact by the shared rule in Sun::direction / sun_vector)."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformance import shadow_oracle as so  # noqa: E402
import test_oracle_fast as tof  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_BIN = None


def _bin():
    global _BIN
    if _BIN is None:
        cargo = str(Path.home() / ".cargo/bin/cargo") if (Path.home() / ".cargo/bin/cargo").exists() else shutil.which("cargo")
        if not cargo:
            return None
        r = subprocess.run([cargo, "build", "--release", "-q", "--examples"], cwd=REPO, capture_output=True, text=True)
        b = REPO / "target/release/examples/raw_shadow"
        _BIN = b if r.returncode == 0 and b.exists() else False
    return _BIN or None


def num(x):
    return repr(float(x))


def rust(z, gt, th, curv, sun):
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "z.f64"), Path(d, "o.u8")
        np.ascontiguousarray(z, "<f8").tofile(src)
        rows, cols = z.shape
        args = [str(_bin()), str(src), str(rows), str(cols), num(gt[0]), num(gt[1]), num(gt[3]), str(dst),
                num(th), "1" if curv else "0", "0.13", "6371008.8"]
        args += (["angles", num(sun[0]), num(sun[1])] if len(sun) == 2 else ["vec", *map(num, sun)])
        p = subprocess.run(args, check=True, capture_output=True, text=True,
                           env={**os.environ, "VIEWSHED_SHADOW_CHECK": "1"})
        vec = tuple(float(v) for v in p.stdout.split()[1:4])
        return np.fromfile(dst, np.uint8).reshape(rows, cols).astype(bool), vec


pytestmark = pytest.mark.skipif(_bin() is None, reason="Rust toolchain not available")

VECS = [(1.0, 0.0, 0.5), (0.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 0.25), (0.8, 0.6, 0.3),
        (0.6, -0.8, 2.0), (-0.7, -0.3, 0.05), (2.0, 1.0, 1.0)]
ANGLES = [(0.0, 10.0), (90.0, 10.0), (180.0, 30.0), (270.0, 5.0), (45.0, 15.0), (135.0, 3.0), (217.3, 20.0), (-30.0, 8.0)]


def small_cases():
    seen = set()
    for name, z, gt, obs, oh, th, md, cv in tof.CASES:
        key = (z.shape, float(np.nansum(z)), float(gt[1]))
        if key in seen or z.size > 400:
            continue
        seen.add(key)
        yield name, np.ascontiguousarray(z, float), gt, bool(cv.get("curvature", False))


def random_cases(n=12, seed=5):
    rng = np.random.default_rng(seed)
    for k in range(n):
        m = int(rng.integers(8, 24))
        z = rng.normal(0, rng.uniform(0.2, 3.0), (m, m)).cumsum(0)
        if k % 3 == 0:
            z = np.round(z)
        if k % 4 == 1:
            z[rng.random((m, m)) < 0.1] = np.nan
        res = [1.0, 10.0, 0.5][k % 3]
        gt = (500000.0 + 0.37, res, 0.0, 4000000.0 + 0.61, 0.0, -res)
        yield f"rand{k}", z, gt, bool(k % 2)


@pytest.mark.parametrize("case", list(small_cases()) + list(random_cases()), ids=lambda c: c[0])
def test_rust_shadow_equals_oracle(case):
    name, z, gt, curv = case
    for sun in VECS + ANGLES:
        for th in (0.0, 0.5):
            ours, vec = rust(z, gt, th, curv, sun)
            want, _ = so.shadow_vec(z, gt, vec, th, curv)
            if len(sun) == 2:
                assert vec == tuple(so.sun_vector(gt, *sun)), f"sun vector differs for {sun}: {vec}"
            assert (ours == want).all(), f"{name} sun {sun} th {th}: {np.argwhere(ours != want)[:4].tolist()}"


def test_cardinal_vectors_exact():
    gt = (0.0, 2.0, 0.0, 10.0, 0.0, -2.0)
    for az, (sa, ca) in ((0, (0.0, 1.0)), (90, (1.0, 0.0)), (180, (0.0, -1.0)), (270, (-1.0, 0.0)), (360, (0.0, 1.0)), (-90, (-1.0, 0.0))):
        dx, dy, dz = so.sun_vector(gt, az, 10.0)
        assert (dx, dy) == (sa, -ca)
