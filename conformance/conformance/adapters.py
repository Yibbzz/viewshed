"""One small adapter per viewshed tool: same inputs and heights, a bool mask out
on the input grid. Missing tools are skipped, not failed.

Every adapter takes jobs (dem, geotransform, observer (x, y), observer_height,
target_height, max_distance) and returns (mask, seconds) per job. GRASS runs
all jobs in one session so its start-up is paid once and reported separately.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NODATA = -9999.0


# Semi-major axis of the Airy 1830 ellipsoid (EPSG:27700), which GRASS uses as
# the earth radius in the temporary location the GRASS adapter runs in.
AIRY_A = 6377563.396


@dataclass
class Job:
    dem: np.ndarray
    gt: tuple
    observer: tuple
    observer_height: float
    target_height: float
    max_distance: float = None
    curvature: bool = False
    refraction: float = 0.0
    earth_radius: float = AIRY_A

    def curve_kwargs(self):
        return dict(curvature=self.curvature, refraction=self.refraction,
                    earth_radius=self.earth_radius)


class Adapter:
    name = ""
    cell_rule = "centre"
    notes = ""

    def available(self):
        return True

    def run(self, job):
        raise NotImplementedError

    def run_many(self, jobs):
        out = []
        for j in jobs:
            t = time.perf_counter()
            mask = self.run(j)
            out.append((mask, time.perf_counter() - t))
        return out


class Oracle(Adapter):
    name = "oracle-bilinear"
    notes = "exact line of sight on the bilinear model (the conformance oracle itself)"

    def run(self, job):
        from . import oracle
        return oracle.viewshed(job.dem, job.gt, job.observer, job.observer_height,
                               job.target_height, job.max_distance, **job.curve_kwargs())[0]


class OracleRviewshed(Adapter):
    name = "oracle-rviewshed"
    notes = "the r.viewshed model (three gradient samples per cell), checked cell by cell"

    def run(self, job):
        from . import oracle_rviewshed
        return oracle_rviewshed.viewshed(job.dem, job.gt, job.observer, job.observer_height,
                                         job.target_height, job.max_distance,
                                         **job.curve_kwargs())


class ReferencePy(Adapter):
    name = "viewshed-py-reference"
    notes = "viewshed.reference: pure NumPy bilinear reference, samples every 1/4 cell, flat earth"

    def available(self):
        try:
            _import_viewshed()
            return True
        except ImportError:
            return False

    def run(self, job):
        if job.curvature:
            return None         # the NumPy reference has no curvature
        return _import_viewshed().reference.viewshed(job.dem, job.gt, job.observer, job.observer_height,
                                           job.target_height, job.max_distance)


class PythonBindings(Adapter):
    name = "viewshed-py"
    notes = "the Python package (Rust core via PyO3), default r.viewshed model"

    def available(self):
        try:
            _import_viewshed()
            return True
        except ImportError:
            return False

    def run(self, job):
        return _import_viewshed().viewshed(
            job.dem, job.gt, job.observer, job.observer_height, job.target_height,
            job.max_distance, **job.curve_kwargs())


def _import_viewshed():
    try:
        import viewshed
    except ImportError:
        sys.path.insert(0, str(REPO / "python"))
        import viewshed
    return viewshed


class Rust(Adapter):
    """The Rust crate via examples/raw_viewshed; time is the computation only."""
    _bin = None

    def __init__(self, surface):
        self.surface = surface
        self.name = {"rviewshed": "viewshed-rs-sweep", "bilinear": "viewshed-rs-bilinear"}[surface]
        self.notes = {"rviewshed": "Rust radial sweep, r.viewshed model (the default)",
                      "bilinear": "Rust bilinear reference, samples every 1/4 cell"}[surface]

    def available(self):
        if Rust._bin is not None:
            return True
        cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
        if not Path(cargo).exists():
            return False
        r = subprocess.run([cargo, "build", "--release", "-q", "--example", "raw_viewshed"],
                           cwd=REPO, capture_output=True)
        b = REPO / "target/release/examples/raw_viewshed"
        if r.returncode == 0 and b.exists():
            Rust._bin = b
        return Rust._bin is not None

    def run(self, j):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "z.f64"), Path(d, "o.u8")
            np.ascontiguousarray(j.dem, "<f8").tofile(src)
            rows, cols = j.dem.shape
            md = -1.0 if j.max_distance is None else j.max_distance
            p = subprocess.run([str(self._bin), str(src), str(rows), str(cols), repr(j.gt[0]),
                                repr(j.gt[1]), repr(j.gt[3]), repr(j.observer[0]),
                                repr(j.observer[1]), repr(j.observer_height), str(dst),
                                repr(j.target_height), repr(md), self.surface,
                                "1" if j.curvature else "0", repr(j.refraction),
                                repr(j.earth_radius)],
                               check=True, capture_output=True, text=True)
            self.last_time = float(p.stdout)
            return np.fromfile(dst, np.uint8).reshape(rows, cols).astype(bool)

    def run_many(self, jobs):
        out = []
        for j in jobs:
            mask = self.run(j)
            out.append((mask, self.last_time))
        return out


def _mem_dataset(dem, gt):
    from osgeo import gdal, osr
    rows, cols = dem.shape
    ds = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(27700)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(NODATA)
    band.WriteArray(np.where(np.isnan(dem), NODATA, dem))
    return ds


class Gdal(Adapter):
    notes = "gdal.ViewshedGenerate, curvature off (cc=0)"

    def __init__(self, mode):
        self.mode = mode
        self.name = f"gdal-{mode.lower()}"

    def available(self):
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            return hasattr(gdal, "ViewshedGenerate")
        except ImportError:
            return False

    def run(self, job):
        from osgeo import gdal
        ds = _mem_dataset(job.dem, job.gt)
        out = gdal.ViewshedGenerate(
            ds.GetRasterBand(1), "MEM", "", [], job.observer[0], job.observer[1],
            job.observer_height, job.target_height, 1, 0, 0, 0,
            (1.0 - job.refraction) if job.curvature else 0.0,
            getattr(gdal, "GVM_" + self.mode), job.max_distance or 0.0)
        return out.GetRasterBand(1).ReadAsArray() == 1

    @property
    def version(self):
        from osgeo import gdal
        return gdal.__version__


class Grass(Adapter):
    name = "grass-r.viewshed"
    notes = "GRASS r.viewshed -b, one session for all jobs; time excludes start-up"
    startup = None

    def available(self):
        return shutil.which("grass") is not None

    def run_many(self, jobs):
        from osgeo import gdal
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lines = []
            for i, j in enumerate(jobs):
                tif = d / f"dem{i}.tif"
                gdal.Translate(str(tif), _mem_dataset(j.dem, j.gt), format="GTiff")
                md = j.max_distance if j.max_distance is not None else -1
                flags = "-b" + ("cr" if j.curvature else "")
                refr = f" refraction_coeff={j.refraction!r}" if j.curvature else ""
                lines += [
                    f"r.in.gdal -o input={tif} output=dem{i} --quiet --overwrite",
                    f"g.region raster=dem{i} --quiet",
                    f"s=$(date +%s.%N)",
                    f"r.viewshed {flags} input=dem{i} output=vs{i} coordinates={j.observer[0]!r},"
                    f"{j.observer[1]!r} observer_elevation={j.observer_height!r} "
                    f"target_elevation={j.target_height!r} max_distance={md}{refr} "
                    "--quiet --overwrite",
                    f"e=$(date +%s.%N); echo \"$s $e\" > {d}/t{i}",
                    f"r.out.gdal input=vs{i} output={d}/vs{i}.tif format=GTiff --quiet",
                ]
            (d / "run.sh").write_text("set -e\n" + "\n".join(lines) + "\n")
            env = dict(os.environ, GRASS_SKIP_MAPSET_OWNER_CHECK="1")
            t = time.perf_counter()
            subprocess.run(["grass", "--tmp-location", "EPSG:27700", "--exec", "bash",
                            str(d / "run.sh")], check=True, capture_output=True, env=env)
            total = time.perf_counter() - t
            out = []
            for i, j in enumerate(jobs):
                s, e = map(float, (d / f"t{i}").read_text().split())
                ds = gdal.Open(str(d / f"vs{i}.tif"))
                assert ds.GetGeoTransform() == tuple(j.gt), "GRASS changed the grid"
                out.append((ds.GetRasterBand(1).ReadAsArray() == 1, e - s))
            self.startup = total - sum(t for _, t in out)
            return out


class PhotoFootprint(Adapter):
    name = "photofootprint-sweep"
    cell_rule = "any"
    notes = "NumPy radial sweep; 'any part' rule, fixed 0.1 m target offset (ignores target height)"

    def _mod(self):
        path = os.environ.get("PHOTOFOOTPRINT", str(Path.home() / "photofootprint"))
        if path not in sys.path:
            sys.path.insert(0, path)
        from photofootprint.terrain import dem, visibility
        return dem, visibility

    def available(self):
        try:
            self._mod()
            return True
        except Exception:
            return False

    def run(self, job):
        dem_mod, vis = self._mod()
        win = dem_mod.Window(job.dem, tuple(job.gt), "")
        from photofootprint.terrain.raycast import sample
        ground = float(sample(win, job.observer[0], job.observer[1]))
        rows, cols = job.dem.shape
        reach = job.max_distance or np.hypot(rows, cols) * abs(job.gt[1])
        return vis.visible_cells(win, job.observer[0], job.observer[1],
                                 ground + job.observer_height, reach)


def all_adapters():
    return [Oracle(), OracleRviewshed(), PythonBindings(), ReferencePy(), Rust("rviewshed"), Rust("bilinear"),
            *[Gdal(m) for m in ("Edge", "Diagonal", "Min", "Max")],
            Grass(), PhotoFootprint()]
