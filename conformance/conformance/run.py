"""Run every available tool on the datasets and scenes; write a Markdown report.

    python -m conformance.run [--quick] [--out report.md] [--tools name,...]
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import datasets, oracle, oracle_rviewshed, scenes
from .adapters import Job, all_adapters
from .metrics import BANDS, compare

CACHE = Path(__file__).resolve().parents[1] / ".cache"


def cached_oracle(job):
    h = hashlib.sha256()
    h.update(oracle.VERSION.encode())
    h.update(np.ascontiguousarray(job.dem, float).tobytes())
    h.update(json.dumps([list(job.gt), list(job.observer), job.observer_height,
                         job.target_height, job.max_distance]).encode())
    f = CACHE / f"oracle-{h.hexdigest()[:24]}.npz"
    if f.exists():
        d = np.load(f)
        return d["visible"], d["clearance"], int(d["n_exact"])
    vis, clr, n_exact = oracle.viewshed(job.dem, job.gt, job.observer, job.observer_height,
                                        job.target_height, job.max_distance)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(f, visible=vis, clearance=clr, n_exact=n_exact)
    return vis, clr, n_exact


def zone_of(job, truth_valid):
    rows, cols = job.dem.shape
    xs = job.gt[0] + (np.arange(cols) + 0.5) * job.gt[1]
    ys = job.gt[3] + (np.arange(rows) + 0.5) * job.gt[5]
    gx, gy = np.meshgrid(xs, ys)
    d = np.hypot(gx - job.observer[0], gy - job.observer[1])
    zone = truth_valid & (d > 1.5 * abs(job.gt[1]))    # skip the observer's own cells
    if job.max_distance is not None:
        zone &= d < job.max_distance - abs(job.gt[1])
    return zone


def pct(x):
    return f"{100 * x:.2f}%"


def secs(t):
    return "n/a" if t is None else (f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--quick", action="store_true", help="coarser scenes, fewer cases")
    ap.add_argument("--tools", help="comma-separated tool names (default: all available)")
    a = ap.parse_args(argv)

    tools = [t for t in all_adapters() if not a.tools or t.name in a.tools.split(",")]
    tools = [t for t in tools if t.available()]
    lines = ["# Viewshed conformance report", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · {platform.platform()} · oracle `{oracle.VERSION}`", "",
             "## Tools", "", "| Tool | Cell rule | Notes |", "| --- | --- | --- |"]
    lines += [f"| {t.name} | {t.cell_rule} | {t.notes} |" for t in tools]

    # 1. Synthetic rasters against the exact bilinear oracle
    lines += ["", "## 1. Synthetic rasters vs the exact bilinear oracle", "",
              "61 × 61 cells at 1 m, observer 1.6 m at the centre. *Agree (r.viewshed model)* compares "
              "with the r.viewshed-model oracle; every other column is against the exact bilinear "
              "oracle. Rates are fractions of cells; "
              "band columns count wrong cells by the oracle's clearance (how far the sightline "
              "clears or misses the terrain). *local slope* = hidden only by the target's own half cell.", ""]
    names = ["flat_wall", "hills"] if a.quick else list(datasets.DATASETS)
    ths = (0.0,) if a.quick else datasets.TARGET_HEIGHTS
    jobs = [(n, th, Job(datasets.DATASETS[n](), datasets.GT, datasets.OBSERVER,
                        datasets.OBSERVER_HEIGHT, th)) for n in names for th in ths]
    truths = []
    for n, th, job in jobs:
        vis, clr, n_exact = cached_oracle(job)
        rv = oracle_rviewshed.viewshed(job.dem, job.gt, job.observer, job.observer_height,
                                       job.target_height, job.max_distance)
        truths.append((vis, clr, zone_of(job, ~np.isnan(job.dem)), n_exact, rv))
    results = {t.name: t.run_many([j for _, _, j in jobs]) for t in tools}
    band_names = ["local slope"] + [b[0] for b in BANDS]
    head = "| Tool | Agree (r.viewshed model) | Agree | Wrongly hidden | Wrongly visible | Edge err | Interior err | " + \
        " | ".join(band_names) + " | Time |"
    for k, (n, th, job) in enumerate(jobs):
        vis, clr, zone, n_exact, rv = truths[k]
        lines += ["", f"### {n}, target height {th} m", "",
                  f"{int(zone.sum())} cells, {pct(vis[zone].mean())} visible, "
                  f"{n_exact} decided in exact arithmetic.", "", head,
                  "|" + " --- |" * (8 + len(band_names))]
        for t in tools:
            mask, dt = results[t.name][k]
            if mask is None:
                lines.append(f"| {t.name} | not supported |" + " |" * (7 + len(band_names)))
                continue
            m = compare(vis, mask, zone, clr)
            agree_rv = (mask == rv)[zone].mean()
            lines.append(f"| {t.name} | {pct(agree_rv)} | {pct(m['agree'])} | {pct(m['wrongly_hidden'])} | "
                         f"{pct(m['wrongly_visible'])} | {pct(m['edge_error'])} | "
                         f"{pct(m['interior_error'])} | "
                         + " | ".join(str(m["band_errors"][b]) for b in band_names)
                         + f" | {secs(dt)} |")

    # 2. Analytic scenes: convergence with resolution
    res_list = (4.0, 2.0, 1.0) if a.quick else scenes.RESOLUTIONS
    lines += ["", "## 2. Analytic scenes: error vs cell size", "",
              "True visibility from closed-form geometry, centre rule. Cell = % of cells wrong "
              "(wrongly hidden / wrongly visible). A sound method's error shrinks as cells get finer.", ""]
    skip_slow = {"oracle-bilinear": 1.0, "oracle-rviewshed": 1.0, "viewshed-py-0.0.1": 1.0}   # smallest res they run at
    for sc in scenes.SCENES:
        for th in ((0.0,) if a.quick else scenes.SCENE_TARGET_HEIGHTS):
            sc.target_height = th
            sjobs, struths = [], []
            for res in res_list:
                dem, gt, obs = sc.raster(res)
                sjobs.append(Job(dem, gt, obs, sc.observer_height, th))
                struths.append(sc.truth(res))
            lines += ["", f"### {sc.name} ({sc.notes}), target height {th} m", "",
                      "| Tool | " + " | ".join(f"{r:g} m" for r in res_list) + " |",
                      "|" + " --- |" * (1 + len(res_list))]
            for t in tools:
                keep = [i for i, r in enumerate(res_list) if r >= skip_slow.get(t.name, 0)]
                outs = dict(zip(keep, t.run_many([sjobs[i] for i in keep])))
                cells = []
                for i, (job, truth) in enumerate(zip(sjobs, struths)):
                    mask = outs.get(i, (None, None))[0]
                    if mask is None:
                        cells.append("–")
                        continue
                    m = compare(truth, mask, zone_of(job, np.ones_like(truth)))
                    cells.append(f"{pct(1 - m['agree'])} ({pct(m['wrongly_hidden'])} / "
                                 f"{pct(m['wrongly_visible'])})")
                lines.append(f"| {t.name} | " + " | ".join(cells) + " |")
    grass = next((t for t in tools if t.name == "grass-r.viewshed"), None)
    if grass and grass.startup is not None:
        lines += ["", f"GRASS session start-up (paid once per batch, excluded above): "
                      f"{grass.startup:.2f} s."]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
