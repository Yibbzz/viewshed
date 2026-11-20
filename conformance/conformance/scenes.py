"""Analytic scenes: continuous shapes on flat ground, with true visibility from
closed-form geometry. Rasterised at several resolutions to see whether each
tool's error shrinks as cells get finer.

World coordinates put the observer at (0, 0) on the ground. Each raster has an
odd number of cells per side with the observer at the centre of the middle
cell, so tools that snap the observer to a cell centre get the same point.

Truth for a cell is the centre rule: is the point target_height above the true
surface at the cell centre visible from the eye? Shapes are convex solids
(boxes, ridges, cones); a sightline is blocked when it enters the interior of
one. Touching a face, edge or the ground does not block.
"""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-7   # solids are shrunk by this much so touching never counts as blocking


@dataclass
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    h: float

    def height(self, x, y):
        inside = (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)
        return np.where(inside, self.h, 0.0)

    def halfspaces(self):   # rows (nx, ny, nz, d): n . p <= d
        return [(-1, 0, 0, -self.x0), (1, 0, 0, self.x1), (0, -1, 0, -self.y0),
                (0, 1, 0, self.y1), (0, 0, 1, self.h), (0, 0, -1, 1.0)]


@dataclass
class Ridge:
    """Triangular prism running north-south: crest at x = xc, half-width w."""
    xc: float
    w: float
    y0: float
    y1: float
    h: float

    def height(self, x, y):
        z = self.h * (1 - np.abs(x - self.xc) / self.w)
        return np.where((y >= self.y0) & (y <= self.y1), np.maximum(z, 0.0), 0.0)

    def halfspaces(self):
        k = self.h / self.w
        return [(k, 0, 1, self.h + k * self.xc), (-k, 0, 1, self.h - k * self.xc),
                (0, -1, 0, -self.y0), (0, 1, 0, self.y1), (0, 0, -1, 1.0)]


@dataclass
class Cone:
    cx: float
    cy: float
    r: float
    h: float

    def height(self, x, y):
        return np.maximum(self.h * (1 - np.hypot(x - self.cx, y - self.cy) / self.r), 0.0)


def _blocked_convex(shape, eye, pts):
    """Cyrus-Beck clip of the segments eye -> pts against a convex polyhedron."""
    d = pts - eye
    tmin = np.zeros(len(pts))
    tmax = np.ones(len(pts))
    for nx, ny, nz, dd in shape.halfspaces():
        n = np.array([nx, ny, nz], float)
        norm = np.linalg.norm(n)
        num = (dd - EPS * norm) - eye @ n
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        tmax = np.where(den > 0, np.minimum(tmax, t), tmax)
        tmin = np.where(den < 0, np.maximum(tmin, t), tmin)
        tmax = np.where((den == 0) & (num < 0), -1.0, tmax)
    return tmin < tmax


def _blocked_cone(cone, eye, pts, iters=100):
    """f(t) = cone surface - line height is concave, so golden-section finds its max."""
    d = pts - eye
    g = (np.sqrt(5) - 1) / 2
    a, b = np.zeros(len(pts)), np.ones(len(pts))

    def f(t):
        p = eye + t[:, None] * d
        return cone.h * (1 - np.hypot(p[:, 0] - cone.cx, p[:, 1] - cone.cy) / cone.r) - p[:, 2]

    c1, c2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(c1), f(c2)
    for _ in range(iters):
        left = f1 > f2
        b = np.where(left, c2, b)
        a = np.where(left, a, c1)
        c1n, c2n = b - g * (b - a), a + g * (b - a)
        f1, f2 = f(c1n), f(c2n)
        c1, c2 = c1n, c2n
    best = np.maximum(f(c1), f(c2))
    return best > EPS


@dataclass
class Scene:
    name: str
    shapes: list
    half_extent: float = 60.0
    observer_height: float = 1.6
    target_height: float = 0.0
    notes: str = ""

    def height(self, x, y):
        z = np.zeros(np.broadcast(x, y).shape)
        for s in self.shapes:
            z = np.maximum(z, s.height(x, y))
        return z

    def raster(self, res):
        """(dem, geotransform, observer (x, y)) at cell size `res`."""
        n = int(round(2 * self.half_extent / res)) | 1
        x0 = -n * res / 2
        gt = (x0, res, 0.0, -x0, 0.0, -res)
        xs = x0 + (np.arange(n) + 0.5) * res
        gx, gy = np.meshgrid(xs, -xs)
        return self.height(gx, gy), gt, (0.0, 0.0)

    def truth(self, res):
        dem, gt, obs = self.raster(res)
        n = dem.shape[0]
        xs = gt[0] + (np.arange(n) + 0.5) * res
        gx, gy = np.meshgrid(xs, -xs)
        pts = np.column_stack([gx.ravel(), gy.ravel(), dem.ravel() + self.target_height])
        eye = np.array([0.0, 0.0, float(self.height(0.0, 0.0)) + self.observer_height])
        blocked = np.zeros(len(pts), bool)
        for s in self.shapes:
            if isinstance(s, Cone):
                blocked |= _blocked_cone(s, eye, pts)
            else:
                blocked |= _blocked_convex(s, eye, pts)
        return ~blocked.reshape(dem.shape)


SCENES = [
    Scene("buildings", [Box(10, 18, -6, 6, 8), Box(-30, -22, 15, 25, 12), Box(5, 9, -40, -30, 4),
                        Box(-12, -8, -12, -8, 3)], notes="box buildings, 3-12 m"),
    Scene("cone_hill", [Cone(28, 5, 15, 10), Box(48, 52, -2, 12, 6)], notes="a cone hill with a building behind"),
    Scene("ridge", [Ridge(20, 8, -60, 60, 6), Ridge(-35, 12, -20, 60, 9)], notes="two ridges"),
    Scene("thin_wall", [Box(15, 15.3, -40, 40, 3), Box(-20, -10, -25.2, -25, 2.5)],
          notes="30 cm and 20 cm walls, thinner than most cells"),
]
SCENE_TARGET_HEIGHTS = (0.0, 1.5)
RESOLUTIONS = (4.0, 2.0, 1.0, 0.5)
