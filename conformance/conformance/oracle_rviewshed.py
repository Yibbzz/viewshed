"""Oracle for the surface model GRASS r.viewshed actually uses.

Written from a description of the method (Haverkort, Toma & Zhuang, and a
reading of the GRASS manual and source), not from GRASS's GPL code. It checks
every closer cell directly instead of sweeping, so it is slow but simple.

Model (square cells, no curvature):
- The observer V is the centre of the cell containing the observer point; the
  eye is that cell's value + observer_height. The V cell never blocks.
- Every other cell B with data has three sample points: its centre and the two
  corners that bound its angular extent from V (ENTER and EXIT). Corner height
  is the mean of the four cells meeting there, or B's own value if any of them
  is no-data or off the grid. Each sample has a gradient atan(dz / distance).
- Angles are counter-clockwise from east. B is active for a target direction
  theta when ENTER < theta < EXIT (strict); its gradient at theta is linearly
  interpolated in angle between ENTER-centre or centre-EXIT.
- A target T (its centre, + target_height) is hidden when an active cell whose
  centre is strictly closer than T's has an interpolated gradient strictly
  greater than T's gradient. Ties (equal gradient) are visible.

Known difference from GRASS itself: GRASS keys its search tree by distance
alone, so cells at exactly equal distance can interfere (see the conformance
README). This oracle implements the intended rule.
"""

import math

import numpy as np

VERSION = "rviewshed-2"   # 2: earth curvature and refraction

# (sign of d_row, sign of d_col) -> (ENTER (d_col, d_row), EXIT (d_col, d_row)), row grows south
_CORNERS = {
    (-1, -1): ((0.5, -0.5), (-0.5, 0.5)),
    (-1, 0): ((0.5, 0.5), (-0.5, 0.5)),
    (-1, 1): ((0.5, 0.5), (-0.5, -0.5)),
    (0, 1): ((-0.5, 0.5), (-0.5, -0.5)),
    (1, 1): ((-0.5, 0.5), (0.5, -0.5)),
    (1, 0): ((-0.5, -0.5), (0.5, -0.5)),
    (1, -1): ((-0.5, -0.5), (0.5, 0.5)),
    (0, -1): ((0.5, -0.5), (0.5, 0.5)),
}


def _angle(dc, dr):
    """Counter-clockwise from east, in [0, 2 pi); dr > 0 is south."""
    return np.mod(np.arctan2(-dr, dc), 2 * np.pi)


def _corner_height(z, r, c, oc, orow):
    rows, cols = z.shape
    sr, sc = int(np.sign(orow)), int(np.sign(oc))
    rr, cc = r + sr, c + sc
    if not (0 <= rr < rows and 0 <= cc < cols):
        return z[r, c]
    vals = (z[r, c], z[r, cc], z[rr, c], z[rr, cc])
    if any(math.isnan(v) for v in vals):
        return z[r, c]
    return sum(vals) / 4


def viewshed(z, transform, observer, observer_height=1.6, target_height=0.0, max_distance=None,
             curvature=False, refraction=0.13, earth_radius=6371008.8):
    """Visibility of every cell under the r.viewshed model. With `curvature`, each
    sample (centre or corner, after averaging) and the target are lowered by
    d^2 / (2 R) * (1 - refraction) at their own distance d from the observer."""
    z = np.asarray(z, dtype=float)
    rows, cols = z.shape
    res = abs(transform[1])
    if abs(abs(transform[5]) - res) > 0 or transform[2] or transform[4]:
        raise ValueError("needs a north-up raster with square cells")
    cv = int(math.floor((observer[0] - transform[0]) / transform[1]))
    rv = int(math.floor((observer[1] - transform[3]) / transform[5]))
    if not (0 <= rv < rows and 0 <= cv < cols) or math.isnan(z[rv, cv]):
        raise ValueError("observer outside the raster or on no-data")
    eye = z[rv, cv] + observer_height
    drop = (1.0 - refraction) / (2.0 * earth_radius) if curvature else 0.0

    rr, cc = np.nonzero(~np.isnan(z))
    keep = ~((rr == rv) & (cc == cv))
    rr, cc = rr[keep], cc[keep]
    dr, dc = rr - rv, cc - cv
    dist = np.sqrt((dr * dr + dc * dc).astype(float)) * res   # exact integer, one rounding
    if max_distance is not None:
        ok = dist <= max_distance
        rr, cc, dr, dc, dist = rr[ok], cc[ok], dr[ok], dc[ok], dist[ok]
    n = len(rr)
    a = np.empty((n, 3))
    g = np.empty((n, 3))
    zc = z[rr, cc]
    a[:, 1] = _angle(dc, dr)
    g[:, 1] = np.arctan((zc - dist * dist * drop - eye) / dist)
    for i in range(n):
        enter, exit_ = _CORNERS[(int(np.sign(dr[i])), int(np.sign(dc[i])))]
        for j, (oc, orow) in ((0, enter), (2, exit_)):
            pc, pr = dc[i] + oc, dr[i] + orow
            a[i, j] = _angle(pc, pr)
            h = _corner_height(z, rr[i], cc[i], oc, orow)
            d = math.sqrt(pc * pc + pr * pr) * res
            g[i, j] = math.atan((h - d * d * drop - eye) / d)

    order = np.argsort(dist, kind="stable")
    rr, cc, dist, a, g, zc = rr[order], cc[order], dist[order], a[order], g[order], zc[order]
    visible = np.zeros(z.shape, dtype=bool)
    for i in range(n):
        k = np.searchsorted(dist, dist[i], side="left")     # strictly closer cells
        theta = a[i, 1]
        gt = math.atan((zc[i] - dist[i] * dist[i] * drop + target_height - eye) / dist[i])
        if k == 0:
            visible[rr[i], cc[i]] = True
            continue
        rel = np.mod(a[:k] - theta + np.pi, 2 * np.pi) - np.pi   # angles relative to theta
        d0, d1, d2 = rel[:, 0], rel[:, 1], rel[:, 2]
        active = (d0 < 0) & (d2 > 0)
        if not active.any():
            visible[rr[i], cc[i]] = True
            continue
        d0, d1, d2, gb = d0[active], d1[active], d2[active], g[:k][active]
        with np.errstate(divide="ignore", invalid="ignore"):
            before = gb[:, 1] + (gb[:, 0] - gb[:, 1]) * d1 / (d1 - d0)    # theta < centre
            after = gb[:, 1] + (gb[:, 2] - gb[:, 1]) * (-d1) / (d2 - d1)  # theta > centre
        gi = np.where(d1 > 0, before, np.where(d1 < 0, after, gb[:, 1]))
        visible[rr[i], cc[i]] = not (gi > gt).any()
    visible[rv, cv] = True
    return visible
