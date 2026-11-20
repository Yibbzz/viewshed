"""Compare a tool's mask with a baseline."""

import numpy as np

BANDS = [("<1 cm (ambiguous)", 0.0, 0.01), ("1-10 cm", 0.01, 0.1), ("10 cm-1 m", 0.1, 1.0),
         (">1 m", 1.0, np.inf)]


def edge_cells(visible, zone):
    """Cells within one cell (8-neighbour) of a cell with the other visibility."""
    v = np.where(zone, visible, False)
    p = np.pad(v, 1, mode="edge")
    z = np.pad(zone, 1, mode="constant")
    edge = np.zeros_like(v)
    rows, cols = v.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nb = p[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
            nz = z[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
            edge |= nz & (nb != v)
    return edge & zone


def compare(truth, mask, zone, clearance=None):
    """Error rates (fractions of the zone) and, with clearance, error counts by band."""
    n = int(zone.sum())
    hidden = truth & ~mask & zone        # wrongly hidden
    shown = ~truth & mask & zone         # wrongly visible
    edge = edge_cells(truth, zone)
    inner = zone & ~edge
    err = hidden | shown
    out = {
        "cells": n,
        "agree": 1 - err.sum() / n,
        "wrongly_hidden": hidden.sum() / n,
        "wrongly_visible": shown.sum() / n,
        "edge_error": err[edge].sum() / max(edge.sum(), 1),
        "interior_error": err[inner].sum() / max(inner.sum(), 1),
    }
    if clearance is not None:
        local = zone & ~truth & (clearance >= 0)   # hidden only by the target's own half cell
        out["band_errors"] = {"local slope": int((err & local).sum())}
        mag = np.abs(clearance)
        for name, lo, hi in BANDS:
            sel = zone & ~local & (mag >= lo) & (mag < hi)
            out["band_errors"][name] = int((err & sel).sum())
    return out
