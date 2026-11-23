"""Polygon helpers for table and staff zones."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon


class ZoneShape:
    """A zone polygon plus an optional grown copy (the area staff stand in to serve)."""

    def __init__(self, polygon: list[list[int]], margin: float = 0.0):
        self.points = np.array(polygon, dtype=np.int32)
        self.poly = Polygon(polygon)
        if not self.poly.is_valid:
            self.poly = self.poly.buffer(0)
        self.grown = self.poly.buffer(margin) if margin > 0 else self.poly

    def contains(self, x: float, y: float) -> bool:
        return self.poly.contains(Point(x, y))

    def near(self, x: float, y: float) -> bool:
        return self.grown.contains(Point(x, y))

    def grown_points(self) -> np.ndarray:
        geom = self.grown
        if geom.geom_type != "Polygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        return np.array(geom.exterior.coords, dtype=np.int32)


def anchor_points(xyxy: np.ndarray, ratio: float) -> np.ndarray:
    """Horizontal centre and a point `ratio` of the way down each box (near the seat when sitting)."""
    if len(xyxy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = xyxy[:, 1] + (xyxy[:, 3] - xyxy[:, 1]) * ratio
    return np.stack([cx, cy], axis=1).astype(np.float32)
