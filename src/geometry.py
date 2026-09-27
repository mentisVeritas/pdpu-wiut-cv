"""Scene layout. scene.json holds normalized [0, 1] points drawn on scene_ref.jpg;
scene_align moves them into each video's frame before they reach the rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCENE_PATH = Path(__file__).with_name("scene.json")

Point = tuple[float, float]
Poly = list[Point]


def load_scene() -> dict:
    data = json.loads(_SCENE_PATH.read_text())
    data.pop("comment", None)
    return data


def point_in_poly(x: float, y: float, poly: Poly) -> bool:
    if len(poly) < 3:
        return False
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def in_any(x: float, y: float, polys: list[Poly]) -> int:
    """Index of the first polygon containing (x, y), or -1."""
    for i, poly in enumerate(polys):
        if point_in_poly(x, y, poly):
            return i
    return -1


@dataclass
class Line:
    """Directed segment a->b with a unit normal pointing to its 'past' side."""

    a: np.ndarray
    b: np.ndarray
    normal: np.ndarray

    @classmethod
    def from_points(cls, a: Point, b: Point, before: Point | None = None) -> "Line":
        a_, b_ = np.asarray(a, float), np.asarray(b, float)
        d = b_ - a_
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        if before is not None and float(np.dot(np.asarray(before, float) - a_, n)) > 0:
            n = -n
        return cls(a_, b_, n)

    def project(self, p: Point) -> tuple[float, float]:
        """(position along a->b in [0, 1] when beside the segment, signed distance in px)."""
        d = self.b - self.a
        v = np.asarray(p, float) - self.a
        t = float(np.dot(v, d) / (np.dot(d, d) + 1e-9))
        return t, float(np.dot(v, self.normal))


@dataclass
class StopLine:
    line: Line
    light: str | None


@dataclass
class ScenePx:
    width: int
    height: int
    roadways: list[Poly] = field(default_factory=list)
    crossings: list[Poly] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    solid_lines: list[list[Line]] = field(default_factory=list)

    def on_road(self, p: Point) -> bool:
        return in_any(p[0], p[1], self.roadways) >= 0

    def crossing_at(self, p: Point) -> int:
        return in_any(p[0], p[1], self.crossings)


def scene_px(scene: dict, width: int, height: int) -> ScenePx:
    def px(points: list) -> list[Point]:
        return [(float(x) * width, float(y) * height) for x, y in points]

    out = ScenePx(width, height)
    out.roadways = [px(s["points"]) for s in scene.get("roadways", []) if len(s.get("points", [])) >= 3]
    out.crossings = [px(s["points"]) for s in scene.get("crossings", []) if len(s.get("points", [])) >= 3]
    for s in scene.get("stop_lines", []):
        pts = px(s.get("points", []))
        if len(pts) == 3:
            out.stop_lines.append(StopLine(Line.from_points(pts[0], pts[1], before=pts[2]), s.get("light")))
    for s in scene.get("solid_lines", []):
        pts = px(s.get("points", []))
        segs = [Line.from_points(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        if segs:
            out.solid_lines.append(segs)
    return out
