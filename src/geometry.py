"""Scene layout helpers. Coordinates in scene.json are normalized to [0, 1]."""

from __future__ import annotations

import json
from pathlib import Path

_SCENE_PATH = Path(__file__).with_name("scene.json")


def load_scene() -> dict:
    data = json.loads(_SCENE_PATH.read_text())
    data.pop("comment", None)
    return data


def _to_px(poly: list[list[float]], width: int, height: int) -> list[tuple[float, float]]:
    return [(p[0] * width, p[1] * height) for p in poly]


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
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


def scene_polys(scene: dict, width: int, height: int) -> dict:
    crossings = scene.get("crossings") or []
    if not crossings and scene.get("crossing"):
        crossings = [scene["crossing"]]
    return {
        "roadway": _to_px(scene.get("roadway") or [], width, height),
        "crossings": [_to_px(p, width, height) for p in crossings if p],
        "crossing": _to_px(scene.get("crossing") or [], width, height),
        "stop_line": _to_px(scene.get("stop_line") or [], width, height),
    }


def on_any_crossing(x: float, y: float, crossings: list[list[tuple[float, float]]]) -> bool:
    return any(point_in_poly(x, y, poly) for poly in crossings if len(poly) >= 3)
