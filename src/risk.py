"""Causal Part B: are two tracked vehicles on a collision course?

step() sees the frames in order and nothing else: no video file, no Part A.
A pair raises the risk only when their straight-line paths bring the boxes into
contact within the horizon and they close fast. Pairs whose boxes already
overlap (queued cars seen at this oblique angle) and passes that clear each
other are ignored, and a hazard must persist for about half a second.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import RISK_HORIZON_SEC, RISK_STRIDE
from .model import VEHICLE_COCO, get_model, infer_device

CAR_LIKE = sorted(VEHICLE_COCO - {1})  # bicycles are too small to track reliably at 640 px


class _Track:
    def __init__(self, tid: int, cx: float, cy: float, w: float, t: float):
        self.tid, self.cx, self.cy, self.w, self.t = tid, cx, cy, w, t
        self.vx = self.vy = 0.0
        self.age = 0


class CausalRiskEstimator:
    def reset(self, meta: dict) -> None:
        self.height = float(meta.get("height") or 1080)
        self.i = 0
        self.last_score = 0.0
        self.tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.recent: deque[float] = deque(maxlen=4)
        self.gaps: dict[tuple[int, int], deque] = {}

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self.i % max(1, RISK_STRIDE):
            self.i += 1
            return self.last_score
        self.i += 1
        self._update(frame, t_sec)
        self.recent.append(self._hazard())
        self.last_score = float(min(self.recent))
        return self.last_score

    def _update(self, frame: np.ndarray, t: float) -> None:
        result = get_model().predict(frame, verbose=False, imgsz=640, conf=0.3, classes=CAR_LIKE,
                                     device=infer_device())[0]
        dets = []
        if result.boxes is not None:
            for x1, y1, x2, y2 in result.boxes.xyxy.cpu().tolist():
                dets.append((0.5 * (x1 + x2), 0.5 * (y1 + y2), max(4.0, x2 - x1)))
        taken: set[int] = set()
        for cx, cy, w in sorted(dets, key=lambda d: -d[2]):
            best, best_d = None, 1e9
            for tid, tr in self.tracks.items():
                d = math.hypot(cx - tr.cx, cy - tr.cy)
                if tid not in taken and d < best_d and d < 0.8 * max(w, tr.w):
                    best, best_d = tid, d
            if best is None:
                self.tracks[self._next_id] = _Track(self._next_id, cx, cy, w, t)
                taken.add(self._next_id)
                self._next_id += 1
                continue
            tr = self.tracks[best]
            dt = max(1e-3, t - tr.t)
            tr.vx = 0.5 * tr.vx + 0.5 * (cx - tr.cx) / dt
            tr.vy = 0.5 * tr.vy + 0.5 * (cy - tr.cy) / dt
            tr.cx, tr.cy, tr.w, tr.t = cx, cy, w, t
            tr.age += 1
            taken.add(best)
        for tid in [tid for tid, tr in self.tracks.items() if t - tr.t > 1.0]:
            del self.tracks[tid]

    def _hazard(self) -> float:
        fast = 0.03 * self.height
        items = [tr for tr in self.tracks.values() if tr.age >= 2]
        live = {tr.tid for tr in items}
        self.gaps = {k: v for k, v in self.gaps.items() if k[0] in live and k[1] in live}
        worst = 0.0
        for i in range(len(items)):
            a = items[i]
            for b in items[i + 1:]:
                avg_w = 0.5 * (a.w + b.w)
                px, py = b.cx - a.cx, b.cy - a.cy
                gap = math.hypot(px, py)
                key = (a.tid, b.tid) if a.tid < b.tid else (b.tid, a.tid)
                hist = self.gaps.setdefault(key, deque(maxlen=4))
                hist.append(gap)
                if gap < 0.6 * avg_w or gap > 4.0 * avg_w:
                    continue
                # closing steadily over the last updates, not a one-frame jitter
                if len(hist) < 4 or any(later >= earlier for earlier, later in zip(hist, list(hist)[1:])) \
                        or hist[0] - hist[-1] < 0.3 * avg_w:
                    continue
                if max(math.hypot(a.vx, a.vy), math.hypot(b.vx, b.vy)) < fast:
                    continue
                vx, vy = b.vx - a.vx, b.vy - a.vy
                rel2 = vx * vx + vy * vy
                if rel2 < fast * fast:
                    continue
                t_star = -(px * vx + py * vy) / rel2
                if not 0.0 < t_star <= RISK_HORIZON_SEC:
                    continue
                if math.hypot(px + vx * t_star, py + vy * t_star) > 0.3 * avg_w:
                    continue
                worst = max(worst, 1.0 / (1.0 + math.exp((t_star - 1.5) / 0.4)))
        return worst
