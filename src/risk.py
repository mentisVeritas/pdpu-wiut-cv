"""Causal Part B: TTC from a light on-the-fly tracker. No video file, no Part A."""

from __future__ import annotations

import numpy as np

from .config import RISK_STRIDE, TTC_ALARM_SEC
from .model import VEHICLE_COCO, get_model, infer_device


class _SimpleTrack:
    def __init__(self, tid: int, cx: float, cy: float, t: float, w: float):
        self.tid = tid
        self.cx = cx
        self.cy = cy
        self.t = t
        self.w = w
        self.vx = 0.0
        self.vy = 0.0


class CausalRiskEstimator:
    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.last_score = 0.0
        self.i = 0
        self.tracks: dict[int, _SimpleTrack] = {}
        self._next_id = 1

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self.i % max(1, RISK_STRIDE) != 0:
            self.i += 1
            return self.last_score
        self.i += 1
        self._update(frame, t_sec)
        self.last_score = self._score()
        return self.last_score

    def _update(self, frame: np.ndarray, t: float) -> None:
        model = get_model()
        result = model.predict(
            frame,
            verbose=False,
            imgsz=640,
            conf=0.3,
            classes=list(VEHICLE_COCO),
            device=infer_device(),
        )[0]
        dets = []
        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().tolist():
                x1, y1, x2, y2 = box
                dets.append((0.5 * (x1 + x2), 0.5 * (y1 + y2), max(4.0, x2 - x1)))
        assigned: set[int] = set()
        used_det: set[int] = set()
        for di, (cx, cy, w) in enumerate(dets):
            best_id, best_d = None, 1e9
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                d = float(np.hypot(cx - tr.cx, cy - tr.cy))
                if d < best_d and d < 1.8 * max(w, tr.w):
                    best_d, best_id = d, tid
            if best_id is None:
                continue
            tr = self.tracks[best_id]
            dt = max(1e-3, t - tr.t)
            tr.vx = (cx - tr.cx) / dt
            tr.vy = (cy - tr.cy) / dt
            tr.cx, tr.cy, tr.t, tr.w = cx, cy, t, w
            assigned.add(best_id)
            used_det.add(di)
        for di, (cx, cy, w) in enumerate(dets):
            if di in used_det:
                continue
            self.tracks[self._next_id] = _SimpleTrack(self._next_id, cx, cy, t, w)
            self._next_id += 1
        stale = [tid for tid, tr in self.tracks.items() if t - tr.t > 1.5]
        for tid in stale:
            del self.tracks[tid]

    def _score(self) -> float:
        items = list(self.tracks.values())
        if len(items) < 2:
            return 0.0
        min_ttc = 1e9
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                dx, dy = a.cx - b.cx, a.cy - b.cy
                dist = float(np.hypot(dx, dy))
                rvx, rvy = a.vx - b.vx, a.vy - b.vy
                closing = -(dx * rvx + dy * rvy)
                rel2 = rvx * rvx + rvy * rvy
                if closing <= 0 or rel2 < 1e-6:
                    continue
                ttc = max(0.0, closing / rel2)  # -dot(p, v) / ||v||^2
                if 0.0 < ttc < min_ttc:
                    min_ttc = ttc
        if min_ttc >= 1e8:
            return 0.0
        # logistic: 0.5 when TTC == horizon (5 s)
        k = 1.2
        score = 1.0 / (1.0 + np.exp((min_ttc - TTC_ALARM_SEC) / k))
        return float(min(1.0, max(0.0, score)))


# harness-facing name lives in solution.py; this is the implementation
RiskEstimator = CausalRiskEstimator
