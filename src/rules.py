"""Trajectory rules that do not require a painted scene (geometry is optional)."""

from __future__ import annotations

import numpy as np

from .config import STOPPED_SEC, STOPPED_SPEED_PX
from .geometry import load_scene, on_any_crossing, point_in_poly, scene_polys
from .tracks import Track, speed_at, velocity


def _times(track: Track) -> list[float]:
    return [o.t for o in track.obs]


def _runs(mask: list[bool], times: list[float]) -> list[tuple[float, float]]:
    """Maximal True runs -> (start, end) using sample times."""
    out: list[tuple[float, float]] = []
    start = None
    last = None
    for flag, t in zip(mask, times):
        if flag:
            if start is None:
                start = t
            last = t
        elif start is not None and last is not None:
            out.append((start, last))
            start = None
    if start is not None and last is not None:
        out.append((start, last))
    return out


def _dominant_flow(tracks: dict[int, Track]) -> tuple[float, float]:
    vecs = []
    for tr in tracks.values():
        if not tr.is_vehicle or len(tr.obs) < 4:
            continue
        a, b = tr.obs[0], tr.obs[-1]
        dt = max(1e-3, b.t - a.t)
        vecs.append(((b.cx - a.cx) / dt, (b.cy - a.cy) / dt))
    if not vecs:
        return 0.0, 0.0
    arr = np.asarray(vecs)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


def rule_stopped_and_congestion(tracks: dict[int, Track], duration: float) -> list[list]:
    events: list[list] = []
    vehicles = [tr for tr in tracks.values() if tr.is_vehicle and len(tr.obs) >= 3]
    if not vehicles:
        return events

    # sample 2 Hz
    step = 0.5
    ts = np.arange(0.0, duration + 1e-6, step)
    slow_counts = []
    total_counts = []
    for t in ts:
        speeds = [speed_at(tr, float(t)) for tr in vehicles if tr.obs[0].t <= t <= tr.obs[-1].t]
        total_counts.append(len(speeds))
        slow_counts.append(sum(1 for s in speeds if s < STOPPED_SPEED_PX))

    cong_mask = [
        tot >= 5 and slow / max(1, tot) >= 0.7
        for tot, slow in zip(total_counts, slow_counts)
    ]
    for start, end in _runs(cong_mask, [float(t) for t in ts]):
        if end - start >= 8.0:
            events.append([start, min(end + step, duration), "congestion"])

    cong_intervals = [(s, e) for s, e, lab in events if lab == "congestion"]

    def in_congestion(t0: float, t1: float) -> bool:
        for s, e in cong_intervals:
            inter = min(t1, e) - max(t0, s)
            if inter > 0.5 * (t1 - t0):
                return True
        return False

    for tr in vehicles:
        times = _times(tr)
        mask = [speed_at(tr, t) < STOPPED_SPEED_PX for t in times]
        for start, end in _runs(mask, times):
            if end - start >= STOPPED_SEC and not in_congestion(start, end):
                events.append([start, end, "stopped_vehicle"])
    return events


def rule_wrong_way(tracks: dict[int, Track]) -> list[list]:
    fx, fy = _dominant_flow(tracks)
    flow_norm = float(np.hypot(fx, fy))
    if flow_norm < 5.0:
        return []
    events = []
    for tr in tracks.values():
        if not tr.is_vehicle or len(tr.obs) < 6:
            continue
        times = _times(tr)
        mask = []
        for t in times:
            vx, vy = velocity(tr, t, window=1.5)
            speed = float(np.hypot(vx, vy))
            if speed < 8.0:
                mask.append(False)
                continue
            cos = (vx * fx + vy * fy) / (speed * flow_norm + 1e-6)
            mask.append(cos < -0.55)
        for start, end in _runs(mask, times):
            if end - start >= 1.5:
                events.append([start, end, "wrong_way"])
    return events


def rule_jaywalking(tracks: dict[int, Track], width: int, height: int) -> list[list]:
    scene = load_scene()
    polys = scene_polys(scene, width, height)
    roadway = polys["roadway"]
    crossings = polys["crossings"]
    if len(roadway) < 3:
        return []
    events = []
    for tr in tracks.values():
        if not tr.is_person or len(tr.obs) < 3:
            continue
        times = _times(tr)
        mask = []
        for o in tr.obs:
            on_road = point_in_poly(o.cx, o.cy, roadway)
            on_cross = on_any_crossing(o.cx, o.cy, crossings)
            mask.append(on_road and not on_cross)
        for start, end in _runs(mask, times):
            if end - start >= 0.8:
                events.append([start, end, "jaywalking"])
    return events


def _iou(a, b) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _obs_at(track: Track, t: float):
    best = None
    best_dt = 1e9
    for o in track.obs:
        dt = abs(o.t - t)
        if dt < best_dt:
            best, best_dt = o, dt
    if best is None or best_dt > 0.6:
        return None
    return best


def _settles(track_ids: set[int], id_to_track: dict[int, Track], after_t: float) -> bool:
    """Per the class definition, an accident ends when 'all involved objects
    stop moving or leave the frame'. A close pass where every vehicle keeps
    driving afterward is ordinary dense traffic, not a crash -- require the
    post-contact speed to actually drop before calling it an accident."""
    window = np.arange(after_t, after_t + 1.5, 0.2)
    for tid in track_ids:
        tr = id_to_track.get(tid)
        if tr is None or not tr.obs:
            continue
        in_frame = [w for w in window if tr.obs[0].t <= w <= tr.obs[-1].t]
        if not in_frame:
            continue  # track ended near the contact -> "leaves the frame"
        speeds = [speed_at(tr, float(w)) for w in in_frame]
        if float(np.median(speeds)) > STOPPED_SPEED_PX * 3:
            return False
    return True


def rule_near_miss_accident(tracks: dict[int, Track], duration: float) -> list[list]:
    vehicles = [tr for tr in tracks.values() if tr.is_vehicle and len(tr.obs) >= 3]
    if len(vehicles) < 2:
        return []
    id_to_track = {id(tr): tr for tr in vehicles}
    step = 0.2
    ts = np.arange(0.0, duration + 1e-6, step)
    acc_mask = [False] * len(ts)
    miss_mask = [False] * len(ts)
    acc_ids: list[set[int]] = [set() for _ in ts]
    for i, t in enumerate(ts):
        t = float(t)
        boxes = []
        for tr in vehicles:
            o = _obs_at(tr, t)
            if o is not None:
                boxes.append((tr, o, velocity(tr, t)))
        for j in range(len(boxes)):
            for k in range(j + 1, len(boxes)):
                tr_a, a, va = boxes[j]
                tr_b, b, vb = boxes[k]
                iou = _iou(a, b)
                dx, dy = a.cx - b.cx, a.cy - b.cy
                dist = float(np.hypot(dx, dy))
                rvx, rvy = va[0] - vb[0], va[1] - vb[1]
                closing = -(dx * rvx + dy * rvy) / (dist + 1e-6)
                avg_w = 0.5 * ((a.x2 - a.x1) + (b.x2 - b.x1))
                if iou >= 0.12 or dist < 0.25 * avg_w:
                    acc_mask[i] = True
                    acc_ids[i] |= {id(tr_a), id(tr_b)}
                elif dist < 0.9 * avg_w and closing > 12.0:
                    miss_mask[i] = True
    events = []
    times = [float(t) for t in ts]
    for start, end in _runs(acc_mask, times):
        if end - start < 0.4:
            continue
        ids_here: set[int] = set()
        for i, t in enumerate(times):
            if start <= t <= end:
                ids_here |= acc_ids[i]
        if _settles(ids_here, id_to_track, end):
            events.append([start, min(end + 0.4, duration), "accident"])
        else:
            for i, t in enumerate(times):
                if start <= t <= end:
                    miss_mask[i] = True
    for start, end in _runs(miss_mask, times):
        # drop near-miss that overlaps an accident of the same window
        if any(not (end < s or start > e) for s, e, lab in events if lab == "accident"):
            continue
        if end - start >= 0.4:
            events.append([start, end, "near_miss"])
    return events


def apply_rules(tracks: dict[int, Track], meta: dict) -> list[list]:
    duration = float(meta["duration"])
    width, height = int(meta["width"]), int(meta["height"])
    events: list[list] = []
    events += rule_stopped_and_congestion(tracks, duration)
    events += rule_wrong_way(tracks)
    events += rule_jaywalking(tracks, width, height)
    events += rule_near_miss_accident(tracks, duration)
    return events
