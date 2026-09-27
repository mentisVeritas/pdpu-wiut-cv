"""Event rules over tracks and the per-video aligned scene.

Learned: YOLO boxes and ByteTrack ids. Everything here is hand-written logic
that follows the organisers' start/end convention for each class.
Speeds are in px/s and scale with the frame height, so the same thresholds
hold for the 4K originals and smaller copies.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .config import STOPPED_SEC
from .geometry import ScenePx
from .signals import LightTimeline
from .tracks import Obs, Track, speed_at, velocity

CAR_LIKE = {2, 3, 5, 7}  # car, motorcycle, bus, truck
TWO_WHEEL = {1, 3}       # bicycle, motorcycle: their riders are detected as 'person' too


class Ctx:
    def __init__(self, scene: ScenePx, lights: dict[str, LightTimeline], duration: float):
        self.scene = scene
        self.lights = lights
        self.duration = duration
        h = scene.height
        self.stop_speed = max(2.0, 0.004 * h)  # below: stationary
        self.move_speed = 0.02 * h             # above: really driving
        self.fast_speed = 0.05 * h

    def signal_changes(self) -> list[float]:
        out = []
        for lt in self.lights.values():
            prev = None
            for t, s in zip(lt.times, lt.states):
                if prev is not None and (s == "G") != (prev == "G"):
                    out.append(t)
                prev = s
        return sorted(out)


def _times(track: Track) -> list[float]:
    return [o.t for o in track.obs]


def _runs(mask: list[bool], times: list[float]) -> list[tuple[float, float]]:
    """Maximal True runs -> (start, end) using sample times."""
    out: list[tuple[float, float]] = []
    start = last = None
    for flag, t in zip(mask, times):
        if flag:
            if start is None:
                start = t
            last = t
        elif start is not None:
            out.append((start, last))
            start = None
    if start is not None:
        out.append((start, last))
    return out


def _vehicles(tracks: dict[int, Track]) -> list[Track]:
    return [tr for tr in tracks.values() if tr.cls in CAR_LIKE and len(tr.obs) >= 3]


# --------------------------------------------------------------- pedestrians

def _pedestrians(tracks: dict[int, Track], ctx: Ctx) -> list[Track]:
    """Person tracks that are people on foot, not riders of a bike or motorcycle."""
    bikes: dict[float, list[Obs]] = defaultdict(list)
    for tr in tracks.values():
        if tr.cls in TWO_WHEEL:
            for o in tr.obs:
                bikes[round(o.t, 3)].append(o)

    def riding(o: Obs) -> bool:
        fx, fy = o.foot
        for b in bikes.get(round(o.t, 3), []):
            w, h = b.x2 - b.x1, b.y2 - b.y1
            if b.x1 - 0.3 * w <= fx <= b.x2 + 0.3 * w and b.y1 <= fy <= b.y2 + 0.3 * h:
                return True
        return False

    out = []
    for tr in tracks.values():
        if not tr.is_person or len(tr.obs) < 3:
            continue
        if sum(riding(o) for o in tr.obs) > 0.3 * len(tr.obs):
            continue
        if float(np.median([speed_at(tr, o.t) for o in tr.obs])) > 0.1 * ctx.scene.height:
            continue
        feet = np.array([o.foot for o in tr.obs])
        width = float(np.median([o.x2 - o.x1 for o in tr.obs]))
        if tr.obs[-1].t - tr.obs[0].t > 5.0 and float(np.hypot(*(feet.max(0) - feet.min(0)))) < 1.5 * width:
            continue  # never moved: a pole or signal head misread as a person
        out.append(tr)
    return out


def rule_jaywalking(peds: list[Track], ctx: Ctx) -> list[list]:
    events = []
    for tr in peds:
        times = _times(tr)
        mask = [ctx.scene.on_road(o.foot) and ctx.scene.crossing_at(o.foot) < 0 for o in tr.obs]
        for start, end in _runs(mask, times):
            if end - start < 1.0:
                continue
            run = [o for o in tr.obs if start <= o.t <= end]
            # "Steps onto / leaves the road" means walking. Measure how far the feet
            # actually got, not the summed path: box jitter on a static false
            # positive (a traffic-light head read as a person) adds up to a long path.
            feet = np.array([o.foot for o in run])
            spread = float(np.hypot(*(feet.max(axis=0) - feet.min(axis=0))))
            if spread < 2.5 * float(np.mean([o.x2 - o.x1 for o in run])):
                continue
            events.append([start, end, "jaywalking"])
    return events


def rule_failure_to_yield(vehicles: list[Track], peds: list[Track], ctx: Ctx) -> list[list]:
    walk_max = 0.05 * ctx.scene.height  # faster than a walk: a cyclist or scooter the detector called a person
    feet: dict[float, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for tr in peds:
        for o in tr.obs:
            ci = ctx.scene.crossing_at(o.foot)
            if ci >= 0 and speed_at(tr, o.t) < walk_max:
                feet[round(o.t, 3)].append((ci, o.foot))
    events = []
    for tr in vehicles:
        idx = [ctx.scene.crossing_at(o.foot) for o in tr.obs]
        i = 0
        while i < len(idx):
            if idx[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < len(idx) and idx[j + 1] == idx[i] and tr.obs[j + 1].t - tr.obs[j].t < 0.6:
                j += 1
            run = tr.obs[i:j + 1]
            if run[-1].t - run[0].t >= 0.3 and float(np.median([speed_at(tr, o.t) for o in run])) >= ctx.move_speed:
                for o in run:
                    vx, vy = velocity(tr, o.t)
                    reach = 1.5 * (o.x2 - o.x1)

                    def conflict(ci: int, pf: tuple[float, float]) -> bool:
                        dx, dy = pf[0] - o.foot[0], pf[1] - o.foot[1]
                        # close by and ahead of the car: it drives into the walker's path
                        return ci == idx[i] and math.hypot(dx, dy) < reach and dx * vx + dy * vy > 0

                    if any(conflict(ci, pf) for ci, pf in feet.get(round(o.t, 3), [])):
                        events.append([run[0].t, run[-1].t, "failure_to_yield"])
                        break
            i = j + 1
    return events


# ------------------------------------------------------------ stops, queues

def rule_stopped_and_congestion(vehicles: list[Track], ctx: Ctx) -> list[list]:
    events: list[list] = []
    if not vehicles:
        return events
    step = 0.5
    ts = [float(t) for t in np.arange(0.0, ctx.duration + 1e-6, step)]
    cong_mask = []
    for t in ts:
        speeds = [speed_at(tr, t) for tr in vehicles
                  if tr.obs[0].t <= t <= tr.obs[-1].t and ctx.scene.on_road(tr.obs[0].foot)]
        slow = sum(1 for s in speeds if s < 2 * ctx.stop_speed)
        cong_mask.append(len(speeds) >= 6 and slow / len(speeds) >= 0.7)

    def green_time(s: float, e: float) -> float:
        if not ctx.lights:
            return float("inf")
        return step * sum(1 for t in ts if s <= t <= e and any(lt.state_at(t) == "G" for lt in ctx.lights.values()))

    cong: list[tuple[float, float]] = []
    for start, end in _runs(cong_mask, ts):
        # A queue at red is normal; congestion is a queue that does not clear on green.
        if end - start >= 8.0 and green_time(start, end) >= 10.0:
            cong.append((start, min(end + step, ctx.duration)))
            events.append([start, min(end + step, ctx.duration), "congestion"])

    changes = ctx.signal_changes()
    for tr in vehicles:
        mask = [speed_at(tr, o.t) < ctx.stop_speed and ctx.scene.on_road(o.foot) for o in tr.obs]
        for start, end in _runs(mask, _times(tr)):
            if end - start < STOPPED_SEC:
                continue
            if any(min(e, end) - max(s, start) > 0.5 * (end - start) for s, e in cong):
                continue
            # Moved off within a few seconds of a signal change: it was waiting at a light.
            if any(start < c <= end + 1.0 and end - c < 8.0 for c in changes):
                continue
            if ctx.lights and sum(any(lt.state_at(o.t) == "R" for lt in ctx.lights.values())
                                  for o in tr.obs if start <= o.t <= end) > 0.7 * sum(
                                  1 for o in tr.obs if start <= o.t <= end):
                continue
            events.append([start, end, "stopped_vehicle"])
    return events


# ------------------------------------------------------------ signal rules

def rule_red_light_and_stop_line(vehicles: list[Track], ctx: Ctx) -> list[list]:
    events = []
    for sl in ctx.scene.stop_lines:
        lt = ctx.lights.get(sl.light or "")
        if lt is None:
            continue
        for tr in vehicles:
            proj = [sl.line.project(o.foot) for o in tr.obs]
            for i in range(1, len(tr.obs)):
                (u0, d0), (u1, d1) = proj[i - 1], proj[i]
                if not (-0.1 <= u0 <= 1.1 and -0.1 <= u1 <= 1.1) or not (d0 < 0 <= d1):
                    continue
                a, b = tr.obs[i - 1], tr.obs[i]
                if b.t - a.t > 0.6:
                    continue
                tc = a.t + (b.t - a.t) * (-d0) / (d1 - d0 + 1e-9)
                box_h = b.y2 - b.y1
                through = any(o.t <= tc + 4.0 and p[1] > box_h for o, p in zip(tr.obs[i:], proj[i:]))
                if through:
                    if lt.is_red(tc):
                        events.append([tc, min(tr.obs[-1].t, tc + 8.0), "red_light"])
                else:
                    ev = _stop_line_event(tr, i, proj, lt, ctx)
                    if ev:
                        events.append(ev)
                break
    return events


def _stop_line_event(tr: Track, i: int, proj: list, lt: LightTimeline, ctx: Ctx) -> list | None:
    obs = tr.obs[i:]
    mask = [speed_at(tr, o.t) < ctx.stop_speed and 0 < p[1] < (o.y2 - o.y1) and lt.state_at(o.t) == "R"
            for o, p in zip(obs, proj[i:])]
    for start, end in _runs(mask, [o.t for o in obs]):
        if end - start >= 1.5:
            green = lt.next_green_after(start)
            stop = green if green is not None and green <= end + 3.0 else end
            return [start, stop, "stop_line"]
    return None


def rule_solid_line_crossing(vehicles: list[Track], ctx: Ctx) -> list[list]:
    events = []
    for tr in vehicles:
        for segs in ctx.scene.solid_lines:
            samples = []
            for o in tr.obs:
                best = None
                for seg in segs:
                    u, d = seg.project(o.foot)
                    if 0.0 <= u <= 1.0 and (best is None or abs(d) < abs(best)):
                        best = d
                if best is not None:
                    samples.append((o.t, best, o.x2 - o.x1))
            for k in range(1, len(samples)):
                t0, d0, _ = samples[k - 1]
                t1, d1, w = samples[k]
                if (d0 < 0) == (d1 < 0) or t1 - t0 > 0.6:
                    continue
                before = [d for t, d, _ in samples if t0 - 0.6 <= t <= t0]
                after = [d for t, d, _ in samples if t1 <= t <= t1 + 0.6]
                if len(before) < 2 or len(after) < 2:
                    continue
                if any((d < 0) != (d0 < 0) for d in before) or any((d < 0) != (d1 < 0) for d in after):
                    continue
                if speed_at(tr, t1) < ctx.move_speed:
                    continue
                start, end = t0, t1
                for t, d, bw in reversed(samples[:k]):
                    if abs(d) > 0.5 * bw or t < t0 - 3.0:
                        break
                    start = t
                for t, d, bw in samples[k:]:
                    if abs(d) > 0.5 * bw or t > t1 + 3.0:
                        break
                    end = t
                events.append([start, max(end, start + 0.4), "solid_line_crossing"])
                break
    return events


# ------------------------------------------------------------ direction

def rule_wrong_way(vehicles: list[Track], ctx: Ctx, grid: tuple[int, int] = (24, 14)) -> list[list]:
    """Against the locally dominant direction. A single global 'main flow' is
    meaningless on a two-way road, so the direction is learned per grid cell."""
    w, h = ctx.scene.width, ctx.scene.height
    gx, gy = grid

    def cell(p: tuple[float, float]) -> tuple[int, int]:
        return min(gx - 1, max(0, int(p[0] / w * gx))), min(gy - 1, max(0, int(p[1] / h * gy)))

    acc: dict[tuple[int, int], list] = defaultdict(list)
    owners: dict[tuple[int, int], set] = defaultdict(set)
    for tr in vehicles:
        for o in tr.obs[::3]:
            vx, vy = velocity(tr, o.t, 1.0)
            sp = math.hypot(vx, vy)
            if sp >= ctx.move_speed:
                c = cell(o.foot)
                acc[c].append((vx / sp, vy / sp))
                owners[c].add(tr.tid)
    flow = {}
    for c, vecs in acc.items():
        if len(owners[c]) < 4 or len(vecs) < 8:
            continue
        m = np.mean(vecs, axis=0)
        r = float(np.hypot(*m))
        if r >= 0.6:
            flow[c] = m / r

    events = []
    for tr in vehicles:
        mask = []
        for o in tr.obs:
            vx, vy = velocity(tr, o.t, 1.5)
            sp = math.hypot(vx, vy)
            f = flow.get(cell(o.foot))
            mask.append(f is not None and sp >= ctx.move_speed and (vx * f[0] + vy * f[1]) / sp < -0.5
                        and ctx.scene.on_road(o.foot))
        for start, end in _runs(mask, _times(tr)):
            if end - start >= 2.0:
                events.append([start, end, "wrong_way"])
    return events


def apply_rules(tracks: dict[int, Track], ctx: Ctx) -> list[list]:
    vehicles = _vehicles(tracks)
    peds = _pedestrians(tracks, ctx)
    events: list[list] = []
    events += rule_stopped_and_congestion(vehicles, ctx)
    events += rule_wrong_way(vehicles, ctx)
    events += rule_jaywalking(peds, ctx)
    events += rule_failure_to_yield(vehicles, peds, ctx)
    events += rule_red_light_and_stop_line(vehicles, ctx)
    events += rule_solid_line_crossing(vehicles, ctx)
    return [[max(0.0, s), min(ctx.duration, e), lab] for s, e, lab in events if e > s]
