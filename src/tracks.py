"""Build per-id trajectories from YOLO + ByteTrack."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from .config import DETECT_STRIDE
from .model import TRACK_CLASSES, VEHICLE_COCO, get_model, infer_device, reset_tracker
from .signals import lamp_scores


@dataclass
class Obs:
    t: float
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def foot(self) -> tuple[float, float]:
        """Ground contact point. On this high oblique view the box centre sits
        well above the road, so scene regions are tested with the bottom edge."""
        return 0.5 * (self.x1 + self.x2), self.y2


@dataclass
class Track:
    tid: int
    cls: int
    obs: list[Obs] = field(default_factory=list)

    @property
    def is_vehicle(self) -> bool:
        return self.cls in VEHICLE_COCO

    @property
    def is_person(self) -> bool:
        return self.cls == 0


def extract_tracks(
    video_path: str,
    stride: int = DETECT_STRIDE,
    lights: dict[str, list] | None = None,
) -> tuple[dict[int, Track], dict]:
    """Run detector+tracker over one video. Returns (tracks, video_meta).

    lights: {light_id: [red_lamp_xy, green_lamp_xy]} in normalized coords; the
    lamp colours are sampled from the same decoded frames the tracker sees and
    returned as meta["light_samples"][light_id] = [(t, red, green), ...].
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    meta = {
        "fps": fps,
        "n_frames": n,
        "width": width,
        "height": height,
        "duration": n / fps if fps else 0.0,
    }

    reset_tracker()
    model = get_model()
    tracks: dict[int, Track] = {}
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        vid_stride=max(1, stride),
        classes=TRACK_CLASSES,
        verbose=False,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=infer_device(),
    )
    lights = lights or {}
    light_samples: dict[str, list] = {lid: [] for lid in lights}
    meta["light_samples"] = light_samples
    frame_idx = 0
    for result in results:
        t = frame_idx / fps
        frame_idx += max(1, stride)
        for lid, (red_p, green_p) in lights.items():
            red, green = lamp_scores(result.orig_img, red_p, green_p)
            light_samples[lid].append((t, red, green))
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        for tid, box, cls in zip(ids, xyxy, clss):
            obs = Obs(t=t, x1=box[0], y1=box[1], x2=box[2], y2=box[3], cls=int(cls))
            if tid not in tracks:
                tracks[tid] = Track(tid=int(tid), cls=int(cls))
            tracks[tid].obs.append(obs)
            tracks[tid].cls = int(cls)
    return tracks, meta


def _obs_times(track: Track) -> list[float]:
    ts = track.__dict__.get("_ts")
    if ts is None or len(ts) != len(track.obs):
        ts = [o.t for o in track.obs]
        track.__dict__["_ts"] = ts
    return ts


def obs_at(track: Track, t: float, tol: float = 0.35) -> Obs | None:
    ts = _obs_times(track)
    i = bisect.bisect_left(ts, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(ts) and abs(ts[j] - t) <= tol and (best is None or abs(ts[j] - t) < abs(best.t - t)):
            best = track.obs[j]
    return best


def velocity(track: Track, t: float, window: float = 1.0) -> tuple[float, float]:
    ts = _obs_times(track)
    lo = bisect.bisect_left(ts, t - window)
    hi = bisect.bisect_right(ts, t) - 1
    if hi - lo < 1:
        return 0.0, 0.0
    a, b = track.obs[lo], track.obs[hi]
    dt = max(1e-3, b.t - a.t)
    return (b.cx - a.cx) / dt, (b.cy - a.cy) / dt


def speed_at(track: Track, t: float, window: float = 1.0) -> float:
    vx, vy = velocity(track, t, window)
    return float(np.hypot(vx, vy))
