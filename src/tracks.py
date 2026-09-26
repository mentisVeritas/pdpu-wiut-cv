"""Build per-id trajectories from YOLO + ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DETECT_STRIDE
from .model import TRACK_CLASSES, VEHICLE_COCO, get_model, infer_device, reset_tracker


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


def extract_tracks(video_path: str, stride: int = DETECT_STRIDE) -> tuple[dict[int, Track], dict]:
    """Run detector+tracker over one video. Returns (tracks, video_meta)."""
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
    frame_idx = 0
    for result in results:
        t = frame_idx / fps
        frame_idx += max(1, stride)
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


def velocity(track: Track, t: float, window: float = 1.0) -> tuple[float, float]:
    pts = [o for o in track.obs if t - window <= o.t <= t]
    if len(pts) < 2:
        return 0.0, 0.0
    a, b = pts[0], pts[-1]
    dt = max(1e-3, b.t - a.t)
    return (b.cx - a.cx) / dt, (b.cy - a.cy) / dt


def speed_at(track: Track, t: float, window: float = 1.0) -> float:
    vx, vy = velocity(track, t, window)
    return float(np.hypot(vx, vy))
