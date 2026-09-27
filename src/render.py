"""Annotated previews for the website and the live demo (not used for scoring)."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .geometry import scene_px

BOX_COLORS = {0: (90, 220, 90), 1: (220, 220, 90), 2: (250, 180, 60), 3: (60, 200, 250), 5: (80, 120, 250), 7: (200, 110, 250)}
EVENT_COLORS = {
    "red_light": (60, 60, 240), "stop_line": (180, 120, 250), "solid_line_crossing": (250, 120, 230),
    "jaywalking": (250, 190, 60), "failure_to_yield": (240, 100, 100), "stopped_vehicle": (160, 200, 60),
    "congestion": (120, 110, 250), "wrong_way": (60, 200, 130),
}
SIGNAL_COLORS = {"R": (40, 40, 230), "G": (60, 200, 60), "Y": (40, 190, 240), "?": (150, 150, 150)}
SIGNAL_NAMES = {"R": "RED", "G": "GREEN", "Y": "AMBER", "?": "?"}


def draw_scene(frame: np.ndarray, sc) -> None:
    overlay = frame.copy()
    for poly in sc.crossings:
        cv2.fillPoly(overlay, [np.int32(poly)], (235, 235, 235))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for poly in sc.roadways:
        cv2.polylines(frame, [np.int32(poly)], True, (230, 190, 70), 1, cv2.LINE_AA)
    for sl in sc.stop_lines:
        cv2.line(frame, tuple(np.int32(sl.line.a)), tuple(np.int32(sl.line.b)), (50, 50, 235), 3, cv2.LINE_AA)
    for segs in sc.solid_lines:
        for seg in segs:
            cv2.line(frame, tuple(np.int32(seg.a)), tuple(np.int32(seg.b)), (240, 110, 230), 2, cv2.LINE_AA)


def chip(frame: np.ndarray, text: str, x: int, y: int, color: tuple) -> int:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 12, y + 4), color, -1)
    cv2.putText(frame, text, (x + 6, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return x + tw + 18


def render_video(preview: Path, out_mp4: Path, tracks: dict, events: list, lights: dict, scene: dict,
                 src_w: int, out_w: int = 854, step: int = 2,
                 examples: dict | None = None, ex_dir: Path | None = None) -> None:
    """Burn boxes, the aligned scene, the signal state and active events into
    an H.264 mp4 that browsers play. With `examples`, the first frame of each
    class is also saved as a still (example_<class>.jpg in ex_dir)."""
    cap = cv2.VideoCapture(str(preview))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pw, ph = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_h = int(round(out_w * ph / pw / 2) * 2)
    k = out_w / src_w
    sc = scene_px(scene, out_w, out_h)
    by_bin: dict[int, list] = defaultdict(list)
    for tr in tracks.values():
        for o in tr.obs:
            by_bin[int(round(o.t * 10))].append((tr.tid, tr.cls, o))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
         "-r", f"{fps / step:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)], stdin=subprocess.PIPE)
    i = 0
    while cap.grab():
        if i % step:
            i += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = i / fps
        i += 1
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        draw_scene(frame, sc)
        best: dict[int, tuple] = {}
        for b in (int(round(t * 10)) - 1, int(round(t * 10)), int(round(t * 10)) + 1):
            for tid, cls, o in by_bin.get(b, []):
                if tid not in best or abs(o.t - t) < abs(best[tid][1].t - t):
                    best[tid] = (cls, o)
        for cls, o in best.values():
            cv2.rectangle(frame, (int(o.x1 * k), int(o.y1 * k)), (int(o.x2 * k), int(o.y2 * k)),
                          BOX_COLORS.get(cls, (200, 200, 200)), 1, cv2.LINE_AA)
        x = 10
        for lid, lt in lights.items():
            s = lt.state_at(t)
            x = chip(frame, f"{lid} {SIGNAL_NAMES[s]}", x, 24, SIGNAL_COLORS[s])
        chip(frame, f"{t:6.1f}s", out_w - 80, 24, (40, 40, 40))
        x = 10
        for s0, e0, lab in events:
            if s0 <= t <= e0:
                x = chip(frame, lab, x, out_h - 12, EVENT_COLORS.get(lab, (90, 90, 90)))
                if examples is not None and lab not in examples and t >= s0 + 0.4 * (e0 - s0):
                    cv2.imwrite(str(ex_dir / f"example_{lab}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    examples[lab] = {"video": out_mp4.stem, "t": round(t, 1), "start": round(s0, 2), "end": round(e0, 2)}
        ff.stdin.write(frame.tobytes())
    cap.release()
    ff.stdin.close()
    ff.wait()
