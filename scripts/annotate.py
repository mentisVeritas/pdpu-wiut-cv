#!/usr/bin/env python3
"""Quick keyboard labeler for sample videos.

Keys
  space     play / pause
  a / d     back / forward 1s
  A / D     back / forward 5s
  1-9,0     start an event of that class (see list on screen)
  e         end the open event at the current time
  u         undo last finished event
  w         write my_labels.json
  q         quit (writes first)

Output shape matches evaluate.py ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "my_labels.json"

CLASSES = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]
KEY_TO_CLASS = {ord(str(i % 10)): CLASSES[i] for i in range(min(10, len(CLASSES)))}
# 1..9 first nine, 0 -> 10th class (stop_line). Remaining via extra keys:
KEY_TO_CLASS[ord("z")] = "congestion"
KEY_TO_CLASS[ord("x")] = "road_obstacle"
KEY_TO_CLASS[ord("c")] = "fire_smoke"
KEY_TO_CLASS[ord("v")] = "illegal_turn"
KEY_TO_CLASS[ord("b")] = "solid_line_crossing"


def load_labels() -> dict:
    if OUT.is_file():
        return json.loads(OUT.read_text())
    return {}


def save_labels(data: dict) -> None:
    OUT.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT}")


def main() -> int:
    videos = sorted(SAMPLES.glob("*.mp4")) + sorted(SAMPLES.glob("*.MP4"))
    if not videos:
        print("no sample videos yet")
        return 1
    idx = 0
    if len(sys.argv) > 1:
        wanted = sys.argv[1]
        for i, p in enumerate(videos):
            if p.name == wanted or p.stem == wanted:
                idx = i
    path = videos[idx]
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n / fps if fps else 0.0
    labels = load_labels()
    entry = labels.setdefault(path.name, {"duration": duration, "fps": fps, "events": []})
    entry["duration"] = duration
    entry["fps"] = fps

    playing = False
    frame_i = 0
    open_ev: list | None = None  # [start, None, label]

    def show(frame):
        vis = frame.copy()
        t = frame_i / fps
        cv2.putText(vis, f"{path.name}  t={t:.2f}s / {duration:.1f}s", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        hint = "1-9/0 start  e end  z cong  x obst  c fire  w save  space play"
        cv2.putText(vis, hint, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        if open_ev:
            cv2.putText(vis, f"OPEN {open_ev[2]} from {open_ev[0]:.2f}", (16, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
        y = 130
        for ev in entry["events"][-8:]:
            cv2.putText(vis, f"{ev[2]} [{ev[0]:.1f}, {ev[1]:.1f}]", (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1)
            y += 22
        cv2.imshow("PDPU annotator", vis)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ok, frame = cap.read()
        if not ok:
            playing = False
            frame_i = max(0, n - 1)
            continue
        show(frame)
        delay = 1 if playing else 0
        key = cv2.waitKey(delay if playing else 0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            frame_i = min(n - 1, frame_i + int(fps))
        elif key == ord("D"):
            frame_i = min(n - 1, frame_i + int(5 * fps))
        elif key == ord("a"):
            frame_i = max(0, frame_i - int(fps))
        elif key == ord("A"):
            frame_i = max(0, frame_i - int(5 * fps))
        elif key == ord("e") and open_ev:
            end = frame_i / fps
            if end > open_ev[0]:
                entry["events"].append([round(open_ev[0], 3), round(end, 3), open_ev[2]])
            open_ev = None
        elif key == ord("u") and entry["events"]:
            entry["events"].pop()
        elif key == ord("w"):
            save_labels(labels)
        elif key in KEY_TO_CLASS:
            if open_ev:
                end = frame_i / fps
                if end > open_ev[0]:
                    entry["events"].append([round(open_ev[0], 3), round(end, 3), open_ev[2]])
            open_ev = [frame_i / fps, None, KEY_TO_CLASS[key]]
        if playing:
            frame_i = min(n - 1, frame_i + 1)
            if frame_i >= n - 1:
                playing = False

    cap.release()
    cv2.destroyAllWindows()
    save_labels(labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
