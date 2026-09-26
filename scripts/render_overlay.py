#!/usr/bin/env python3
"""Burn event labels onto a video for the website results page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

COLORS = {
    "accident": (80, 80, 255),
    "near_miss": (0, 140, 255),
    "wrong_way": (220, 80, 220),
    "stopped_vehicle": (240, 200, 0),
    "jaywalking": (80, 220, 80),
    "congestion": (200, 160, 40),
}


def active(events, t: float) -> list[str]:
    return [lab for s, e, lab in events if s <= t <= e]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--events", required=True, help="predictions.json or a list file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sec", type=float, default=0.0)
    args = ap.parse_args()

    payload = json.loads(Path(args.events).read_text())
    name = Path(args.video).name
    if isinstance(payload, dict) and "videos" in payload:
        events = payload["videos"].get(name, {}).get("events", [])
    elif isinstance(payload, dict) and name in payload:
        events = payload[name].get("events", [])
    else:
        events = payload

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if args.max_sec and t > args.max_sec:
            break
        labels = active(events, t)
        cv2.putText(frame, f"{t:.1f}s", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        y = 72
        for lab in labels:
            color = COLORS.get(lab, (220, 220, 220))
            cv2.putText(frame, lab, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += 34
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
