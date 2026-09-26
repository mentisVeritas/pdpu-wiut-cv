#!/usr/bin/env python3
"""Print fps / size / duration for every sample and dump preview frames."""

from __future__ import annotations

from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "notebooks" / "frames"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    videos = sorted(SAMPLES.glob("*.mp4")) + sorted(SAMPLES.glob("*.MP4"))
    if not videos:
        print(f"no .mp4 in {SAMPLES} yet — wait for samples/download.sh")
        return
    print(f"{'file':<20} {'WxH':>12} {'fps':>7} {'frames':>8} {'sec':>8} {'MB':>8}")
    for path in videos:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dur = n / fps if fps else 0.0
        mb = path.stat().st_size / 1e6
        print(f"{path.name:<20} {w}x{h:>6} {fps:7.2f} {n:8d} {dur:8.1f} {mb:8.1f}")
        for name, idx in (("first", 0), ("mid", max(0, n // 2)), ("last", max(0, n - 1))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                dest = OUT / f"{path.stem}_{name}.jpg"
                cv2.imwrite(str(dest), frame)
        cap.release()
    print(f"preview frames -> {OUT}")


if __name__ == "__main__":
    main()
