"""Map the hand-drawn scene onto each video.

The scene in scene.json was drawn on one reference frame (scene_ref.jpg). The
sample clips are not pixel-identical: framing shifts by a few pixels between
recordings and sample_003 is slightly zoomed in. Every point of the scene is
therefore moved through a similarity transform (scale + rotation + shift)
estimated from SIFT matches between median background frames, which drops
moving cars and people from the matching.
"""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np

from .config import SEED

REF_PATH = Path(__file__).with_name("scene_ref.jpg")
WORK_W = 1280
MIN_INLIERS = 15


def background_frame(video_path: str, n: int = 7, work_w: int = WORK_W) -> np.ndarray | None:
    """Median of n frames spread over the video, grayscale, work_w wide."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n) if total else 0)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (work_w, round(h * work_w / w)), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def estimate_transform(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray | None:
    """2x3 similarity matrix mapping ref pixels to tgt pixels, or None."""
    cv2.setRNGSeed(SEED)
    sift = cv2.SIFT_create(nfeatures=5000)
    k1, d1 = sift.detectAndCompute(ref, None)
    k2, d2 = sift.detectAndCompute(tgt, None)
    if d1 is None or d2 is None or len(k1) < MIN_INLIERS or len(k2) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999
    )
    if M is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return None
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    shift = float(np.hypot(M[0, 2], M[1, 2]))
    if not (0.75 < scale < 1.33) or shift > 0.3 * ref.shape[1]:
        return None
    return M


def _map_points(points: list, M: np.ndarray, ref_wh: tuple[int, int], tgt_wh: tuple[int, int]) -> list:
    rw, rh = ref_wh
    tw, th = tgt_wh
    out = []
    for x, y in points:
        px, py = x * rw, y * rh
        qx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        qy = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        out.append([qx / tw, qy / th])
    return out


def transform_scene(scene: dict, M: np.ndarray, ref_wh: tuple[int, int], tgt_wh: tuple[int, int]) -> dict:
    moved = copy.deepcopy(scene)
    for key, shapes in moved.items():
        if not isinstance(shapes, list):
            continue
        for shape in shapes:
            if isinstance(shape, dict) and "points" in shape:
                shape["points"] = _map_points(shape["points"], M, ref_wh, tgt_wh)
    return moved


def align_scene(scene: dict, video_path: str) -> tuple[dict, dict]:
    """Scene with points moved into this video's frame, plus a small report."""
    ref = cv2.imread(str(REF_PATH), cv2.IMREAD_GRAYSCALE)
    tgt = background_frame(video_path)
    if ref is None or tgt is None:
        return scene, {"aligned": False, "reason": "no reference or no frames"}
    M = estimate_transform(ref, tgt)
    if M is None:
        return scene, {"aligned": False, "reason": "not enough matches"}
    moved = transform_scene(scene, M, (ref.shape[1], ref.shape[0]), (tgt.shape[1], tgt.shape[0]))
    return moved, {
        "aligned": True,
        "scale": round(float(np.hypot(M[0, 0], M[1, 0])), 4),
        "shift_px": [round(float(M[0, 2]), 1), round(float(M[1, 2]), 1)],
    }
