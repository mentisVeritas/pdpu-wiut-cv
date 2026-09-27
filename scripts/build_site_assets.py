#!/usr/bin/env python3
"""Build the website's per-video material from the real pipeline output.

For every sample video: an annotated preview (tracked boxes, the aligned scene,
the signal state, active events), a JSON with events / signal / object counts /
risk curve, a motion heat map, a flow-direction map, and one still per class.

    python scripts/build_site_assets.py --videos samples --preview samples/h264 \
        --pred predictions_samples.json --out website/assets
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.geometry import scene_px  # noqa: E402
from src.pipeline import analyze  # noqa: E402
from src.render import render_video  # noqa: E402
from src.rules import Ctx, _pedestrians, _vehicles, flow_field, FLOW_GRID  # noqa: E402
from src.scene_align import background_frame  # noqa: E402

GROUPS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def phase_stats(runs: list[list], duration: float) -> dict:
    """Mean length of complete phases. Green lasts from its first green sample to
    the next red; a single noisy sample or the blinking end does not split it."""
    phases: list[list] = []
    for s, e, st in runs:
        kind = "R" if st == "R" else ("G" if st == "G" or (phases and phases[-1][2] == "G") else None)
        if kind is None:
            continue
        if phases and phases[-1][2] == kind:
            phases[-1][1] = e
        else:
            phases.append([s, e, kind])
    out = {}
    for state in ("R", "G"):
        full = [e - s for s, e, st in phases[1:-1] if st == state]
        out[state] = round(float(np.mean(full)), 1) if full else None
    return out


def counts_per_second(tracks: dict, duration: float) -> dict[str, list[int]]:
    n = int(math.ceil(duration))
    seen: dict[str, list[set]] = {g: [set() for _ in range(n)] for g in GROUPS.values()}
    for tr in tracks.values():
        g = GROUPS.get(tr.cls)
        if g is None:
            continue
        for o in tr.obs:
            seen[g][min(n - 1, int(o.t))].add(tr.tid)
    return {g: [len(s) for s in secs] for g, secs in seen.items()}


def heat_image(bg: np.ndarray, points: list[tuple[float, float]], scale: float) -> np.ndarray:
    h, w = bg.shape[:2]
    acc = np.zeros((h, w), np.float32)
    for x, y in points:
        xi, yi = int(x * scale), int(y * scale)
        if 0 <= xi < w and 0 <= yi < h:
            acc[yi, xi] += 1.0
    acc = cv2.GaussianBlur(acc, (0, 0), 7)
    if acc.max() > 0:
        acc = np.log1p(acc / acc.max() * 50) / np.log1p(50)
    heat = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    base = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR) if bg.ndim == 2 else bg.copy()
    alpha = np.clip(acc * 1.4, 0, 0.85)[..., None]
    return (base * (1 - alpha) + heat * alpha).astype(np.uint8)


def flow_image(bg: np.ndarray, flow: dict, sx: float, sy: float, ctx: Ctx) -> np.ndarray:
    img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR) if bg.ndim == 2 else bg.copy()
    img = (img * 0.6).astype(np.uint8)
    gx, gy = FLOW_GRID
    cw, ch = ctx.scene.width / gx, ctx.scene.height / gy
    for (cx, cy), d in flow.items():
        x, y = (cx + 0.5) * cw * sx, (cy + 0.5) * ch * sy
        L = 0.42 * cw * sx
        # hue follows the direction so the two carriageways read as two colours
        hue = int((math.degrees(math.atan2(d[1], d[0])) % 360) / 2)
        col = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.arrowedLine(img, (int(x - d[0] * L), int(y - d[1] * L)), (int(x + d[0] * L), int(y + d[1] * L)),
                        col, 2, cv2.LINE_AA, tipLength=0.4)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="samples")
    ap.add_argument("--preview", default="samples/h264", help="same clips at lower resolution, for rendering")
    ap.add_argument("--pred", default="predictions_samples.json", help="harness output, for the risk curves")
    ap.add_argument("--out", default="website/assets")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = json.loads(Path(args.pred).read_text()) if Path(args.pred).is_file() else {"videos": {}}
    examples: dict = {}
    index = []
    for video in sorted(Path(args.videos).glob("*.mp4")):
        stem = video.stem
        preview = Path(args.preview) / video.name
        preview = preview if preview.is_file() else video
        print(f"[{stem}] analyzing", flush=True)
        events, info = analyze(str(video))
        meta, tracks, lights, scene = info["meta"], info["tracks"], info["lights"], info["scene"]
        ctx = Ctx(scene_px(scene, meta["width"], meta["height"]), lights, meta["duration"])
        vehicles, peds = _vehicles(tracks), _pedestrians(tracks, ctx)

        bg = background_frame(str(preview))
        k = bg.shape[1] / meta["width"]
        cv2.imwrite(str(out / f"{stem}_heat_vehicles.jpg"),
                    heat_image(bg, [o.foot for tr in vehicles for o in tr.obs], k), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(out / f"{stem}_heat_people.jpg"),
                    heat_image(bg, [o.foot for tr in peds for o in tr.obs], k), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(out / f"{stem}_flow.jpg"), flow_image(bg, flow_field(vehicles, ctx), k, k, ctx),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        runs = {lid: lt.runs() for lid, lt in lights.items()}
        risk = pred["videos"].get(video.name, {}).get("risk", [])
        stepr = max(1, int(round(meta["fps"] / 2)))
        by_class = defaultdict(int)
        for tr in tracks.values():
            if tr.cls in GROUPS:
                by_class[GROUPS[tr.cls]] += 1
        counts = counts_per_second(tracks, meta["duration"])
        summary = {
            "video": video.name,
            "width": meta["width"], "height": meta["height"], "fps": round(meta["fps"], 3),
            "duration": round(meta["duration"], 2),
            "brightness": int(bg.mean()),
            "align": info["align"],
            "runtime": pred.get("log", {}).get(video.name),
            "tracks_by_class": dict(by_class),
            "pedestrians_on_foot": len(peds),
            "peak_vehicles": int(max((sum(counts[g][s] for g in ("car", "bus", "truck", "motorcycle"))
                                      for s in range(len(counts["car"]))), default=0)),
            "signal_mean_sec": {lid: phase_stats(r, meta["duration"]) for lid, r in runs.items()},
            "events_by_class": {lab: sum(1 for e in events if e[2] == lab) for lab in sorted({e[2] for e in events})},
        }
        (out / f"{stem}.json").write_text(json.dumps({
            **summary,
            "events": [[round(s, 2), round(e, 2), lab] for s, e, lab in events],
            "signal": runs,
            "counts": counts,
            "risk": [[round(t, 2), round(v, 3)] for t, v in risk[::stepr]],
        }))
        index.append(summary)
        print(f"[{stem}] rendering preview", flush=True)
        render_video(preview, out / f"{stem}.mp4", tracks, events, lights, scene, meta["width"],
                     examples=examples, ex_dir=out)
    (out / "index.json").write_text(json.dumps({"videos": index, "examples": examples}, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
