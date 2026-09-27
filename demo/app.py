"""Live demo: upload a clip from the camera, get the events back, visualized.

    python demo/app.py        # local, http://127.0.0.1:7860

On the Hugging Face Space this file is app.py at the Space root; scripts/pack_space.sh
assembles that folder from this repository.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.pipeline import analyze  # noqa: E402
from src.render import EVENT_COLORS, render_video  # noqa: E402
from src.risk import CausalRiskEstimator  # noqa: E402

MAX_SEC = 180
EXAMPLE = ROOT / "tests" / "fixtures" / "mini_clip.mp4"
SIGNAL_HEX = {"R": "#e5484d", "G": "#30a46c", "Y": "#f5a524", "?": "#8b8d98"}


def _hex(bgr: tuple) -> str:
    return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])


def prepare(src: str, workdir: Path) -> Path:
    """First MAX_SEC seconds at most 1280 px wide: the detector runs at 640 px
    anyway, and a smaller file decodes far faster on a CPU-only Space."""
    out = workdir / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-t", str(MAX_SEC),
                    "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-an", str(out)], check=True)
    return out


def risk_curve(path: Path) -> list[tuple[float, float]]:
    """Part B exactly as the harness runs it: every frame, in order, nothing else."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    est = CausalRiskEstimator()
    est.reset({"video_id": path.name, "fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append((i / fps, float(est.step(frame, i / fps))))
        i += 1
    cap.release()
    return out


def timeline(events: list, duration: float, signal: dict, risk: list):
    labels = sorted({e[2] for e in events})
    rows = [f"signal {lid}" for lid in signal] + labels
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(11, 2.4 + 0.42 * len(rows)),
                                  gridspec_kw={"height_ratios": [max(1.5, 0.42 * len(rows)), 1.6]})
    for row, runs in enumerate(signal.values()):
        for t0, t1, s in runs:
            ax.barh(row, max(t1 - t0, 0.2), left=t0, height=0.6, color=SIGNAL_HEX[s])
    for s, e, lab in events:
        ax.barh(rows.index(lab), e - s, left=s, height=0.6, color=_hex(EVENT_COLORS.get(lab, (130, 130, 130))))
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlim(0, duration)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Part A: events (signal row: red / green / amber as read from the lamp)", fontsize=10, loc="left")
    axr.plot([t for t, _ in risk], [v for _, v in risk], color="#e5484d", lw=1)
    axr.axhline(0.5, ls="--", color="#8b8d98", lw=0.8)
    axr.set_ylim(0, 1)
    axr.set_ylabel("P(accident ≤ 5 s)")
    axr.set_xlabel("seconds")
    axr.set_title("Part B: causal risk, alarm threshold 0.5", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def run(video: str | None, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload an .mp4 first, or click the example below.")
    work = Path(tempfile.mkdtemp(prefix="pdpu_"))
    progress(0.02, desc="Preparing the clip")
    clip = prepare(video, work)
    progress(0.1, desc="Part A: aligning the scene, detecting, tracking, reading the traffic light")
    events, info = analyze(str(clip))
    progress(0.55, desc="Part B: accident risk, frame by frame")
    risk = risk_curve(clip)
    progress(0.8, desc="Rendering the annotated video")
    meta = info["meta"]
    annotated = work / "annotated.mp4"
    render_video(clip, annotated, info["tracks"], events, info["lights"], info["scene"], meta["width"], out_w=960)
    signal = {lid: lt.runs() for lid, lt in info["lights"].items()}
    rows = [[round(s, 2), round(e, 2), lab] for s, e, lab in events]
    align = info["align"]
    status = (f"**{len(rows)} events** in {meta['duration']:.0f} s. Scene aligned to this clip: "
              + (f"shift {align['shift_px']} px, scale {align['scale']}." if align.get("aligned")
                 else f"no ({align.get('reason')}), drawn scene used as is."))
    return str(annotated), timeline(events, meta["duration"], signal, risk), rows, json.dumps(rows, indent=1), status


with gr.Blocks(title="PDPU traffic events") as demo:
    gr.Markdown(
        "# PDPU · traffic events from the WIUT road camera\n"
        "Upload an **.mp4 from this camera** (any length is accepted; the first **3 minutes** are analysed, "
        "up to about 500 MB). It runs on a free CPU, so a 2-minute clip takes a few minutes. "
        "Part A returns `[start_sec, end_sec, label]`; Part B is the causal accident risk. "
        "Model: YOLOv8n + ByteTrack + scene rules. "
        "[Repository](https://github.com/mentisVeritas/pdpu-wiut-cv) · "
        "[Website](https://mentisveritas.github.io/pdpu-wiut-cv/)"
    )
    with gr.Row():
        inp = gr.Video(label="Your clip (.mp4)", sources=["upload"])
        out_video = gr.Video(label="Annotated: boxes, scene, signal, active events")
    btn = gr.Button("Detect events", variant="primary")
    status = gr.Markdown()
    plot = gr.Plot(label="Timeline and risk curve")
    with gr.Row():
        table = gr.Dataframe(headers=["start_sec", "end_sec", "label"], label="Events", interactive=False)
        raw = gr.Code(language="json", label="detect_events() output")
    if EXAMPLE.is_file():
        gr.Examples([[str(EXAMPLE)]], inputs=inp, label="No clip at hand? A 12 s example from the camera")
    btn.click(run, inp, [out_video, plot, table, raw, status])

if __name__ == "__main__":
    demo.queue(max_size=4).launch()
