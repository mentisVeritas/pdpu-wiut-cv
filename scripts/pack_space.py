#!/usr/bin/env python3
"""Assemble the Hugging Face Space for the live demo and optionally upload it.

    python scripts/pack_space.py --out build/space
    python scripts/pack_space.py --out build/space --push <user>/pdpu-traffic-demo

--push needs a prior `huggingface-cli login` with a write token (done by the
account owner in their own terminal).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: PDPU Traffic Events
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
license: agpl-3.0
---

Live demo for team PDPU, WIUT Hackathon 2026 (computer vision track).
Upload a clip from the WIUT road camera; the app runs Part A (event segments)
and Part B (causal accident risk) and returns an annotated video, a timeline
and the raw `detect_events()` output.

Source: https://github.com/mentisVeritas/pdpu-wiut-cv
"""

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
ultralytics>=8.3.0
opencv-python-headless>=4.8
numpy>=1.24
lap>=0.5.12
matplotlib>=3.8
"""

PACKAGES = "ffmpeg\nlibgl1\nlibglib2.0-0\n"


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    (out / "tests" / "fixtures").mkdir(parents=True)
    shutil.copy(ROOT / "demo" / "app.py", out / "app.py")
    shutil.copytree(ROOT / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "weights" / "yolov8n.pt", out / "weights" / "yolov8n.pt")
    shutil.copy(ROOT / "tests" / "fixtures" / "mini_clip.mp4", out / "tests" / "fixtures" / "mini_clip.mp4")
    (out / "README.md").write_text(SPACE_README)
    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "packages.txt").write_text(PACKAGES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/space")
    ap.add_argument("--push", help="Space id, e.g. user/pdpu-traffic-demo")
    args = ap.parse_args()
    out = Path(args.out)
    build(out)
    print(f"Space folder ready: {out}")
    if args.push:
        from huggingface_hub import upload_folder

        upload_folder(repo_id=args.push, repo_type="space", folder_path=str(out),
                      commit_message="Update PDPU demo from the main repository")
        print(f"uploaded to https://huggingface.co/spaces/{args.push}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
