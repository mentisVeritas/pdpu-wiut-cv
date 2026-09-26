"""Part A entry: tracks → rules → merged segments."""

from __future__ import annotations

from .config import DETECT_STRIDE
from .model import set_seeds
from .postprocess import merge_segments
from .rules import apply_rules
from .tracks import extract_tracks


def run_detect(video_path: str) -> list[list]:
    set_seeds()
    tracks, meta = extract_tracks(video_path, stride=DETECT_STRIDE)
    events = apply_rules(tracks, meta)
    return merge_segments(events)
