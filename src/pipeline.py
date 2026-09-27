"""Part A entry: align scene -> tracks + lamp colours -> rules -> merged segments."""

from __future__ import annotations

from .config import DETECT_STRIDE
from .geometry import load_scene, scene_px
from .model import set_seeds
from .postprocess import merge_segments
from .rules import Ctx, apply_rules
from .scene_align import align_scene
from .signals import LightTimeline
from .tracks import extract_tracks


def analyze(video_path: str) -> tuple[list[list], dict]:
    """Events plus the intermediate results the demo and the report draw on."""
    set_seeds()
    scene, align = align_scene(load_scene(), video_path)
    lamps = {lt["id"]: lt["points"] for lt in scene.get("traffic_lights", []) if len(lt.get("points", [])) == 2}
    tracks, meta = extract_tracks(video_path, stride=DETECT_STRIDE, lights=lamps)
    lights = {lid: LightTimeline(samples) for lid, samples in meta["light_samples"].items()}
    ctx = Ctx(scene_px(scene, meta["width"], meta["height"]), lights, meta["duration"])
    events = merge_segments(apply_rules(tracks, ctx))
    return events, {"meta": meta, "align": align, "lights": lights, "tracks": tracks, "scene": scene}


def run_detect(video_path: str) -> list[list]:
    return analyze(video_path)[0]
