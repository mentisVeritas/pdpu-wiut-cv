"""Shared constants for the PDPU pipeline."""

from __future__ import annotations

SEED = 42

# Official labels. Shrink this list only if we never emit a class.
CLASSES: list[str] = [
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

# Frame sampling for Part A (every Nth frame). Part B still sees every frame
# from the harness; the estimator may skip internally.
DETECT_STRIDE = 3

# Segment post-process (seconds)
MIN_EVENT_SEC = 0.5
MERGE_GAP_SEC = 1.0

# Stopped-vehicle rule (the organisers' definition: stationary for 10 s or more)
STOPPED_SEC = 10.0

# Part B
RISK_HORIZON_SEC = 5.0
RISK_STRIDE = 5
