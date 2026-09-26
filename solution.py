"""
solution.py — the ONLY file a team has to implement.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B (optional)

Keep the names and signatures exactly as they are. Everything else — models,
tracking, rules, helper modules under src/ — is up to you.

Labels must come from CLASSES. You may REMOVE classes you never predict;
do not add new ids.
"""
from __future__ import annotations

import numpy as np

from src.config import CLASSES as _SRC_CLASSES
from src.config import RISK_HORIZON_SEC
from src.pipeline import run_detect
from src.risk import CausalRiskEstimator

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = list(_SRC_CLASSES)


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection."""
    return run_detect(video_path)


class RiskEstimator(CausalRiskEstimator):
    """Part B — causal accident anticipation."""

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return float(super().step(frame, t_sec))
