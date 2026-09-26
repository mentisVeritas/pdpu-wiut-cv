"""Lazy YOLO loader. Weights must live in weights/ for the offline run."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .config import SEED

ROOT = Path(__file__).resolve().parent.parent
WEIGHT_CANDIDATES = (
    ROOT / "weights" / "yolov8n.pt",
    ROOT / "weights" / "yolo11n.pt",
    Path("yolov8n.pt"),
)

VEHICLE_COCO = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
PERSON_COCO = {0}
TRACK_CLASSES = sorted(VEHICLE_COCO | PERSON_COCO)

_model = None


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def weight_path() -> Path:
    for path in WEIGHT_CANDIDATES:
        if path.is_file():
            return path
    return WEIGHT_CANDIDATES[0]


def get_model():
    global _model
    if _model is None:
        set_seeds()
        from ultralytics import YOLO

        path = weight_path()
        _model = YOLO(str(path))
    return _model


def reset_tracker() -> None:
    if _model is None:
        return
    _model.predictor = None
