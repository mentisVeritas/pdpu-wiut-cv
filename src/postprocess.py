"""Merge / drop event fragments so tIoU at 0.7 stays usable."""

from __future__ import annotations

from .config import MERGE_GAP_SEC, MIN_EVENT_SEC


def merge_segments(
    events: list[list],
    min_sec: float = MIN_EVENT_SEC,
    merge_gap: float = MERGE_GAP_SEC,
) -> list[list]:
    """events: [start, end, label]. Same-class fragments closer than merge_gap are joined."""
    by_label: dict[str, list[list]] = {}
    for start, end, label in events:
        by_label.setdefault(label, []).append([float(start), float(end), label])

    out: list[list] = []
    for label, segs in by_label.items():
        segs.sort(key=lambda x: x[0])
        merged: list[list] = []
        for seg in segs:
            if merged and seg[0] <= merged[-1][1] + merge_gap:
                merged[-1][1] = max(merged[-1][1], seg[1])
            else:
                merged.append(seg)
        for start, end, lab in merged:
            if end - start >= min_sec:
                out.append([start, end, lab])
    out.sort(key=lambda x: (x[0], x[2]))
    return out
