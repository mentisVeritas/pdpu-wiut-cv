"""Traffic-light state from lamp colour at fixed (per-video aligned) pixel spots.

Brightness does not work in daylight: the sunlit housing is brighter than a
lit lamp. Colour does: a lit red lamp has R well above G and B, a lit green
(teal) lamp has G well above R, an unlit lamp is grey.
"""

from __future__ import annotations

import bisect

import numpy as np

LIT_SCORE = 35
YELLOW_SEC = 4.0


def lamp_scores(frame: np.ndarray, red_p: list[float], green_p: list[float]) -> tuple[int, int]:
    """(redness at the red lamp, greenness at the green lamp) for one BGR frame."""
    h, w = frame.shape[:2]
    r = max(2, round(w * 3 / 1280))

    def patch(p: list[float]) -> np.ndarray:
        x, y = int(round(p[0] * w)), int(round(p[1] * h))
        return frame[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1].reshape(-1, 3).astype(np.int32)

    pr, pg = patch(red_p), patch(green_p)
    if not len(pr) or not len(pg):
        return 0, 0
    red = int((pr[:, 2] - np.maximum(pr[:, 1], pr[:, 0])).max())
    green = int((pg[:, 1] - pg[:, 2]).max())
    return red, green


class LightTimeline:
    """State per sample: 'R', 'G', 'Y' (just after green: amber or blinking) or '?'.

    A washed-out red lamp reads as '?': once green has been off for longer than
    an amber phase, the light is red whatever the red lamp's colour says.
    """

    def __init__(self, samples: list[tuple[float, int, int]]):
        self.times: list[float] = []
        self.states: list[str] = []
        last_green = None
        last_definite = None
        for t, red, green in sorted(samples):
            if green >= LIT_SCORE:
                s = "G"
                last_green = t
            elif red >= LIT_SCORE:
                s = "R"
            elif last_green is not None and t - last_green <= YELLOW_SEC:
                s = "Y"
            elif last_green is not None or last_definite == "R":
                s = "R"
            else:
                s = "?"
            if s in ("R", "G"):
                last_definite = s
            self.times.append(t)
            self.states.append(s)
        # Before the first definite reading: green is always preceded by red.
        nxt = None
        for i in range(len(self.states) - 1, -1, -1):
            if self.states[i] in ("R", "G"):
                nxt = self.states[i]
            elif self.states[i] == "?" and nxt is not None:
                self.states[i] = "R"

    def runs(self) -> list[list]:
        """[[start, end, state], ...] with consecutive equal states merged."""
        out: list[list] = []
        for t, s in zip(self.times, self.states):
            if out and out[-1][2] == s:
                out[-1][1] = round(t, 2)
            else:
                out.append([round(t, 2), round(t, 2), s])
        return out

    def state_at(self, t: float) -> str:
        if not self.times:
            return "?"
        i = bisect.bisect_right(self.times, t) - 1
        return self.states[max(0, i)]

    def is_red(self, t: float, hold: float = 0.5) -> bool:
        return self.state_at(t) == "R" and self.state_at(t - hold) == "R"

    def green_onsets(self) -> list[float]:
        out = []
        prev = None
        for t, s in zip(self.times, self.states):
            if s == "G" and prev not in ("G", "Y"):
                out.append(t)
            prev = s
        return out

    def next_green_after(self, t: float) -> float | None:
        for g in self.green_onsets():
            if g > t:
                return g
        return None
