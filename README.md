# PDPU — Traffic Event Detection & Accident Anticipation

WIUT Hackathon 2026 · Computer Vision elimination task  
Team **PDPU** · ref **F863582F**

Fixed CCTV camera → time segments `[start_sec, end_sec, label]` (Part A) and a causal per-frame accident risk score (Part B).

## Repository layout

```
.
├── solution.py               # interface used by the harness
├── run_submission.py         # starter kit, unchanged
├── evaluate.py               # starter kit, unchanged
├── requirements.txt
├── weights/                  # open-weight checkpoints (or download.sh)
├── src/                      # detector, tracker, scene rules
├── notebooks/                # EDA / experiments
├── samples/                  # unlabeled clips from the same camera
├── examples/                 # official format examples
├── predictions_samples.json  # our output on the sample videos
└── README.md
```

## Install and run

```bash
python3 -m pip install -r requirements.txt
# optional: fetch YOLO weights once (internet)
bash weights/download.sh

# sample clips (from docs/Videos.pdf)
bash samples/download.sh

python run_submission.py --videos samples --out predictions_samples.json --team PDPU
python evaluate.py --pred predictions_samples.json --validate-only
# with our own labels:
# python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

What the organizers run offline (no internet):

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
```

Weights total ≤ 5 GB. Time budget: **3 × video duration** for Part A + Part B.

## Approach

```
video ─► align scene ─► YOLOv8n + ByteTrack (every 3rd frame) ─► tracks ─┐
             │                 └─► lamp colour at L1 ─► signal timeline ─┤
             └─► roadway, crossings, stop line, solid lines (px) ────────┴─► rules ─► merged segments
```

**Learned (open weights, no fine-tuning):** YOLOv8n trained on COCO (Ultralytics, AGPL-3.0) for cars, buses, trucks, motorcycles, bicycles and people; ByteTrack for ids. No other training data is used.

**Hand-made scene, drawn once:** `src/scene.json` holds the roadway, pedestrian crossings (plus refuge islands, where a pedestrian may stand), the stop line with the traffic light that controls it, and the solid lane lines before the stop line. It was drawn on `src/scene_ref.jpg` (from `sample_004`).

**Per-video alignment (`src/scene_align.py`):** the sample clips are not pixel-identical (framing shifts by 11–30 px at 720p). Each video's median background is matched to `scene_ref.jpg` with SIFT + RANSAC, and every scene point is moved through the resulting similarity transform.

**Signal state (`src/signals.py`):** the lamp colour at the aligned red and green lamp positions, read from the frames the tracker already decodes. Colour, not brightness, because in daylight the sunlit housing is brighter than a lit lamp. A washed-out red counts as red once green has been off longer than an amber phase.

**Rules (`src/rules.py`), following the organisers' start/end conventions:**

| Class | Rule |
|---|---|
| `red_light` | vehicle's ground point crosses the stop line towards the junction and drives on while the controlling light is red |
| `stop_line` | vehicle stops with its front past the stop line on red, without entering the junction; ends when the light turns green |
| `solid_line_crossing` | vehicle's ground point changes side of a solid line (stable on both sides); from wheel touching the line to fully in the new lane |
| `jaywalking` | walking person (riders of bikes and static false positives removed) whose feet are on the roadway outside every crossing |
| `failure_to_yield` | moving vehicle inside a crossing with a walking pedestrian close by, ahead of it |
| `stopped_vehicle` | stationary ≥ 10 s on the roadway; excluded when it drives off right after a signal change (a queue) or when it waited mostly on red |
| `congestion` | ≥ 70 % of vehicles crawling for ≥ 8 s, and the queue does not clear during ≥ 10 s of green |
| `wrong_way` | ≥ 2 s against the locally dominant direction, learned per grid cell (one global direction is meaningless on a two-way road) |

**Not emitted in Part A:** `accident`, `near_miss`, `illegal_turn`, `illegal_u_turn`, `road_obstacle`, `fire_smoke`. Box-overlap collision rules fired on every queue on this oblique view (every hit we inspected on the samples was false), and a class we predict that is absent from the test set lowers the macro score. Accident anticipation remains in Part B.

**Part B:** causal time-to-collision on an online tracker. `step()` never opens the video and never reads Part A.

**Determinism:** seed `42` (`src/config.py`) for Python, NumPy, PyTorch and OpenCV's RANSAC.

## Team

| Member | GitHub | Role |
|---|---|---|
| Begzad Kenesbaev | [mentisVeritas](https://github.com/mentisVeritas) | Pipeline, repo |
| Fariza Raxman | [farizarakhmanova](https://github.com/farizarakhmanova) | Annotation, EDA |
| mallokodev | [httpswap](https://github.com/httpswap) | Website, demo |

See `TEAM.md` for who does what this weekend.

## Website and demo

- Site (GitHub Pages): https://mentisveritas.github.io/pdpu-wiut-cv/ — built from `website/`; its data comes from
  `python scripts/build_site_assets.py --videos samples --preview samples/h264 --pred predictions_samples.json --out website/assets`
- Live demo (Gradio): `pip install -r demo/requirements.txt && python demo/app.py`; the hosted Hugging Face Space is assembled by `python scripts/pack_space.py`
- Smoke test without the multi-GB samples: `python run_submission.py --videos tests/fixtures --out /tmp/p.json`

## Docs

- `docs/task.md` — full task (RU)
- `docs/WIUT Hackathon _ CV Track Elimination Task.pdf` — official brief
- `docs/Videos.pdf` — sample video links
- `docs/starter/README.md` — starter-kit readme
- `TEAM.md` — who does what
