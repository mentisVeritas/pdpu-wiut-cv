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

Learned: YOLOv8n (COCO, open weights, AGPL/Ultralytics licence) + ByteTrack.  
Rule-based: stopped vehicle, congestion, wrong-way vs dominant flow, jaywalking (needs roadway/crossing in `src/scene.json`), near-miss / accident from closing distance.  
Part B: causal time-to-collision on an online tracker. `step()` never opens the video and never reads Part A.

No extra training data yet. Seed `42` (`src/config.py`).

Classes that need painted geometry (`red_light`, `stop_line`, `illegal_turn`, `solid_line_crossing`, `illegal_u_turn`) are not emitted until `src/scene.json` is filled.

## Team

| Member | GitHub | Role |
|---|---|---|
| Begzad Kenesbaev | [mentisVeritas](https://github.com/mentisVeritas) | Pipeline, repo |
| Fariza Raxman | [farizarakhmanova](https://github.com/farizarakhmanova) | Annotation, EDA |
| mallokodev | [httpswap](https://github.com/httpswap) | Website, demo |

See `TEAM.md` for who does what this weekend.

## Website and demo

- Site (GitHub Pages): https://mentisveritas.github.io/pdpu-wiut-cv/
- Local site: `python3 -m http.server 8080 --directory website`
- Model demo: `pip install -r demo/requirements.txt && streamlit run demo/app.py`

## Docs

- `docs/task.md` — full task (RU)
- `docs/WIUT Hackathon _ CV Track Elimination Task.pdf` — official brief
- `docs/Videos.pdf` — sample video links
- `docs/starter/README.md` — starter-kit readme
- `TEAM.md` — who does what
