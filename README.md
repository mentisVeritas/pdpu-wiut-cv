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

## Approach (draft)

Learned: open-weight detector (YOLO family) + tracker.  
Rule-based: scene layout (lanes, stop line, crossing) and trajectory rules for the 14 official classes.  
Part B: causal time-to-collision / kinematics from tracks only (no future frames, no Part A leakage).

Public datasets used for any training will be listed here with licences. Seeds will be fixed before the tagged submit commit.

## Team

| Member | Role |
|---|---|
| Kenesbaev Begzad Akilbek Uli | — |
| TBD | — |
| TBD | — |

## Docs

- `docs/task.md` — full task (RU)
- `docs/WIUT Hackathon _ CV Track Elimination Task.pdf` — official brief
- `docs/Videos.pdf` — sample video links
- `docs/starter/README.md` — starter-kit readme
