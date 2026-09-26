#!/usr/bin/env bash
# Fetched once, with internet, before the offline evaluation run.
# Keep total weights <= 5 GB.
set -euo pipefail
cd "$(dirname "$0")"

# Ultralytics will cache YOLO weights here after the first run.
# If you vendor a specific checkpoint, put the URL below.
echo "weights/: place open-weight checkpoints here (YOLO / tracker)."
echo "Nothing extra to download yet — models are fetched by the pipeline on first use,"
echo "or drop .pt files into this folder."
