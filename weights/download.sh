#!/usr/bin/env bash
# Fetched once, with internet, before the offline evaluation run.
# Keep total weights <= 5 GB.
set -euo pipefail
cd "$(dirname "$0")"

URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
OUT="yolov8n.pt"

if [[ -f "$OUT" ]]; then
  echo "already have $OUT"
  ls -lh "$OUT"
  exit 0
fi

echo "downloading $OUT ..."
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 5 -o "$OUT" "$URL"
else
  python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve(
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8n.pt",
)
PY
fi
ls -lh "$OUT"
