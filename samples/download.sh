#!/usr/bin/env bash
# Sample clips from Videos.pdf (same camera / angle as the hidden test set).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v gdown >/dev/null 2>&1; then
  python3 -m pip install --quiet gdown
fi

# file_id  local_name
download() {
  local id="$1"
  local name="$2"
  if [[ -f "$name" ]]; then
    echo "already have $name"
    return
  fi
  echo "downloading $name ..."
  python3 -m gdown "https://drive.google.com/uc?id=${id}" -O "$name" --continue --retries 5
}

download "1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS" "sample_001.mp4"
download "1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_" "sample_002.mp4"
download "10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ" "sample_003.mp4"
download "1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd" "sample_004.mp4"

ls -lh ./*.mp4
