#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/room.mp4"
  exit 1
fi

VIDEO="$1"
WORKDIR="/tmp/surestep_example"
CONFIG="/Users/alextellez/Documents/New project/3d-fall/examples/example_config.yaml"

"/Users/alextellez/Documents/New project/3d-fall/scripts/run_all.py" \
  --video "$VIDEO" \
  --workdir "$WORKDIR" \
  --config "$CONFIG" \
  --fps 2 \
  --auto-windows

