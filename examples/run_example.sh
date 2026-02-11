#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  VIDEO="/Users/alextellez/Documents/New project/3d-fall/examples/surestep_living.mp4"
else
  VIDEO="$1"
fi
WORKDIR="/tmp/surestep_example"
CONFIG="/Users/alextellez/Documents/New project/3d-fall/examples/example_config.yaml"

"/Users/alextellez/Documents/New project/3d-fall/scripts/run_all.py" \
  --video "$VIDEO" \
  --workdir "$WORKDIR" \
  --config "$CONFIG" \
  --fps 2 \
  --auto-windows
