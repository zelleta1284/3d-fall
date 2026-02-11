#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  VIDEO="$REPO_ROOT/examples/surestep_living.mp4"
else
  VIDEO="$1"
fi
WORKDIR="/tmp/surestep_example"
CONFIG="$REPO_ROOT/examples/example_config.yaml"

"$REPO_ROOT/scripts/run_all.py" \
  --video "$VIDEO" \
  --workdir "$WORKDIR" \
  --config "$CONFIG" \
  --fps 2 \
  --auto-windows
