# SureStep.ai examples

These examples assume you have a local .mp4 scan of a room. The pipeline works from video frames.

## End-to-end run (single command)

```bash
../scripts/run_all.py \
  --video /path/to/room.mp4 \
  --workdir /tmp/surestep_example \
  --config /Users/alextellez/Documents/New\ project/3d-fall/examples/example_config.yaml \
  --fps 2 \
  --auto-windows
```

## Optional: scale calibration

If you know a real-world distance in the room, run with `--scale-distance` and optionally provide a `picked_points.json`.

```bash
../scripts/run_all.py \
  --video /path/to/room.mp4 \
  --workdir /tmp/surestep_example \
  --config /Users/alextellez/Documents/New\ project/3d-fall/examples/example_config.yaml \
  --scale-distance 3.2 \
  --auto-windows
```

## Outputs

- `/tmp/surestep_example/mesh.ply` (or `mesh_scaled.ply` if scaled)
- `/tmp/surestep_example/risk/risk_heatmap.png`
- `/tmp/surestep_example/report/report.json`
- `/tmp/surestep_example/report/report.pdf`
