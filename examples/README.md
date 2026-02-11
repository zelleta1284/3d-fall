# SureStep.ai examples

This folder includes small synthetic room scan videos you can use directly:

- `examples/surestep_living.mp4`
- `examples/surestep_bedroom.mp4`
- `examples/surestep_kitchen.mp4`
- `examples/surestep_bathroom.mp4`

To regenerate the synthetic videos:

```bash
../scripts/generate_example_video.py --out /Users/alextellez/Documents/New\ project/3d-fall/examples/surestep_living.mp4 --layout living
../scripts/generate_example_video.py --out /Users/alextellez/Documents/New\ project/3d-fall/examples/surestep_bedroom.mp4 --layout bedroom
../scripts/generate_example_video.py --out /Users/alextellez/Documents/New\ project/3d-fall/examples/surestep_kitchen.mp4 --layout kitchen
../scripts/generate_example_video.py --out /Users/alextellez/Documents/New\ project/3d-fall/examples/surestep_bathroom.mp4 --layout bathroom
```

## End-to-end run (single command)

```bash
../scripts/run_all.py \
  --video /Users/alextellez/Documents/New\ project/3d-fall/examples/surestep_living.mp4 \
  --workdir /tmp/surestep_example \
  --config /Users/alextellez/Documents/New\ project/3d-fall/examples/example_config.yaml \
  --fps 2 \
  --auto-windows
```

## Optional: scale calibration

If you know a real-world distance in the room, run with `--scale-distance` and optionally provide a `picked_points.json`.

```bash
../scripts/run_all.py \
  --video /Users/alextellez/Documents/New\ project/3d-fall/examples/surestep_bedroom.mp4 \
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
- `/tmp/surestep_example/mesh_preview.mp4`
