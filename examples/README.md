# SureStep.ai examples

This folder includes small synthetic room scan videos you can use directly:

- `examples/surestep_living.mp4`
- `examples/surestep_bedroom.mp4`
- `examples/surestep_kitchen.mp4`
- `examples/surestep_bathroom.mp4`

And matching example meshes + mesh preview videos:

- `examples/meshes/surestep_living.ply`
- `examples/meshes/surestep_bedroom.ply`
- `examples/meshes/surestep_kitchen.ply`
- `examples/meshes/surestep_bathroom.ply`
- `examples/mesh_preview_living.mp4`
- `examples/mesh_preview_bedroom.mp4`
- `examples/mesh_preview_kitchen.mp4`
- `examples/mesh_preview_bathroom.mp4`

Sample intake payload:

- `examples/intake_example.json`
- `examples/keragon_payload_example.json`

To regenerate the synthetic videos:

```bash
../scripts/generate_example_video.py --out /path/to/3d-fall/examples/surestep_living.mp4 --layout living
../scripts/generate_example_video.py --out /path/to/3d-fall/examples/surestep_bedroom.mp4 --layout bedroom
../scripts/generate_example_video.py --out /path/to/3d-fall/examples/surestep_kitchen.mp4 --layout kitchen
../scripts/generate_example_video.py --out /path/to/3d-fall/examples/surestep_bathroom.mp4 --layout bathroom
```

## End-to-end run (single command)

```bash
../scripts/run_all.py \
  --video /path/to/3d-fall/examples/surestep_living.mp4 \
  --workdir /tmp/surestep_example \
  --config /path/to/3d-fall/examples/example_config.yaml \
  --fps 2 \
  --auto-windows
```

Include intake data:

```bash
../scripts/run_all.py \
  --video /path/to/3d-fall/examples/surestep_living.mp4 \
  --workdir /tmp/surestep_example \
  --config /path/to/3d-fall/examples/example_config.yaml \
  --intake /path/to/3d-fall/examples/intake_example.json \
  --auto-windows
```

Keragon payload example:

```bash
../scripts/run_all.py \
  --keragon /path/to/3d-fall/examples/keragon_payload_example.json \
  --room living \
  --workdir /tmp/surestep_example \
  --auto-windows
```

## Optional: scale calibration

If you know a real-world distance in the room, run with `--scale-distance` and optionally provide a `picked_points.json`.

```bash
../scripts/run_all.py \
  --video /path/to/3d-fall/examples/surestep_bedroom.mp4 \
  --workdir /tmp/surestep_example \
  --config /path/to/3d-fall/examples/example_config.yaml \
  --scale-distance 3.2 \
  --auto-windows
```

## Outputs

- `/tmp/surestep_example/mesh.ply` (or `mesh_scaled.ply` if scaled)
- `/tmp/surestep_example/risk/risk_heatmap.png`
- `/tmp/surestep_example/report/report.json`
- `/tmp/surestep_example/report/report.pdf`
- `/tmp/surestep_example/mesh_preview.mp4`
- `/tmp/surestep_example/risk/room_interpretation.json`
- `/tmp/surestep_example/risk/risk_summary.json`
- `/tmp/surestep_example/room_output.json`
