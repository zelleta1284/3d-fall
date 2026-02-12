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

## Overlay video (synthetic input)

Use the `scripts/overlay_risk_video.py` helper to add fall-risk colors to the original footage. It projects component-specific fall hazards onto each frame and blends them with the original capture:

```
../scripts/overlay_risk_video.py \
  --video /path/to/3d-fall/examples/surestep_living.mp4 \
  --workdir /tmp/surestep_example \
  --output /tmp/surestep_example/livingRoom_risk_overlay.mp4
```

If the run already computed `risk_components.npz`, the overlay highlights obstacle/trip/slip/turn/glare/physics layers; otherwise it uses the overall heatmap as a fallback.

## Semantic hazards (object recognition)

By default, `run_all.py` runs an object-detection pass (Mask R-CNN) to tag common household objects and project them into the floor grid. This improves obstacle/trip coverage for things like furniture, tables, chairs, and clutter even if the mesh is coarse. A rug-like heuristic also boosts low-profile, flat patches just above the floor.

You can disable it with `--no-semantic`, or adjust detection density with:

```
--semantic-score-threshold 0.45
--semantic-frame-stride 2
--semantic-pixel-stride 4
--semantic-rug-min-height 0.01
--semantic-rug-max-height 0.08
--semantic-rug-gradient-max 0.04
--semantic-rug-weight 0.75
--semantic-small-object-area 0.015
--semantic-small-object-trip-boost 1.4
```

When enabled, the run writes `semantic_hazards.npz` and `semantic_hazards.json` in the workdir and merges them into the risk components.

If object detection stalls on Apple MPS, force CPU with `SURESTEP_DEVICE=cpu`.

## Room output JSON

Each room run emits `room_output.json` with patient intake, inferred adjustments, the room interpretation/risk summary, mitigation guidance, and the generated assets. The sanitized `examples/room_output_example.json` shows the expected structure with `<workdir>/…` placeholders for easy integration into other systems.
