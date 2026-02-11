# SureStep.ai

Generate a coarse digital twin from an .mp4 and run a fall-risk heatmap simulation with biomechanics, lighting, and physics.

This project reconstructs a mesh from monocular video (COLMAP + AI depth), then simulates risk using gait parameters, room friction zones, sun glare, and optional physics-engine collision checks.

## Inputs

### Video
- `.mp4` scan of each room (living room, bedroom, kitchen, bathroom).

### Intake (Jotform)
Provide a JSON/YAML payload with:
- `age` (number)
- `gender` (string)
- `fall_last_6_months` (true/false)
- `fall_hospitalized` (true/false)
- `can_get_out_of_bed` (true/false)
- `assistive_aid` (true/false)

### Keragon POST payload
If you ingest through Keragon, save the POST body as JSON and pass it to `--keragon`. The payload should include `intake` and a `rooms` list with `video_path` and optional `config_path`. See `examples/keragon_payload_example.json`.

## What this does

1) Extracts frames from a video.
2) Uses COLMAP to compute camera poses (Structure-from-Motion).
3) Runs MiDaS (AI depth) on frames.
4) Fuses RGB + depth into a mesh.
5) Generates a risk heatmap from the mesh using:
   - Biomechanics (foot clearance, shuffle bias, cane)
   - Lighting glare (sun + point lights)
   - Physics collisions (PyBullet)

## Quick start

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need COLMAP installed and in your PATH.

### Included example video

Use the bundled synthetic scan in `examples/surestep_example.mp4` to test the pipeline quickly.

### 2) One-command run (recommended)

```bash
./scripts/run_all.py \
  --video /path/to/room.mp4 \
  --workdir /tmp/surestep_run \
  --config /path/to/3d-fall/config/example.yaml \
  --auto-windows
```

This also generates a mesh preview video (`mesh_preview.mp4`) with shaded surfaces and optional wireframe overlay so users can see the reconstructed geometry.

You can also pass a Jotform intake payload to include patient context:

```bash
./scripts/run_all.py \
  --video /path/to/room.mp4 \
  --workdir /tmp/surestep_run \
  --config /path/to/3d-fall/config/example.yaml \
  --intake /path/to/3d-fall/examples/intake_example.json \
  --auto-windows
```

Keragon payload example:

```bash
./scripts/run_all.py \
  --keragon /path/to/3d-fall/examples/keragon_payload_example.json \
  --room living \
  --workdir /tmp/surestep_run \
  --auto-windows
```

Auto-scaling: when you do not provide a known distance, the pipeline will estimate scale using home-size priors (ceiling height, door height, countertop height, etc.). Disable with `--no-auto-scale`.
Priors include common heights like door frames (~2.03m), counters (~0.91m), beds (~0.5m), sofa/chair seats (~0.45m), and toilet seats (~0.43m).

### 3) Manual steps (for debugging)

Reconstruct a mesh from an .mp4:

```bash
./scripts/run_pipeline.py --video /path/to/room.mp4 --workdir /tmp/room_recon --fps 2
```

Outputs:
- `/tmp/room_recon/mesh.ply`
- `/tmp/room_recon/depth/*.png`
- `/tmp/room_recon/colmap/*`

### 4) Inspect mesh bounds

```bash
./scripts/inspect_mesh.py --mesh /tmp/room_recon/mesh.ply
```

### 5) Calibrate scale (optional but recommended)

Pick two points in the mesh and enter the real-world distance between them (meters). This rescales the mesh for more accurate risk estimates.

```bash
./scripts/calibrate_scale.py --mesh /tmp/room_recon/mesh.ply --distance 3.2 --out /tmp/room_recon/mesh_scaled.ply
```

### 6) Detect windows (optional helper)

```bash
./scripts/detect_windows.py --mesh /tmp/room_recon/mesh_scaled.ply
```

Paste the output into the `lighting.windows` section of your config.

### 7) Run the fall-risk heatmap

Edit `config/example.yaml` to set path start/goal points, windows, and friction zones.

```bash
./scripts/simulate_risk.py --mesh /tmp/room_recon/mesh_scaled.ply --config config/example.yaml --out /tmp/room_recon/risk
```

Outputs:
- `/tmp/room_recon/risk/risk_heatmap.png`
- `/tmp/room_recon/risk/risk_heatmap.npy`
- `/tmp/room_recon/risk/room_interpretation.json`
- `/tmp/room_recon/risk/risk_summary.json`

`room_interpretation.json` explains how the mesh was interpreted (floor/ceiling estimates, obstacle thresholds) and how intake inputs adjusted parameters.

Each room run also produces a single JSON output at `/tmp/.../room_output.json` containing patient input, inferences, room risk summary, and DME-style mitigation suggestions.

### 8) Generate a report (JSON + PDF)

Use the mesh bounds output from `inspect_mesh.py` to fill `--min-xy`.

```bash
./scripts/generate_report.py --heatmap /tmp/room_recon/risk/risk_heatmap.npy \\
  --config config/example.yaml \\
  --min-xy -1.2,0.4 \\
  --grid 0.05 \\
  --out /tmp/room_recon/report
```

## Notes / limitations

- Monocular video reconstruction is scale-ambiguous. Depth is scaled to a median value (default 2.5m). Calibrate using a known measurement in the room for better scale.
- MiDaS depth is relative, not metric. The mesh is a coarse approximation.
- The physics stage is optional and uses PyBullet for collision checks. It is still a simplification and does not model full human biomechanics.
- Lighting glare is a coarse sun model based on location and time-of-day; it projects window light onto the floor.

## Directory structure

- `src/digital_twin/pipeline.py`: extraction + COLMAP + depth + fusion
- `src/digital_twin/lighting.py`: sun + window glare
- `src/digital_twin/physics.py`: PyBullet collision checks
- `src/digital_twin/simulate_risk.py`: heatmap generation
- `src/digital_twin/window_detection.py`: window plane detection
- `src/digital_twin/reporting.py`: report helpers
- `scripts/run_pipeline.py`: CLI for reconstruction
- `scripts/simulate_risk.py`: CLI for risk simulation
- `scripts/calibrate_scale.py`: mesh scaling helper
- `scripts/detect_windows.py`: window detection helper
- `scripts/generate_report.py`: report generator
- `scripts/render_mesh_video.py`: mesh preview video
- `scripts/run_all.py`: end-to-end pipeline
- `scripts/auto_scale_mesh.py`: auto-scale helper using priors
- `scripts/generate_example_meshes.py`: generate synthetic example meshes
- `config/example.yaml`: example config
