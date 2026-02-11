# SureStep.ai

Generate a coarse digital twin from an .mp4 and run a fall-risk heatmap simulation with biomechanics, lighting, and physics.

This project reconstructs a mesh from monocular video (COLMAP + AI depth), then simulates risk using gait parameters, room friction zones, sun glare, and optional physics-engine collision checks.

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
  --config /Users/alextellez/Documents/New\ project/3d-fall/config/example.yaml \
  --auto-windows
```

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
- `config/example.yaml`: example config
