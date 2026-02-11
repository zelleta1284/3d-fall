#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml
import open3d as o3d

from digital_twin.pipeline import run_pipeline
from digital_twin.simulate_risk import build_grids, load_mesh_vertices, simulate_risk
from digital_twin.window_detection import detect_window_planes
from digital_twin.reporting import compute_hotspots, save_report_json


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _scale_mesh(mesh_path: Path, distance_m: float, picked_path: Path | None, out_path: Path) -> Path:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty.")

    if picked_path is None:
        print("Pick two points in the mesh viewer, then press Q.")
        o3d.visualization.draw_geometries_with_editing([mesh])
        picked_path = Path("picked_points.json")

    if not picked_path.exists():
        raise RuntimeError(f"Picked points file not found: {picked_path}")

    data = json.loads(picked_path.read_text(encoding="utf-8"))
    indices = data.get("picked_points", [])
    if len(indices) < 2:
        raise RuntimeError("Need at least two picked points.")

    vertices = np.asarray(mesh.vertices)
    p0 = vertices[int(indices[0])]
    p1 = vertices[int(indices[1])]
    current_dist = float(np.linalg.norm(p0 - p1))
    if current_dist <= 1e-6:
        raise RuntimeError("Picked points are too close; cannot scale.")

    scale = distance_m / current_dist
    mesh.scale(scale, center=mesh.get_center())
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"Scale factor: {scale:.4f}")
    print(f"Scaled mesh written to: {out_path}")
    return out_path


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _insert_windows(cfg: Dict[str, Any], windows: list[dict]) -> Dict[str, Any]:
    lighting = cfg.get("lighting")
    if not isinstance(lighting, dict):
        lighting = {
            "latitude": 0.0,
            "longitude": 0.0,
            "timezone_offset_hours": 0,
            "datetime_iso": "",
            "glare_weight": 1.0,
            "ambient_weight": 0.3,
        }
    lighting["windows"] = windows
    cfg["lighting"] = lighting
    return cfg


def _make_report(
    heatmap_path: Path,
    config_path: Path,
    mesh_path: Path,
    out_dir: Path,
) -> None:
    heat = np.load(heatmap_path)
    cfg = _load_config(config_path)

    vertices = load_mesh_vertices(mesh_path)
    obstacle_grid, _, min_xy, _ = build_grids(
        vertices,
        float(cfg.get("room", {}).get("grid_size_m", 0.05)),
        float(cfg.get("room", {}).get("obstacle_height_m", 0.2)),
    )

    grid_size = float(cfg.get("room", {}).get("grid_size_m", 0.05))
    hotspots = compute_hotspots(heat, min_xy, grid_size, k=5)

    metadata = {
        "grid_size_m": grid_size,
        "min_xy": [float(min_xy[0]), float(min_xy[1])],
        "paths": cfg.get("paths", []),
        "biomechanics": cfg.get("biomechanics", {}),
        "lighting": cfg.get("lighting", {}),
        "physics": cfg.get("physics", {}),
    }

    _ensure_dir(out_dir)
    json_path = out_dir / "report.json"
    save_report_json(json_path, metadata, hotspots)

    try:
        import matplotlib.pyplot as plt

        max_val = np.percentile(heat, 99) if np.any(heat) else 1.0
        viz = np.clip(heat / max_val, 0, 1)

        plt.figure(figsize=(8.5, 11))
        plt.subplot(2, 1, 1)
        plt.imshow(viz, cmap="hot")
        plt.title("SureStep.ai Fall Risk Heatmap")
        plt.axis("off")

        plt.subplot(2, 1, 2)
        plt.axis("off")
        lines = [
            "Top Hotspots:",
            *[
                f"- ({h.position[0]:.2f}, {h.position[1]:.2f}) score={h.score:.2f}"
                for h in hotspots
            ],
        ]
        plt.text(0.01, 0.98, "\n".join(lines), va="top", fontsize=10)

        pdf_path = out_dir / "report.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"PDF report written to: {pdf_path}")
    except Exception as exc:
        print(f"PDF generation skipped: {exc}")

    print(f"JSON report written to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SureStep.ai end-to-end pipeline")
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--workdir", required=True, help="Working directory for outputs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    parser.add_argument("--median-depth-m", type=float, default=2.5)
    parser.add_argument("--scale-distance", type=float, default=None, help="Known distance in meters")
    parser.add_argument("--picked", help="picked_points.json path for scaling")
    parser.add_argument("--auto-windows", action="store_true", help="Auto-detect windows and update config copy")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    _ensure_dir(workdir)

    mesh_path = run_pipeline(
        video_path=Path(args.video),
        work_dir=workdir,
        fps=args.fps,
        median_depth_m=args.median_depth_m,
        run_colmap_flag=True,
    )

    final_mesh = mesh_path
    if args.scale_distance:
        scaled = workdir / "mesh_scaled.ply"
        picked = Path(args.picked) if args.picked else None
        final_mesh = _scale_mesh(mesh_path, args.scale_distance, picked, scaled)

    config_src = Path(args.config)
    config_out = workdir / "config_used.yaml"
    cfg = _load_config(config_src)

    if args.auto_windows:
        candidates = detect_window_planes(final_mesh)
        windows = [
            {
                "center": [c.center[0], c.center[1], c.center[2]],
                "normal": [c.normal[0], c.normal[1], c.normal[2]],
                "width": c.width,
                "height": c.height,
                "transmittance": 0.7,
            }
            for c in candidates
        ]
        cfg = _insert_windows(cfg, windows)
        print(f"Detected {len(windows)} window candidates.")

    _save_config(config_out, cfg)

    risk_dir = workdir / "risk"
    simulate_risk(final_mesh, config_out, risk_dir)

    heatmap_path = risk_dir / "risk_heatmap.npy"
    report_dir = workdir / "report"
    _make_report(heatmap_path, config_out, final_mesh, report_dir)

    print("Done. Outputs:")
    print(f"- Mesh: {final_mesh}")
    print(f"- Config used: {config_out}")
    print(f"- Heatmap: {risk_dir / 'risk_heatmap.png'}")
    print(f"- Report: {report_dir}")


if __name__ == "__main__":
    main()
