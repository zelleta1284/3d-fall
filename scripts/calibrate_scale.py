#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import open3d as o3d


def load_picked_points(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    indices = data.get("picked_points", [])
    if len(indices) < 2:
        raise RuntimeError("Need at least two picked points.")
    return np.array(indices, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale mesh using a known real-world distance.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--distance", type=float, required=True, help="Real-world distance in meters")
    parser.add_argument("--picked", help="Path to picked_points.json (optional)")
    parser.add_argument("--out", required=True, help="Path to output scaled mesh")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty.")

    if args.picked:
        picked_path = Path(args.picked)
    else:
        print("Pick two points in the mesh viewer, then press Q.")
        o3d.visualization.draw_geometries_with_editing([mesh])
        picked_path = Path("picked_points.json")

    if not picked_path.exists():
        raise RuntimeError(f"Picked points file not found: {picked_path}")

    indices = load_picked_points(picked_path)
    vertices = np.asarray(mesh.vertices)
    p0 = vertices[indices[0]]
    p1 = vertices[indices[1]]
    current_dist = float(np.linalg.norm(p0 - p1))
    if current_dist <= 1e-6:
        raise RuntimeError("Picked points are too close; cannot scale.")

    scale = args.distance / current_dist
    mesh.scale(scale, center=mesh.get_center())
    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"Scale factor: {scale:.4f}")
    print(f"Scaled mesh written to: {args.out}")


if __name__ == "__main__":
    main()
