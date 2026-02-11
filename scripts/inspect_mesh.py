#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import open3d as o3d
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Print mesh bounds and show a viewer.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--no-view", action="store_true", help="Skip viewer")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty.")
    bounds = np.asarray(mesh.vertices)
    min_xyz = bounds.min(axis=0)
    max_xyz = bounds.max(axis=0)
    print(f"min xyz: {min_xyz}")
    print(f"max xyz: {max_xyz}")

    if not args.no_view:
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
