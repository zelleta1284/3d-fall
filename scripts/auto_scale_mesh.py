#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.scale_estimation import estimate_scale_from_mesh, load_mesh_vertices, scale_mesh  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-scale a mesh using home-size priors.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--out", required=True, help="Output scaled mesh path")
    args = parser.parse_args()

    vertices = load_mesh_vertices(Path(args.mesh))
    info = estimate_scale_from_mesh(vertices)
    scale_mesh(Path(args.mesh), Path(args.out), info["scale"])
    print(f"Scale factor: {info['scale']:.4f}")
    print(f"Method: {info['method']}")
    if "matched_priors" in info:
        print(f"Matched priors: {info['matched_priors']}")
    print(f"Scaled mesh written to: {args.out}")


if __name__ == "__main__":
    main()
