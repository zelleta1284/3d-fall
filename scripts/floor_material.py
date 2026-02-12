#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.floor_material import FloorMaterialConfig, infer_floor_material


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer floor material and friction from video frames.")
    parser.add_argument("--workdir", required=True, help="Workdir with frames/depth/colmap outputs")
    parser.add_argument("--mesh", required=True, help="Mesh path used for floor height estimation")
    parser.add_argument("--obstacle-height", type=float, required=True, help="Obstacle height threshold in meters")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--floor-min-height", type=float, default=-0.01)
    parser.add_argument("--floor-max-height", type=float, default=0.03)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--patch-count", type=int, default=24)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    config = FloorMaterialConfig(
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        floor_height_min_m=args.floor_min_height,
        floor_height_max_m=args.floor_max_height,
        patch_size=args.patch_size,
        patch_count=args.patch_count,
        min_confidence=args.min_confidence,
    )

    infer_floor_material(
        workdir=Path(args.workdir),
        mesh_path=Path(args.mesh),
        obstacle_height_m=args.obstacle_height,
        out_path=Path(args.out),
        config=config,
    )


if __name__ == "__main__":
    main()
