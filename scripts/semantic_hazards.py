#!/usr/bin/env python3
import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.semantic_hazards import SemanticConfig, compute_semantic_hazards


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute semantic hazard maps from object detection.")
    parser.add_argument("--workdir", required=True, help="Workdir with frames/depth/colmap outputs")
    parser.add_argument("--mesh", required=True, help="Mesh path used for grid sizing")
    parser.add_argument("--grid-size", type=float, required=True, help="Grid size in meters")
    parser.add_argument("--obstacle-height", type=float, required=True, help="Obstacle height threshold in meters")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--summary", help="Optional JSON summary path")
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--low-height", type=float, default=0.12)
    parser.add_argument("--rug-min-height", type=float, default=0.01)
    parser.add_argument("--rug-max-height", type=float, default=0.08)
    parser.add_argument("--rug-gradient-max", type=float, default=0.04)
    parser.add_argument("--rug-weight", type=float, default=0.75)
    parser.add_argument("--small-object-area", type=float, default=0.015)
    parser.add_argument("--small-object-trip-boost", type=float, default=1.4)
    args = parser.parse_args()

    config = SemanticConfig(
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        low_profile_height_m=args.low_height,
        rug_min_height_m=args.rug_min_height,
        rug_max_height_m=args.rug_max_height,
        rug_gradient_max_m=args.rug_gradient_max,
        rug_weight=args.rug_weight,
        small_object_area_ratio=args.small_object_area,
        small_object_trip_boost=args.small_object_trip_boost,
    )

    compute_semantic_hazards(
        workdir=Path(args.workdir),
        mesh_path=Path(args.mesh),
        grid_size=float(args.grid_size),
        obstacle_height_m=float(args.obstacle_height),
        out_path=Path(args.out),
        summary_path=Path(args.summary) if args.summary else None,
        config=config,
    )


if __name__ == "__main__":
    main()
