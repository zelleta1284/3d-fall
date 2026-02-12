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
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--pixel-stride", type=int, default=6)
    parser.add_argument("--low-height", type=float, default=0.12)
    args = parser.parse_args()

    config = SemanticConfig(
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        low_profile_height_m=args.low_height,
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
