#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a mesh from an MP4 video.")
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument("--workdir", required=True, help="Working directory for outputs")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    parser.add_argument("--median-depth-m", type=float, default=2.5, help="Median depth scale in meters")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP (not recommended)")
    args = parser.parse_args()

    mesh_path = run_pipeline(
        video_path=Path(args.video),
        work_dir=Path(args.workdir),
        fps=args.fps,
        median_depth_m=args.median_depth_m,
        run_colmap_flag=not args.skip_colmap,
    )
    print(f"Mesh written to: {mesh_path}")


if __name__ == "__main__":
    main()
