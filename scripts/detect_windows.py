#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.window_detection import detect_window_planes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect window planes from a mesh.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--min-area", type=float, default=0.4)
    parser.add_argument("--max-area", type=float, default=6.0)
    parser.add_argument("--vertical-tol", type=float, default=0.25)
    parser.add_argument("--max-planes", type=int, default=6)
    args = parser.parse_args()

    candidates = detect_window_planes(
        Path(args.mesh),
        min_area=args.min_area,
        max_area=args.max_area,
        vertical_tol=args.vertical_tol,
        max_planes=args.max_planes,
    )

    if not candidates:
        print("No window candidates detected.")
        return

    print("windows:")
    for c in candidates:
        print("  - center: [{:.3f}, {:.3f}, {:.3f}]".format(*c.center))
        print("    normal: [{:.3f}, {:.3f}, {:.3f}]".format(*c.normal))
        print("    width: {:.3f}".format(c.width))
        print("    height: {:.3f}".format(c.height))
        print("    transmittance: 0.7")


if __name__ == "__main__":
    main()
