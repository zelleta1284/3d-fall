#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_twin.simulate_risk import simulate_risk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fall risk heatmap from a mesh.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    img_path = simulate_risk(Path(args.mesh), Path(args.config), Path(args.out))
    print(f"Heatmap image written to: {img_path}")


if __name__ == "__main__":
    main()
