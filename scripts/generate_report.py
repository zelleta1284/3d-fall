#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from digital_twin.reporting import compute_hotspots, save_report_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSON + PDF report from heatmap.")
    parser.add_argument("--heatmap", required=True, help="Path to risk_heatmap.npy")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--min-xy", required=True, help="Min XY as 'x,y' from inspect_mesh output")
    parser.add_argument("--grid", type=float, required=True, help="Grid size in meters")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    heat = np.load(args.heatmap)
    min_xy = np.array([float(v) for v in args.min_xy.split(",")], dtype=np.float32)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hotspots = compute_hotspots(heat, min_xy, args.grid, k=5)

    metadata = {
        "grid_size_m": args.grid,
        "min_xy": [float(min_xy[0]), float(min_xy[1])],
        "paths": cfg.get("paths", []),
        "biomechanics": cfg.get("biomechanics", {}),
        "lighting": cfg.get("lighting", {}),
        "physics": cfg.get("physics", {}),
    }

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


if __name__ == "__main__":
    main()
