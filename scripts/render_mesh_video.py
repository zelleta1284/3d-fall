#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a rotating mesh preview video.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--out", required=True, help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--points", type=int, default=30000)
    args = parser.parse_args()

    import open3d as o3d
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty.")

    pcd = mesh.sample_points_uniformly(number_of_points=args.points)
    pts = np.asarray(pcd.points)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    # Normalize for view
    center = pts.mean(axis=0)
    pts_centered = pts - center
    scale = np.max(np.linalg.norm(pts_centered, axis=1))
    pts_centered /= max(scale, 1e-6)

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0, 0, 0))
    fig.patch.set_facecolor((0, 0, 0))
    ax.set_axis_off()

    for i in range(args.frames):
        ax.clear()
        ax.set_axis_off()
        az = 360 * (i / max(args.frames - 1, 1))
        ax.view_init(elev=15, azim=az)
        ax.scatter(
            pts_centered[:, 0],
            pts_centered[:, 1],
            pts_centered[:, 2],
            s=0.2,
            c="#c7d5e0",
            alpha=0.8,
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(height, width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

    writer.release()
    plt.close(fig)
    print(f"Mesh preview video written to: {out_path}")


if __name__ == "__main__":
    main()
