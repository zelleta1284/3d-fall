#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def add_box(vertices: List[List[float]], faces: List[List[int]], center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2

    v = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
    idx = len(vertices)
    vertices.extend(v)

    faces.extend(
        [
            [idx + 0, idx + 1, idx + 2],
            [idx + 0, idx + 2, idx + 3],
            [idx + 4, idx + 5, idx + 6],
            [idx + 4, idx + 6, idx + 7],
            [idx + 0, idx + 1, idx + 5],
            [idx + 0, idx + 5, idx + 4],
            [idx + 1, idx + 2, idx + 6],
            [idx + 1, idx + 6, idx + 5],
            [idx + 2, idx + 3, idx + 7],
            [idx + 2, idx + 7, idx + 6],
            [idx + 3, idx + 0, idx + 4],
            [idx + 3, idx + 4, idx + 7],
        ]
    )


def build_room(vertices: List[List[float]], faces: List[List[int]]):
    # Room dimensions
    width, depth, height = 5.0, 4.0, 2.7
    wall_thickness = 0.1

    # Floor
    add_box(vertices, faces, center=(0, 0, 0.025), size=(width, depth, 0.05))

    # Walls
    add_box(vertices, faces, center=(0, depth / 2 + wall_thickness / 2, height / 2), size=(width + wall_thickness, wall_thickness, height))
    add_box(vertices, faces, center=(0, -depth / 2 - wall_thickness / 2, height / 2), size=(width + wall_thickness, wall_thickness, height))
    add_box(vertices, faces, center=(-width / 2 - wall_thickness / 2, 0, height / 2), size=(wall_thickness, depth + wall_thickness, height))
    add_box(vertices, faces, center=(width / 2 + wall_thickness / 2, 0, height / 2), size=(wall_thickness, depth + wall_thickness, height))


def build_layout(layout: str) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    build_room(vertices, faces)

    if layout == "living":
        add_box(vertices, faces, center=(1.4, 1.0, 0.5), size=(1.8, 0.8, 0.9))  # sofa
        add_box(vertices, faces, center=(0.5, 0.1, 0.35), size=(1.0, 0.6, 0.4))  # table
        add_box(vertices, faces, center=(-1.4, 0.5, 0.5), size=(0.8, 0.8, 0.9))  # chair
        add_box(vertices, faces, center=(0.4, 0.0, 0.01), size=(1.6, 1.2, 0.02))  # rug
        add_box(vertices, faces, center=(-0.5, 1.6, 0.6), size=(1.0, 0.4, 0.6))  # console
    elif layout == "bedroom":
        add_box(vertices, faces, center=(-0.3, 0.6, 0.5), size=(2.0, 1.6, 0.6))  # bed
        add_box(vertices, faces, center=(1.2, 0.6, 0.35), size=(0.6, 0.6, 0.7))  # nightstand
        add_box(vertices, faces, center=(-1.8, 1.6, 0.8), size=(1.2, 0.5, 1.0))  # dresser
        add_box(vertices, faces, center=(-1.0, -0.2, 0.01), size=(1.2, 0.8, 0.02))  # rug
    elif layout == "kitchen":
        add_box(vertices, faces, center=(-1.6, 1.0, 0.9), size=(1.6, 0.6, 0.9))  # counter
        add_box(vertices, faces, center=(0.6, 0.2, 0.9), size=(1.4, 0.7, 0.9))  # island
        add_box(vertices, faces, center=(1.6, 0.2, 0.5), size=(0.4, 0.4, 1.0))  # stool
        add_box(vertices, faces, center=(-2.0, -1.4, 1.1), size=(0.7, 0.8, 2.2))  # fridge
    elif layout == "bathroom":
        add_box(vertices, faces, center=(-1.4, 0.8, 0.5), size=(1.4, 0.7, 0.6))  # tub
        add_box(vertices, faces, center=(1.2, 0.6, 0.45), size=(0.7, 0.5, 0.9))  # sink
        add_box(vertices, faces, center=(1.2, -0.6, 0.4), size=(0.6, 0.6, 0.8))  # toilet
        add_box(vertices, faces, center=(0.2, -0.2, 0.01), size=(0.8, 0.6, 0.02))  # mat
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic example meshes for SureStep.ai")
    parser.add_argument("--out-dir", required=True, help="Output directory for meshes")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    layouts = ["living", "bedroom", "kitchen", "bathroom"]

    for layout in layouts:
        verts, faces = build_layout(layout)
        out_path = out_dir / f"surestep_{layout}.ply"
        write_ply(out_path, verts, faces)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
