#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-6, 1.0, n)
    return v / n


def _load_ply_ascii(path: Path):
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        if line != "ply":
            raise RuntimeError("Not a PLY file")
        line = f.readline().strip()
        if "ascii" not in line:
            raise RuntimeError("PLY is not ascii")
        vertex_count = 0
        face_count = 0
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line.startswith("end_header"):
                break
        vertices = []
        for _ in range(vertex_count):
            parts = f.readline().strip().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        faces = []
        for _ in range(face_count):
            parts = f.readline().strip().split()
            if not parts:
                continue
            n = int(parts[0])
            if n >= 3:
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def _draw_triangle(img: np.ndarray, tri2d: np.ndarray, color: np.ndarray, wireframe: bool) -> None:
    import cv2

    pts = tri2d.astype(np.int32)
    cv2.fillConvexPoly(img, pts, color.tolist())
    if wireframe:
        cv2.polylines(img, [pts], isClosed=True, color=(40, 50, 60), thickness=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a rotating mesh preview video.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--out", required=True, help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--wireframe", action="store_true", help="Overlay wireframe edges")
    parser.add_argument("--face-limit", type=int, default=12000, help="Max faces to draw")
    args = parser.parse_args()

    import cv2

    verts = None
    faces = None
    normals = None

    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(args.mesh)
        if mesh.is_empty():
            raise RuntimeError("Mesh is empty.")
        try:
            mesh = mesh.simplify_quadric_decimation(args.face_limit)
        except Exception:
            pass
        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)
    except Exception:
        verts, faces = _load_ply_ascii(Path(args.mesh))
        normals = None

    if faces.shape[0] > args.face_limit:
        idx = np.random.choice(faces.shape[0], args.face_limit, replace=False)
        faces = faces[idx]

    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.max(np.linalg.norm(verts, axis=1))
    verts = verts / max(scale, 1e-6)

    width, height = args.width, args.height
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    light_dir = _normalize(np.array([0.6, 0.3, 0.7], dtype=np.float32))

    for i in range(args.frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        az = 2 * np.pi * (i / max(args.frames - 1, 1))
        el = np.deg2rad(18)
        rot_z = np.array(
            [[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(el), -np.sin(el)], [0.0, np.sin(el), np.cos(el)]],
            dtype=np.float32,
        )
        rot = rot_x @ rot_z

        v = verts @ rot.T
        if normals is not None:
            n = _normalize(normals @ rot.T)
        else:
            n = None

        # Perspective projection
        cam_dist = 3.0
        f = 1.4
        z = v[:, 2] + cam_dist
        x = (v[:, 0] * f / z) * (width * 0.5) + width * 0.5
        y = (-v[:, 1] * f / z) * (height * 0.5) + height * 0.55

        # Draw faces back-to-front
        face_depth = v[faces][:, :, 2].mean(axis=1)
        order = np.argsort(face_depth)[::-1]

        for idx in order:
            fidx = faces[idx]
            tri = np.column_stack([x[fidx], y[fidx], z[fidx]]).astype(np.float32)
            if np.any(np.isnan(tri)):
                continue

            if n is not None:
                tri_norm = n[fidx].mean(axis=0)
            else:
                a = v[fidx[1]] - v[fidx[0]]
                b = v[fidx[2]] - v[fidx[0]]
                tri_norm = _normalize(np.cross(a, b))

            intensity = float(np.clip(np.dot(tri_norm, light_dir), 0.1, 1.0))
            base = np.array([70, 110, 140], dtype=np.float32)
            color = np.clip(base * intensity + 30, 0, 255).astype(np.uint8)

            tri2d = tri[:, :2]
            _draw_triangle(img, tri2d, color, args.wireframe)

        writer.write(img)

    writer.release()
    print(f"Mesh preview video written to: {out_path}")


if __name__ == "__main__":
    main()
