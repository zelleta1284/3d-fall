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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a rotating mesh preview video.")
    parser.add_argument("--mesh", required=True, help="Path to mesh.ply")
    parser.add_argument("--out", required=True, help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--points", type=int, default=30000)
    parser.add_argument("--faces", type=int, default=40000, help="Target face count for simplification")
    parser.add_argument("--mode", choices=["surface", "wireframe", "points"], default="surface")
    parser.add_argument("--wireframe", action="store_true", help="Overlay wireframe edges")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    use_open3d = False
    mesh = None
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(args.mesh)
        if mesh.is_empty():
            raise RuntimeError("Mesh is empty.")
        use_open3d = True
    except Exception:
        use_open3d = False

    if use_open3d:
        if args.mode != "points":
            try:
                mesh = mesh.simplify_quadric_decimation(args.faces)
            except Exception:
                pass
            mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals) if args.mode != "points" else None
    else:
        verts, tris = _load_ply_ascii(Path(args.mesh))
        normals = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0.02, 0.03, 0.04))
    fig.patch.set_facecolor((0.02, 0.03, 0.04))
    ax.set_axis_off()

    if args.mode == "points":
        if use_open3d and mesh is not None:
            pcd = mesh.sample_points_uniformly(number_of_points=args.points)
            pts = np.asarray(pcd.points)
        else:
            pts = verts
        center = pts.mean(axis=0)
        pts_centered = pts - center
        scale = np.max(np.linalg.norm(pts_centered, axis=1))
        pts_centered /= max(scale, 1e-6)
    else:
        center = verts.mean(axis=0)
        verts_centered = verts - center
        scale = np.max(np.linalg.norm(verts_centered, axis=1))
        verts_centered /= max(scale, 1e-6)
        light_dir = _normalize(np.array([0.6, 0.2, 0.7], dtype=np.float32))

    for i in range(args.frames):
        ax.clear()
        ax.set_axis_off()
        az = 360 * (i / max(args.frames - 1, 1))
        ax.view_init(elev=18, azim=az)

        if args.mode == "points":
            ax.scatter(
                pts_centered[:, 0],
                pts_centered[:, 1],
                pts_centered[:, 2],
                s=0.2,
                c="#c7d5e0",
                alpha=0.8,
            )
        else:
            theta = np.deg2rad(az)
            rot = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            v_rot = verts_centered @ rot.T

            tri_verts = v_rot[tris]
            if normals is not None:
                n_rot = _normalize(normals @ rot.T)
                tri_norms = n_rot[tris]
                intensity = np.clip((tri_norms @ light_dir).mean(axis=1), 0.1, 1.0)
            else:
                tri_norms = np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0])
                tri_norms = _normalize(tri_norms)
                intensity = np.clip(tri_norms @ light_dir, 0.1, 1.0)

            cmap = plt.get_cmap("cividis")
            facecolors = cmap(intensity)

            collection = Poly3DCollection(tri_verts, facecolors=facecolors, linewidths=0.1)
            if args.mode == "wireframe" or args.wireframe:
                collection.set_edgecolor((0.2, 0.25, 0.3, 0.5))
            else:
                collection.set_edgecolor((0, 0, 0, 0))

            ax.add_collection3d(collection)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

        fig.canvas.draw()
        if hasattr(fig.canvas, "buffer_rgba"):
            img = np.asarray(fig.canvas.buffer_rgba())
            img = img[:, :, :3]
        else:
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(height, width, 4)
            img = img[:, :, 1:4]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

    writer.release()
    plt.close(fig)
    print(f"Mesh preview video written to: {out_path}")


if __name__ == "__main__":
    main()
