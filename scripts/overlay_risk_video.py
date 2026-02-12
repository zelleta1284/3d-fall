#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array(
        [
            [
                1 - 2 * (qy ** 2 + qz ** 2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx ** 2 + qz ** 2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx ** 2 + qy ** 2),
            ],
        ],
        dtype=np.float64,
    )


def _parse_cameras_txt(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    cameras: Dict[int, Dict[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _parse_images_txt(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    images: Dict[str, Dict[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    idx = 0
    while idx < len(lines):
        parts = lines[idx].split()
        idx += 1
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8])
        name = parts[9]
        images[name] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id}
        idx += 1  # skip point list line
    return images


def _project_points(
    points: np.ndarray,
    camera: Dict[str, np.ndarray],
    cam_params: Dict[int, Dict[str, np.ndarray]],
) -> np.ndarray:
    qvec = camera["qvec"]
    tvec = camera["tvec"]
    cam_id = camera["camera_id"]
    cam = cam_params[cam_id]
    R = _quaternion_to_matrix(qvec)
    pts_cam = (R @ points.T).T + tvec
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    if pts_cam.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    x = pts_cam[:, 0] / pts_cam[:, 2]
    y = pts_cam[:, 1] / pts_cam[:, 2]
    model = cam["model"]
    params = cam["params"]
    if model == "SIMPLE_RADIAL":
        fx, fy, cx, cy = params[:4]
        k1 = params[4] if len(params) >= 5 else 0.0
        r2 = x * x + y * y
        scale = 1 + k1 * r2
        x *= scale
        y *= scale
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    else:
        fx, fy, cx, cy = params[:4]
    u = x * fx + cx
    v = y * fy + cy
    return np.column_stack([u, v]), valid


def _color_map(values: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("inferno")
    colors = cmap(np.clip(values, 0.0, 1.0))[:, :3]
    colors = (colors[:, ::-1] * 255).astype(np.uint8)
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay the risk heatmap on the original video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--workdir", required=True, help="Workdir that produced run_all outputs")
    parser.add_argument("--output", required=True, help="Path for overlay video")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate used by run_all")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha")
    parser.add_argument("--max-points", type=int, default=800, help="Max grid points to project each frame")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    frames_dir = workdir / "frames"
    risk_dir = workdir / "risk"
    colmap_txt = workdir / "colmap_txt"
    heatmap_path = risk_dir / "risk_heatmap.npy"
    room_interp = risk_dir / "room_interpretation.json"
    mesh_path = workdir / "mesh_scaled_auto.ply"

    if not all(p.exists() for p in (heatmap_path, frames_dir, colmap_txt, mesh_path, room_interp)):
        raise RuntimeError("Missing required outputs in workdir.")

    heatmap = np.load(heatmap_path)
    interp = json.loads(room_interp.read_text(encoding="utf-8"))
    grid_size = float(interp["room"]["grid_size_m"])
    floor_z = float(interp["room"]["floor_z_estimate_m"])
    mesh = cv2.imread if False else None  # no-op for lint
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    min_xyz = np.percentile(verts, 2, axis=0)
    min_xy = min_xyz[:2]

    h, w = heatmap.shape
    xs = min_xy[0] + (np.arange(w) + 0.5) * grid_size
    ys = min_xy[1] + (np.arange(h) + 0.5) * grid_size
    xx, yy = np.meshgrid(xs, ys)
    floor_pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, floor_z, dtype=np.float32)])
    heat_flat = heatmap.ravel()
    if heat_flat.max() <= 0:
        raise RuntimeError("Heatmap all zeros; nothing to overlay.")
    norm = heat_flat / heat_flat.max()
    sorted_idx = np.argsort(norm)[::-1]
    sorted_idx = sorted_idx[: args.max_points]
    points = floor_pts[sorted_idx]
    norm_values = norm[sorted_idx]

    cameras = _parse_cameras_txt(colmap_txt / "cameras.txt")
    images = _parse_images_txt(colmap_txt / "images.txt")
    available_frames = sorted(images.keys())
    if not available_frames:
        raise RuntimeError("No images in colmap output.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(video_fps / args.fps)), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output), fourcc, video_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter for overlay.")

    last_camera: Optional[Dict[str, np.ndarray]] = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pose_idx = min(frame_idx // frame_interval, len(available_frames) - 1)
        pose_name = available_frames[pose_idx]
        camera = images.get(pose_name)
        if camera is None:
            camera = last_camera
        else:
            last_camera = camera
        overlay = frame.copy()
        if camera:
            proj, valid = _project_points(points, camera, cameras)
            valid_values = norm_values[valid]
            for (u, v), value in zip(proj.astype(np.int32), valid_values):
                if 0 <= u < width and 0 <= v < height:
                    radius = max(6, int(8 + value * 25))
                    red = int(200 * value + 55)
                    green = int(200 * (1 - value) + 30)
                    color = (0, green, red)
                    cv2.circle(overlay, (u, v), radius, color, thickness=-1)
        frame = cv2.addWeighted(overlay, args.alpha, frame, 1.0 - args.alpha, 0)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Overlay video written to: {args.output}")


if __name__ == "__main__":
    main()
