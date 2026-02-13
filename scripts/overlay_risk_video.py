#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from digital_twin.simulate_risk import (
    a_star,
    build_friction_grid,
    build_grids,
    compute_turn_risk,
    load_config,
    to_grid,
)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qvec = camera["qvec"]
    tvec = camera["tvec"]
    cam_id = camera["camera_id"]
    cam = cam_params[cam_id]
    R = _quaternion_to_matrix(qvec)
    pts_cam = (R @ points.T).T + tvec
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    if pts_cam.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool), np.empty((0,), dtype=np.float32)
    x = pts_cam[:, 0] / pts_cam[:, 2]
    y = pts_cam[:, 1] / pts_cam[:, 2]
    model = cam["model"]
    params = cam["params"]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        fx = fy = f
        r2 = x * x + y * y
        scale = 1 + k1 * r2
        x *= scale
        y *= scale
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        fx = fy = f
        r2 = x * x + y * y
        scale = 1 + k1 * r2 + k2 * r2 * r2
        x *= scale
        y *= scale
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = x * radial + x_tan
        y = y * radial + y_tan
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    else:
        fx, fy, cx, cy = params[:4]
    u = x * fx + cx
    v = y * fy + cy
    return np.column_stack([u, v]), valid, pts_cam[:, 2]


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
    parser.add_argument("--alpha", type=float, default=0.85, help="Overlay alpha")
    parser.add_argument("--max-points", type=int, default=4000, help="Max grid points to project each frame")
    parser.add_argument("--heat-quantile", type=float, default=0.7, help="Quantile threshold for heat points")
    parser.add_argument("--component-quantile", type=float, default=0.8, help="Quantile threshold for component points")
    parser.add_argument("--point-radius-min", type=int, default=10, help="Minimum overlay point radius")
    parser.add_argument("--point-radius-max", type=int, default=40, help="Maximum overlay point radius")
    parser.add_argument("--show-heat", action="store_true", help="Include heatmap layer")
    parser.add_argument("--no-components", dest="show_components", action="store_false", help="Hide component layers")
    parser.set_defaults(show_components=True)
    parser.add_argument("--legend", action="store_true", help="Draw legend")
    parser.add_argument("--depth-occlusion", dest="depth_occlusion", action="store_true", help="Hide overlays behind nearer surfaces")
    parser.add_argument("--no-depth-occlusion", dest="depth_occlusion", action="store_false", help="Disable depth occlusion")
    parser.set_defaults(depth_occlusion=True)
    parser.add_argument("--depth-occlusion-tol-m", type=float, default=0.15, help="Depth occlusion tolerance (meters)")
    parser.add_argument("--depth-occlusion-tol-ratio", type=float, default=0.2, help="Depth occlusion tolerance ratio")
    parser.add_argument("--min-v-ratio", type=float, default=0.5, help="Only draw overlay below this vertical ratio")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    frames_dir = workdir / "frames"
    risk_dir = workdir / "risk"
    colmap_txt = workdir / "colmap_txt"
    if not colmap_txt.exists():
        alt = workdir / "colmap" / "sparse_txt"
        if alt.exists():
            colmap_txt = alt
    heatmap_path = risk_dir / "risk_heatmap.npy"
    room_interp = risk_dir / "room_interpretation.json"
    mesh_path = workdir / "mesh_scaled_auto.ply"
    config_path = workdir / "config_used.yaml"

    depth_dir = workdir / "depth"
    if not all(p.exists() for p in (heatmap_path, frames_dir, colmap_txt, mesh_path, room_interp, config_path)):
        raise RuntimeError("Missing required outputs in workdir.")

    cfg = load_config(config_path)
    scale_result = {}
    try:
        scale_result = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        scale_result = scale_result.get("scale_result", {})
    except Exception:
        scale_result = {}
    heatmap = np.load(heatmap_path)
    obstacle_grid, height_grid, min_xy, floor_z = build_grids(
        np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices),
        cfg.room.grid_size_m,
        cfg.room.obstacle_height_m,
    )
    grid_size = cfg.room.grid_size_m

    scale = None
    center = None
    if scale_result:
        try:
            scale = float(scale_result.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = None
    if scale and scale != 1.0:
        unscaled_mesh_path = workdir / "mesh.ply"
        if unscaled_mesh_path.exists():
            unscaled_mesh = o3d.io.read_triangle_mesh(str(unscaled_mesh_path))
            if not unscaled_mesh.is_empty():
                center = np.asarray(unscaled_mesh.get_center(), dtype=np.float32)

    def _unscale(points: np.ndarray) -> np.ndarray:
        if scale and center is not None:
            return center + (points - center) / scale
        return points

    h, w = heatmap.shape
    xs = min_xy[0] + (np.arange(w) + 0.5) * grid_size
    ys = min_xy[1] + (np.arange(h) + 0.5) * grid_size
    xx, yy = np.meshgrid(xs, ys)
    floor_pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, floor_z, dtype=np.float32)])
    floor_pts = _unscale(floor_pts)

    components_path = risk_dir / "risk_components.npz"
    component_colors: Dict[str, Tuple[int, int, int]] = {
        "obstacle": (0, 0, 255),   # red
        "trip": (0, 85, 255),      # deep orange
        "slip": (255, 0, 0),       # blue
        "turn": (0, 255, 0),       # green
        "physics": (255, 0, 255),  # magenta
    }
    component_layers: List[Dict[str, np.ndarray]] = []

    if args.show_components and components_path.exists():
        comp_data = np.load(components_path)
        per_layer = max(50, args.max_points // max(len(component_colors), 1))
        for name, color in component_colors.items():
            if name not in comp_data:
                continue
            arr = comp_data[name].ravel()
            if not np.any(arr > 0):
                continue
            limit = max(np.percentile(arr, 95), arr.max(), 1e-6)
            norm_vals = arr / limit
            thresh = np.quantile(norm_vals, args.component_quantile) if np.any(norm_vals > 0) else 1.0
            idx = np.where(norm_vals >= thresh)[0]
            if idx.size > per_layer:
                idx = np.random.choice(idx, size=per_layer, replace=False)
            component_layers.append(
                {
                    "name": name,
                    "points": floor_pts[idx],
                    "values": np.clip(norm_vals[idx], 0.0, 1.0),
                    "color": np.array(color, dtype=np.int32),
                    "mode": "component",
                }
            )

    if args.show_heat:
        heat_flat = heatmap.ravel()
        heat_max = heat_flat.max() if heat_flat.size else 0.0
        if heat_max <= 0:
            raise RuntimeError("Heatmap all zeros; nothing to overlay.")
        norm_heat = heat_flat / heat_max
        heat_thresh = np.quantile(norm_heat, args.heat_quantile) if np.any(norm_heat > 0) else 1.0
        idx_heat = np.where(norm_heat >= heat_thresh)[0]
        if idx_heat.size > args.max_points:
            idx_heat = np.random.choice(idx_heat, size=args.max_points, replace=False)
        component_layers.append(
            {
                "name": "heat",
                "points": floor_pts[idx_heat],
                "values": np.clip(norm_heat[idx_heat], 0.0, 1.0),
                "mode": "heat",
            }
        )

    def grid_cells_to_world(indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        xs = min_xy[0] + (indices[:, 1] + 0.5) * grid_size
        ys = min_xy[1] + (indices[:, 0] + 0.5) * grid_size
        zs = np.full(xs.shape, floor_z, dtype=np.float32)
        pts = np.column_stack([xs, ys, zs])
        return _unscale(pts)

    def add_layer(name: str, indices: np.ndarray, values: np.ndarray, color: Tuple[int, int, int]) -> None:
        if indices.size == 0:
            return
        pts = grid_cells_to_world(indices)
        component_layers.append(
            {
                "name": name,
                "points": pts,
                "values": values,
                "color": np.array(color, dtype=np.int32),
                "mode": "component",
            }
        )

    height_diff = height_grid - floor_z
    obstacle_indices = np.column_stack(np.where(obstacle_grid == 1))
    if obstacle_indices.size:
        add_layer(
            "obstacle",
            obstacle_indices,
            np.full(obstacle_indices.shape[0], 1.0, dtype=np.float32),
            component_colors["obstacle"],
        )

    trip_threshold = cfg.biomechanics.foot_clearance_m * (1.0 - 0.5 * cfg.biomechanics.shuffle_bias)
    trip_indices = np.column_stack(np.where(height_diff > trip_threshold))
    if trip_indices.size:
        add_layer(
            "trip",
            trip_indices,
            np.clip(height_diff[trip_indices[:, 0], trip_indices[:, 1]] / (0.2 + trip_threshold), 0.0, 1.0),
            component_colors["trip"],
        )

    friction_grid = build_friction_grid(cfg.room, obstacle_grid.shape, min_xy)
    slip_indices = np.column_stack(np.where(friction_grid < cfg.room.default_friction - 0.05))
    if slip_indices.size:
        add_layer(
            "slip",
            slip_indices,
            np.clip((cfg.room.default_friction - friction_grid[slip_indices[:, 0], slip_indices[:, 1]]) / 0.5, 0.0, 1.0),
            component_colors["slip"],
        )

    for path_spec in cfg.paths:
        start = to_grid(path_spec.start, min_xy, grid_size)
        goal = to_grid(path_spec.goal, min_xy, grid_size)
        path = np.array(a_star(obstacle_grid, start, goal))
        if path.size == 0:
            continue
        risk = compute_turn_risk(path.tolist())
        add_layer(
            "turn",
            path,
            np.clip(risk, 0.0, 1.0),
            component_colors["turn"],
        )

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
    last_pose_name: Optional[str] = None
    last_depth: Optional[np.ndarray] = None
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
            depth_m = None
            if args.depth_occlusion and depth_dir.exists():
                if pose_name != last_pose_name:
                    depth_path = depth_dir / f"{Path(pose_name).stem}.png"
                    if depth_path.exists():
                        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if depth_img is not None:
                            last_depth = depth_img.astype(np.float32) / 1000.0
                            last_pose_name = pose_name
                    else:
                        last_depth = None
                        last_pose_name = pose_name
                depth_m = last_depth

            for layer in component_layers:
                proj, valid, depths = _project_points(layer["points"], camera, cameras)
                if proj.size == 0:
                    continue
                values = layer["values"][valid]
                finite = np.isfinite(proj).all(axis=1)
                if not np.any(finite):
                    continue
                proj = proj[finite]
                values = values[finite]
                depths = depths[finite]
                # Discard points above a conservative image band (avoid wall overlays).
                if args.min_v_ratio > 0:
                    v_min = int(height * args.min_v_ratio)
                    keep = proj[:, 1] >= v_min
                    if not np.any(keep):
                        continue
                    proj = proj[keep]
                    values = values[keep]
                    depths = depths[keep]

                if depth_m is not None:
                    u = np.round(proj[:, 0]).astype(int)
                    v = np.round(proj[:, 1]).astype(int)
                    inside = (u >= 0) & (u < depth_m.shape[1]) & (v >= 0) & (v < depth_m.shape[0])
                    if not np.any(inside):
                        continue
                    u = u[inside]
                    v = v[inside]
                    proj = proj[inside]
                    values = values[inside]
                    depths = depths[inside]
                    depth_at = depth_m[v, u]
                    # Convert COLMAP depth to meters if we have a scale factor
                    depth_colmap_m = depths * (scale if scale else 1.0)
                    visible = (depth_at > 0) & (
                        depth_colmap_m <= depth_at * (1.0 + args.depth_occlusion_tol_ratio) + args.depth_occlusion_tol_m
                    )
                    if not np.any(visible):
                        continue
                    proj = proj[visible]
                    values = values[visible]
                proj = np.nan_to_num(proj, nan=-1e6, posinf=-1e6, neginf=-1e6)
                pts = proj.astype(np.int32)
                if pts.size == 0:
                    continue

                if layer.get("mode") == "heat":
                    colors = _color_map(values)
                    for (u, v), color, value in zip(pts, colors, values):
                        if 0 <= u < width and 0 <= v < height:
                            radius = max(args.point_radius_min, int(args.point_radius_min + value * (args.point_radius_max - args.point_radius_min)))
                            cv2.circle(overlay, (u, v), radius, tuple(int(c) for c in color), thickness=-1)
                else:
                    base_color = layer["color"]
                    for (u, v), value in zip(pts, values):
                        if 0 <= u < width and 0 <= v < height:
                            radius = max(args.point_radius_min, int(args.point_radius_min + value * (args.point_radius_max - args.point_radius_min)))
                            intensity = 0.4 + 0.6 * value
                            color = tuple(
                                min(255, int(base_color[i] * intensity + 10)) for i in range(3)
                            )
                            cv2.circle(overlay, (u, v), radius, color, thickness=-1)
            frame = cv2.addWeighted(overlay, args.alpha, frame, 1.0 - args.alpha, 0)
            if args.legend and component_layers:
                legend_items = [
                    ("Obstacle", component_colors["obstacle"]),
                    ("Trip", component_colors["trip"]),
                    ("Slip", component_colors["slip"]),
                    ("Turn", component_colors["turn"]),
                    ("Physics", component_colors["physics"]),
                ]
                x0, y0 = 12, 18
                line_h = 18
                for i, (label, color) in enumerate(legend_items):
                    y = y0 + i * line_h
                    cv2.rectangle(frame, (x0, y - 12), (x0 + 12, y), color, thickness=-1)
                    cv2.putText(
                        frame,
                        label,
                        (x0 + 18, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Overlay video written to: {args.output}")


if __name__ == "__main__":
    main()
