import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .simulate_risk import build_grids, load_mesh_vertices
from .utils_colmap import (
    colmap_image_pose_to_extrinsic,
    intrinsics_from_camera,
    read_cameras_txt,
    read_images_txt,
)


@dataclass
class SemanticConfig:
    score_threshold: float = 0.45
    mask_threshold: float = 0.4
    frame_stride: int = 2
    pixel_stride: int = 4
    low_profile_height_m: float = 0.12
    rug_min_height_m: float = 0.01
    rug_max_height_m: float = 0.08
    rug_gradient_max_m: float = 0.04
    rug_weight: float = 0.75
    rug_slip_weight: float = 0.6
    table_min_height_m: float = 0.35
    table_max_height_m: float = 0.9
    table_slope_max: float = 0.06
    table_weight: float = 0.5
    rug_texture_quantile: float = 0.85
    small_object_area_ratio: float = 0.015
    small_object_trip_boost: float = 1.4


HAZARD_MAP: Dict[str, Dict[str, float]] = {
    # Bedroom / living room
    "couch": {"obstacle": 1.0, "turn": 0.2},
    "chair": {"obstacle": 0.8, "trip": 0.5, "turn": 0.1},
    "dining table": {"obstacle": 1.0, "turn": 0.2, "trip": 0.4},
    "bed": {"obstacle": 1.0, "turn": 0.2},
    "bench": {"obstacle": 0.7, "trip": 0.3},
    "tv": {"obstacle": 0.6},
    "laptop": {"trip": 0.25},
    "mouse": {"trip": 0.2},
    "remote": {"trip": 0.3},
    "keyboard": {"trip": 0.3},
    "cell phone": {"trip": 0.25},
    "potted plant": {"obstacle": 0.6, "trip": 0.4},
    "book": {"trip": 0.4},
    "clock": {"trip": 0.2},
    "vase": {"trip": 0.4, "obstacle": 0.3},
    "teddy bear": {"trip": 0.3},
    # Bathroom
    "toilet": {"obstacle": 0.9, "slip": 0.2},
    "sink": {"obstacle": 0.6, "slip": 0.2},
    "hair drier": {"trip": 0.2},
    "toothbrush": {"trip": 0.2},
    # Kitchen
    "refrigerator": {"obstacle": 1.0},
    "microwave": {"obstacle": 0.4},
    "oven": {"obstacle": 0.7},
    "toaster": {"obstacle": 0.3},
    "knife": {"trip": 0.2},
    "fork": {"trip": 0.2},
    "spoon": {"trip": 0.2},
    "bottle": {"trip": 0.3},
    "wine glass": {"trip": 0.25},
    "cup": {"trip": 0.25},
    "bowl": {"trip": 0.25},
    "banana": {"slip": 0.2, "trip": 0.1},
    "apple": {"trip": 0.15},
    "orange": {"trip": 0.15},
    "sandwich": {"trip": 0.1},
    "pizza": {"trip": 0.1},
    "donut": {"trip": 0.1},
    "backpack": {"trip": 0.8},
    "handbag": {"trip": 0.7},
    "suitcase": {"trip": 0.8},
    "umbrella": {"trip": 0.4},
    # Hallway / entry / general clutter
    "skis": {"trip": 0.6},
    "snowboard": {"trip": 0.6},
    "sports ball": {"trip": 0.6},
    "frisbee": {"trip": 0.5},
    "baseball bat": {"trip": 0.5},
    "baseball glove": {"trip": 0.4},
    "skateboard": {"trip": 0.6},
    "surfboard": {"trip": 0.6},
    "tennis racket": {"trip": 0.4},
    "dog": {"trip": 0.7, "obstacle": 0.3},
    "cat": {"trip": 0.6, "obstacle": 0.25},
    "scissors": {"trip": 0.2},
}


def resolve_colmap_txt(workdir: Path) -> Optional[Path]:
    candidates = [
        workdir / "colmap_txt",
        workdir / "colmap" / "sparse_txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_scale_transform(workdir: Path) -> Tuple[Optional[float], Optional[np.ndarray]]:
    config_path = workdir / "config_used.yaml"
    if not config_path.exists():
        return None, None
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None, None
    scale_result = cfg.get("scale_result") or {}
    try:
        scale = float(scale_result.get("scale", 1.0))
    except (TypeError, ValueError):
        return None, None
    if scale == 1.0:
        return None, None
    unscaled_mesh = workdir / "mesh.ply"
    if not unscaled_mesh.exists():
        return scale, None
    vertices = load_mesh_vertices(unscaled_mesh)
    if vertices.size == 0:
        return scale, None
    center = np.mean(vertices, axis=0)
    return scale, center


def _load_model(device: str):
    import torch
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()
    return model, weights


def _select_device() -> str:
    import os
    import torch

    forced = os.getenv("SURESTEP_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mask_points(mask: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return ys, xs
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]
    return ys, xs


def _depth_gradient(depth_m: np.ndarray) -> np.ndarray:
    if depth_m.size == 0:
        return depth_m
    depth_blur = cv2.GaussianBlur(depth_m, (5, 5), 0)
    grad_x = cv2.Sobel(depth_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_blur, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def _accumulate_hazards(
    hazard_grids: Dict[str, np.ndarray],
    weights: Dict[str, float],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score: float,
    low_mask: np.ndarray,
) -> None:
    for hazard, weight in weights.items():
        if hazard == "trip":
            idx = low_mask
        elif hazard == "obstacle":
            idx = ~low_mask
        else:
            idx = np.ones_like(low_mask, dtype=bool)

        if not np.any(idx):
            continue
        np.add.at(hazard_grids[hazard], (grid_y[idx], grid_x[idx]), weight * score)


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    if not np.any(grid):
        return grid
    cap = np.percentile(grid, 99)
    if cap <= 0:
        return grid
    return np.clip(grid / cap, 0.0, 1.0)


def compute_semantic_hazards(
    workdir: Path,
    mesh_path: Path,
    grid_size: float,
    obstacle_height_m: float,
    out_path: Path,
    summary_path: Optional[Path] = None,
    config: Optional[SemanticConfig] = None,
    rug_only: bool = False,
    max_frames: Optional[int] = None,
) -> Optional[Path]:
    if config is None:
        config = SemanticConfig()

    colmap_txt = resolve_colmap_txt(workdir)
    if colmap_txt is None:
        print("Semantic hazards skipped: no colmap_txt found")
        return None

    frames_dir = workdir / "frames"
    depth_dir = workdir / "depth"
    if not frames_dir.exists() or not depth_dir.exists():
        print("Semantic hazards skipped: missing frames/depth")
        return None

    cameras = read_cameras_txt(colmap_txt / "cameras.txt")
    images = read_images_txt(colmap_txt / "images.txt")
    images_by_name = {img.name: img for img in images}

    scale, center = _load_scale_transform(workdir)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print("Semantic hazards skipped: no frames")
        return None

    if not rug_only:
        device = _select_device()
        model, weights = _load_model(device)
        preprocess = weights.transforms()
        categories: List[str] = list(weights.meta.get("categories", []))
    else:
        model = None
        preprocess = None
        categories = []

    vertices = load_mesh_vertices(mesh_path)
    obstacle_grid, height_grid, min_xy, floor_z = build_grids(vertices, grid_size, obstacle_height_m)
    grid_shape = obstacle_grid.shape

    hazard_grids = {
        "obstacle": np.zeros(grid_shape, dtype=np.float32),
        "trip": np.zeros(grid_shape, dtype=np.float32),
        "slip": np.zeros(grid_shape, dtype=np.float32),
        "turn": np.zeros(grid_shape, dtype=np.float32),
    }

    # Fast surface heuristics from mesh heights (tables/rugs).
    height_diff = height_grid - floor_z
    grad_y, grad_x = np.gradient(height_diff)
    slope = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    table_mask = (
        (height_diff >= config.table_min_height_m)
        & (height_diff <= config.table_max_height_m)
        & (slope <= config.table_slope_max)
    )
    if np.any(table_mask):
        hazard_grids["trip"][table_mask] += config.table_weight

    rug_mesh_mask = (
        (height_diff >= config.rug_min_height_m)
        & (height_diff <= config.rug_max_height_m)
        & (slope <= config.rug_gradient_max_m)
    )
    if np.any(rug_mesh_mask):
        hazard_grids["trip"][rug_mesh_mask] += config.rug_weight
        hazard_grids["slip"][rug_mesh_mask] += config.rug_slip_weight

    label_counts: Dict[str, int] = {}
    processed = 0
    rug_points = 0

    import torch
    from torchvision.io import read_image

    for idx, frame_path in enumerate(frame_paths):
        if idx % max(config.frame_stride, 1) != 0:
            continue
        if max_frames is not None and processed >= max_frames:
            break
        image_name = frame_path.name
        image = images_by_name.get(image_name)
        if image is None:
            continue
        depth_path = depth_dir / f"{frame_path.stem}.png"
        if not depth_path.exists():
            continue

        cam = cameras.get(image.camera_id)
        if cam is None:
            continue

        K, _, _ = intrinsics_from_camera(cam)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        extrinsic = colmap_image_pose_to_extrinsic(image.qvec, image.tvec)

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth_m = depth.astype(np.float32) / 1000.0
        grad = _depth_gradient(depth_m)

        if not rug_only:
            with torch.no_grad():
                img_tensor = read_image(str(frame_path))
                inputs = [preprocess(img_tensor).to(device)]
                outputs = model(inputs)[0]

            scores = outputs["scores"].detach().cpu().numpy()
            labels = outputs["labels"].detach().cpu().numpy()
            masks = outputs["masks"].detach().cpu().numpy()[:, 0]

            for score, label_id, mask in zip(scores, labels, masks):
                if score < config.score_threshold:
                    continue
                if label_id >= len(categories):
                    continue
                label = categories[int(label_id)]
                hazard_weights = HAZARD_MAP.get(label)
                if not hazard_weights:
                    continue

                label_counts[label] = label_counts.get(label, 0) + 1
                mask_bin = mask > config.mask_threshold
                ys, xs = _mask_points(mask_bin, config.pixel_stride)
                if ys.size == 0:
                    continue
                area_ratio = float(np.count_nonzero(mask_bin)) / float(mask_bin.size)
                if area_ratio < config.small_object_area_ratio and "trip" in hazard_weights:
                    hazard_weights = dict(hazard_weights)
                    hazard_weights["trip"] *= config.small_object_trip_boost

                z = depth_m[ys, xs]
                valid = z > 0
                if not np.any(valid):
                    continue
                ys = ys[valid]
                xs = xs[valid]
                z = z[valid]

                x_cam = (xs - cx) * z / fx
                y_cam = (ys - cy) * z / fy
                pts_cam = np.stack([x_cam, y_cam, z], axis=1)

                pts_world = (extrinsic[:3, :3] @ pts_cam.T).T + extrinsic[:3, 3]
                if scale is not None and center is not None:
                    pts_world = center + (pts_world - center) * scale

                grid_x = ((pts_world[:, 0] - min_xy[0]) / grid_size).astype(int)
                grid_y = ((pts_world[:, 1] - min_xy[1]) / grid_size).astype(int)
                valid_grid = (
                    (grid_x >= 0)
                    & (grid_y >= 0)
                    & (grid_x < grid_shape[1])
                    & (grid_y < grid_shape[0])
                )
                if not np.any(valid_grid):
                    continue

                grid_x = grid_x[valid_grid]
                grid_y = grid_y[valid_grid]
                z_world = pts_world[:, 2][valid_grid]
                low_mask = z_world <= (floor_z + max(config.low_profile_height_m, obstacle_height_m))

                _accumulate_hazards(
                    hazard_grids,
                    hazard_weights,
                    grid_x,
                    grid_y,
                    float(score),
                    low_mask,
                )

        # Rug-like heuristic: low-profile, flat, textured surfaces slightly above the floor
        ys, xs = _mask_points(depth_m > 0, config.pixel_stride)
        if ys.size:
            z = depth_m[ys, xs]
            x_cam = (xs - cx) * z / fx
            y_cam = (ys - cy) * z / fy
            pts_cam = np.stack([x_cam, y_cam, z], axis=1)
            pts_world = (extrinsic[:3, :3] @ pts_cam.T).T + extrinsic[:3, 3]
            if scale is not None and center is not None:
                pts_world = center + (pts_world - center) * scale

            heights = pts_world[:, 2] - floor_z
            flat = grad[ys, xs] <= config.rug_gradient_max_m
            rug_mask = (
                (heights >= config.rug_min_height_m)
                & (heights <= config.rug_max_height_m)
                & flat
            )

            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is not None and np.any(rug_mask):
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
                tex_vals = texture[ys, xs]
                thresh = np.quantile(tex_vals[rug_mask], config.rug_texture_quantile)
                rug_mask = rug_mask & (tex_vals >= thresh)

            if np.any(rug_mask):
                grid_x = ((pts_world[:, 0] - min_xy[0]) / grid_size).astype(int)
                grid_y = ((pts_world[:, 1] - min_xy[1]) / grid_size).astype(int)
                valid_grid = (
                    (grid_x >= 0)
                    & (grid_y >= 0)
                    & (grid_x < grid_shape[1])
                    & (grid_y < grid_shape[0])
                )
                valid_mask = rug_mask & valid_grid
                if np.any(valid_mask):
                    np.add.at(
                        hazard_grids["trip"],
                        (grid_y[valid_mask], grid_x[valid_mask]),
                        config.rug_weight,
                    )
                    np.add.at(
                        hazard_grids["slip"],
                        (grid_y[valid_mask], grid_x[valid_mask]),
                        config.rug_slip_weight,
                    )
                    rug_points += int(np.count_nonzero(valid_mask))

        processed += 1

    for key in hazard_grids:
        hazard_grids[key] = _normalize_grid(hazard_grids[key])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **hazard_grids)

    if summary_path:
        summary = {
            "frames_processed": processed,
            "labels_detected": label_counts,
            "rug_like_points": rug_points,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return out_path
