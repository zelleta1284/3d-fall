import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .simulate_risk import build_grids, load_mesh_vertices
from .utils_colmap import (
    colmap_image_pose_to_extrinsic,
    intrinsics_from_camera,
    read_cameras_txt,
    read_images_txt,
)


@dataclass
class SemanticConfig:
    score_threshold: float = 0.55
    mask_threshold: float = 0.5
    frame_stride: int = 3
    pixel_stride: int = 6
    low_profile_height_m: float = 0.12


HAZARD_MAP: Dict[str, Dict[str, float]] = {
    "couch": {"obstacle": 1.0, "turn": 0.2},
    "chair": {"obstacle": 0.8, "trip": 0.4},
    "dining table": {"obstacle": 1.0, "turn": 0.2},
    "bed": {"obstacle": 1.0, "turn": 0.2},
    "tv": {"obstacle": 0.6},
    "potted plant": {"obstacle": 0.5, "trip": 0.3},
    "toilet": {"obstacle": 0.9, "slip": 0.2},
    "sink": {"obstacle": 0.6, "slip": 0.2},
    "refrigerator": {"obstacle": 1.0},
    "microwave": {"obstacle": 0.4},
    "oven": {"obstacle": 0.7},
    "book": {"trip": 0.4},
    "backpack": {"trip": 0.7},
    "handbag": {"trip": 0.6},
    "suitcase": {"trip": 0.7},
    "sports ball": {"trip": 0.5},
    "vase": {"trip": 0.4, "obstacle": 0.3},
    "bottle": {"trip": 0.3},
    "cup": {"trip": 0.2},
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
    import torch

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

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print("Semantic hazards skipped: no frames")
        return None

    device = _select_device()
    model, weights = _load_model(device)
    preprocess = weights.transforms()
    categories: List[str] = list(weights.meta.get("categories", []))

    vertices = load_mesh_vertices(mesh_path)
    obstacle_grid, _, min_xy, floor_z = build_grids(vertices, grid_size, obstacle_height_m)
    grid_shape = obstacle_grid.shape

    hazard_grids = {
        "obstacle": np.zeros(grid_shape, dtype=np.float32),
        "trip": np.zeros(grid_shape, dtype=np.float32),
        "slip": np.zeros(grid_shape, dtype=np.float32),
        "turn": np.zeros(grid_shape, dtype=np.float32),
    }

    label_counts: Dict[str, int] = {}
    processed = 0

    import torch
    from torchvision.io import read_image

    for idx, frame_path in enumerate(frame_paths):
        if idx % max(config.frame_stride, 1) != 0:
            continue
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

        processed += 1

    for key in hazard_grids:
        hazard_grids[key] = _normalize_grid(hazard_grids[key])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **hazard_grids)

    if summary_path:
        summary = {
            "frames_processed": processed,
            "labels_detected": label_counts,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return out_path
