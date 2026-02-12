import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class FloorMaterialConfig:
    labels: Tuple[str, ...] = (
        "polished hardwood",
        "matte hardwood",
        "tile",
        "vinyl",
        "carpet",
        "concrete",
    )
    friction_map: Dict[str, float] = None
    frame_stride: int = 4
    pixel_stride: int = 8
    floor_height_min_m: float = -0.01
    floor_height_max_m: float = 0.03
    patch_size: int = 192
    patch_count: int = 40
    min_confidence: float = 0.0
    gloss_threshold: float = 0.05
    gloss_boost: float = 0.1

    def __post_init__(self) -> None:
        if self.friction_map is None:
            self.friction_map = {
                "polished hardwood": 0.35,
                "matte hardwood": 0.4,
                "tile": 0.32,
                "vinyl": 0.4,
                "carpet": 0.7,
                "concrete": 0.5,
            }


def _select_device() -> str:
    import os
    import torch

    forced = os.getenv("SURESTEP_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_colmap_txt(workdir: Path) -> Optional[Path]:
    candidates = [
        workdir / "colmap_txt",
        workdir / "colmap" / "sparse_txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _sample_floor_points(
    depth_m: np.ndarray,
    camera: object,
    cam_params: Dict[int, Dict[str, np.ndarray]],
    floor_z: float,
    config: FloorMaterialConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    cam_id = getattr(camera, "camera_id", None)
    qvec = getattr(camera, "qvec", None)
    tvec = getattr(camera, "tvec", None)
    if cam_id is None or qvec is None or tvec is None:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

    cam = cam_params.get(cam_id)
    if cam is None:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

    K, _, _ = intrinsics_from_camera(cam)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    extrinsic = colmap_image_pose_to_extrinsic(qvec, tvec)

    ys, xs = np.where(depth_m > 0)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)
    if config.pixel_stride > 1:
        ys = ys[:: config.pixel_stride]
        xs = xs[:: config.pixel_stride]

    z = depth_m[ys, xs]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)
    pts_world = (extrinsic[:3, :3] @ pts_cam.T).T + extrinsic[:3, 3]

    heights = pts_world[:, 2] - floor_z
    floor_mask = (heights >= config.floor_height_min_m) & (heights <= config.floor_height_max_m)
    if not np.any(floor_mask):
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

    pix = np.stack([xs[floor_mask], ys[floor_mask]], axis=1).astype(np.int32)
    return pix, pts_world[floor_mask]


def _collect_patches(
    frame: np.ndarray,
    floor_pixels: np.ndarray,
    config: FloorMaterialConfig,
) -> List[np.ndarray]:
    if floor_pixels.size == 0:
        return []

    h, w = frame.shape[:2]
    half = config.patch_size // 2
    patches: List[np.ndarray] = []

    for x, y in floor_pixels:
        if len(patches) >= config.patch_count:
            break
        x0 = int(x) - half
        y0 = int(y) - half
        x1 = x0 + config.patch_size
        y1 = y0 + config.patch_size
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        patches.append(patch)
    return patches


def infer_floor_material(
    workdir: Path,
    mesh_path: Path,
    obstacle_height_m: float,
    out_path: Path,
    config: Optional[FloorMaterialConfig] = None,
) -> Optional[Dict[str, object]]:
    if config is None:
        config = FloorMaterialConfig()

    colmap_txt = resolve_colmap_txt(workdir)
    if colmap_txt is None:
        print("Floor material inference skipped: no colmap_txt found")
        return None

    frames_dir = workdir / "frames"
    depth_dir = workdir / "depth"
    if not frames_dir.exists() or not depth_dir.exists():
        print("Floor material inference skipped: missing frames/depth")
        return None

    cameras = read_cameras_txt(colmap_txt / "cameras.txt")
    images = read_images_txt(colmap_txt / "images.txt")
    images_by_name = {img.name: img for img in images}

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print("Floor material inference skipped: no frames")
        return None

    vertices = load_mesh_vertices(mesh_path)
    _, _, _, floor_z = build_grids(vertices, 0.05, obstacle_height_m)

    patches: List[np.ndarray] = []

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

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth_m = depth.astype(np.float32) / 1000.0

        pix, _ = _sample_floor_points(depth_m, image, cameras, floor_z, config)
        if pix.size == 0:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patches.extend(_collect_patches(frame_rgb, pix, config))
        if len(patches) >= config.patch_count:
            break

    if not patches:
        print("Floor material inference skipped: no floor patches sampled")
        return None

    try:
        import torch
        import open_clip
    except Exception as exc:
        print(f"Floor material inference skipped: {exc}")
        return None

    device = _select_device()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompt_templates = [
        "a photo of a {} floor",
        "a close-up photo of {} floor",
        "a photo of {} flooring",
        "a texture photo of {} floor",
        "a photo of {} surface",
    ]

    from PIL import Image

    images = [preprocess(Image.fromarray(p)) for p in patches]
    image_input = torch.stack(images).to(device)
    prompts = []
    label_slices = []
    for label in config.labels:
        start = len(prompts)
        prompts.extend([tpl.format(label) for tpl in prompt_templates])
        label_slices.append((start, len(prompts)))
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Average prompt embeddings per label
        label_features = []
        for start, end in label_slices:
            emb = text_features[start:end].mean(dim=0)
            emb = emb / emb.norm()
            label_features.append(emb)
        label_features = torch.stack(label_features, dim=0)
        logits = image_features @ label_features.T
        probs = logits.softmax(dim=-1)

    mean_probs = probs.mean(dim=0).cpu().numpy()

    # Simple gloss heuristic: bright, low-saturation highlights often indicate polished surfaces.
    gloss_ratios = []
    for patch in patches:
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        gloss = np.mean((v > 0.9) & (s < 0.25))
        gloss_ratios.append(gloss)
    gloss_ratio = float(np.mean(gloss_ratios)) if gloss_ratios else 0.0

    if gloss_ratio > config.gloss_threshold:
        boost = config.gloss_boost * min(2.0, gloss_ratio / max(config.gloss_threshold, 1e-6))
        labels = list(config.labels)
        if "polished hardwood" in labels:
            idx = labels.index("polished hardwood")
            mean_probs[idx] += boost
            mean_probs = mean_probs / max(mean_probs.sum(), 1e-6)
    top_idx = int(mean_probs.argmax())
    label = config.labels[top_idx]
    confidence = float(mean_probs[top_idx])
    friction = float(config.friction_map.get(label, 0.5))
    applied = confidence >= config.min_confidence

    result = {
        "label": label,
        "confidence": confidence,
        "applied": applied,
        "friction": friction,
        "labels": list(config.labels),
        "probabilities": [float(v) for v in mean_probs.tolist()],
        "patches_used": len(patches),
        "floor_height_min_m": config.floor_height_min_m,
        "floor_height_max_m": config.floor_height_max_m,
        "gloss_ratio": gloss_ratio,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
