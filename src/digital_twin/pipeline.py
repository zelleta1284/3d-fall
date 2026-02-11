import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class ColmapPaths:
    base: Path
    database: Path
    images: Path
    sparse: Path
    sparse_txt: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, frames_dir: Path, fps: float = 2.0) -> List[Path]:
    ensure_dir(frames_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(video_fps / fps)), 1)

    paths: List[Path] = []
    idx = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = frames_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            paths.append(out_path)
            idx += 1
        frame_idx += 1

    cap.release()
    if not paths:
        raise RuntimeError("No frames extracted; check input video and fps.")
    return paths


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_colmap(frames_dir: Path, colmap_dir: Path, single_camera: bool = True) -> ColmapPaths:
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP binary not found in PATH. Install COLMAP or skip --run-colmap.")

    ensure_dir(colmap_dir)
    database = colmap_dir / "database.db"
    images = colmap_dir / "images"
    sparse = colmap_dir / "sparse"
    ensure_dir(images)
    ensure_dir(sparse)

    # Copy/links frames into colmap images dir
    for p in frames_dir.glob("*.jpg"):
        target = images / p.name
        if not target.exists():
            shutil.copy2(p, target)

    _run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            "1" if single_camera else "0",
        ]
    )
    _run(["colmap", "exhaustive_matcher", "--database_path", str(database)])
    _run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ]
    )

    # Convert to TXT for easier parsing
    sparse_txt = colmap_dir / "sparse_txt"
    ensure_dir(sparse_txt)
    model_path = sparse / "0"
    _run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(model_path),
            "--output_path",
            str(sparse_txt),
            "--output_type",
            "TXT",
        ]
    )

    return ColmapPaths(base=colmap_dir, database=database, images=images, sparse=sparse, sparse_txt=sparse_txt)


def estimate_depth_midas(
    frames_dir: Path,
    depth_dir: Path,
    device: Optional[str] = None,
    median_depth_m: float = 2.5,
) -> Dict[str, float]:
    import torch

    ensure_dir(depth_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    scales: Dict[str, float] = {}
    image_paths = sorted(frames_dir.glob("*.jpg"))

    for img_path in tqdm(image_paths, desc="Depth"):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        input_batch = transform(img_np).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)
        med = float(np.median(depth)) if np.isfinite(depth).all() else 1.0
        scale = median_depth_m / max(med, 1e-6)
        depth_m = depth * scale

        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        out_path = depth_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), depth_mm)
        scales[img_path.name] = scale

    with (depth_dir / "depth_scales.json").open("w", encoding="utf-8") as f:
        json.dump(scales, f, indent=2)
    return scales


def load_depth_image(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth: {depth_path}")
    depth_m = depth.astype(np.float32) / 1000.0
    return depth_m


def fuse_tsdf(
    colmap_txt_dir: Path,
    images_dir: Path,
    depth_dir: Path,
    mesh_out: Path,
    voxel_length: float = 0.02,
    sdf_trunc: float = 0.04,
    max_frames: Optional[int] = None,
) -> None:
    import open3d as o3d

    from .utils_colmap import (
        colmap_image_pose_to_extrinsic,
        intrinsics_from_camera,
        read_cameras_txt,
        read_images_txt,
    )

    cameras = read_cameras_txt(colmap_txt_dir / "cameras.txt")
    images = read_images_txt(colmap_txt_dir / "images.txt")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    count = 0
    for img in tqdm(images, desc="Integrate"):
        color_path = images_dir / img.name
        depth_path = depth_dir / f"{Path(img.name).stem}.png"
        if not color_path.exists() or not depth_path.exists():
            continue

        camera = cameras[img.camera_id]
        K, width, height = intrinsics_from_camera(camera)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        color = o3d.io.read_image(str(color_path))
        depth = o3d.io.read_image(str(depth_path))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )

        extrinsic = colmap_image_pose_to_extrinsic(img.qvec, img.tvec)
        volume.integrate(rgbd, intrinsic, extrinsic)

        count += 1
        if max_frames and count >= max_frames:
            break

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    ensure_dir(mesh_out.parent)
    o3d.io.write_triangle_mesh(str(mesh_out), mesh)


def run_pipeline(
    video_path: Path,
    work_dir: Path,
    fps: float = 2.0,
    median_depth_m: float = 2.5,
    run_colmap_flag: bool = True,
) -> Path:
    ensure_dir(work_dir)
    frames_dir = work_dir / "frames"
    depth_dir = work_dir / "depth"
    colmap_dir = work_dir / "colmap"
    mesh_out = work_dir / "mesh.ply"

    extract_frames(video_path, frames_dir, fps=fps)

    colmap_paths = None
    if run_colmap_flag:
        colmap_paths = run_colmap(frames_dir, colmap_dir)

    estimate_depth_midas(frames_dir, depth_dir, device=None, median_depth_m=median_depth_m)

    if not colmap_paths:
        raise RuntimeError("COLMAP step was skipped; cannot fuse without camera poses.")

    fuse_tsdf(
        colmap_paths.sparse_txt,
        colmap_paths.images,
        depth_dir,
        mesh_out,
    )

    return mesh_out
