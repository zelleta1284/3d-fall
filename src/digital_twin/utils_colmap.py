import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class Image:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    # COLMAP format: [qw, qx, qy, qz]
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_cameras_txt(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_txt(path: Path) -> List[Image]:
    images: List[Image] = []
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]
            images.append(
                Image(
                    image_id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=name,
                )
            )
            # Skip the next line containing 2D-3D matches
            f.readline()
    return images


def colmap_image_pose_to_extrinsic(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # COLMAP gives world-to-camera: x_cam = R * x_world + t
    # Open3D expects camera-to-world for integration.
    R = qvec2rotmat(qvec)
    t = tvec.reshape(3, 1)
    R_cw = R.T
    t_cw = -R.T @ t
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R_cw
    extrinsic[:3, 3] = t_cw[:, 0]
    return extrinsic


def intrinsics_from_camera(camera: Camera) -> Tuple[np.ndarray, int, int]:
    # Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, OPENCV
    width = camera.width
    height = camera.height
    params = camera.params
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif camera.model in {"PINHOLE", "OPENCV"}:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        # Best effort fallback
        f = params[0]
        cx = params[1] if len(params) > 1 else width / 2
        cy = params[2] if len(params) > 2 else height / 2
        fx = fy = f
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return K, width, height
