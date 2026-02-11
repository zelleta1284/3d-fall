from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class WindowCandidate:
    center: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    width: float
    height: float
    area: float


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    return v / n


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = _normalize(normal)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(up, normal)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent = np.cross(up, normal)
    tangent = _normalize(tangent)
    bitangent = np.cross(normal, tangent)
    return tangent, bitangent


def detect_window_planes(
    mesh_path: Path,
    min_area: float = 0.4,
    max_area: float = 6.0,
    vertical_tol: float = 0.25,
    max_planes: int = 6,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 2000,
) -> List[WindowCandidate]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty; cannot detect windows.")

    pcd = mesh.sample_points_uniformly(number_of_points=80000)
    candidates: List[WindowCandidate] = []

    for _ in range(max_planes):
        if len(pcd.points) < 1000:
            break
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(inliers) < 1000:
            break

        inlier_cloud = pcd.select_by_index(inliers)
        pcd = pcd.select_by_index(inliers, invert=True)

        normal = np.array(plane_model[:3], dtype=np.float32)
        normal = _normalize(normal)
        # vertical plane -> normal has small z component
        if abs(normal[2]) > vertical_tol:
            continue

        points = np.asarray(inlier_cloud.points)
        tangent, bitangent = _plane_basis(normal)
        proj_u = points @ tangent
        proj_v = points @ bitangent
        min_u, max_u = float(proj_u.min()), float(proj_u.max())
        min_v, max_v = float(proj_v.min()), float(proj_v.max())
        width = max_u - min_u
        height = max_v - min_v
        area = width * height

        if area < min_area or area > max_area:
            continue

        center = points.mean(axis=0)
        candidates.append(
            WindowCandidate(
                center=(float(center[0]), float(center[1]), float(center[2])),
                normal=(float(normal[0]), float(normal[1]), float(normal[2])),
                width=float(width),
                height=float(height),
                area=float(area),
            )
        )

    return candidates
