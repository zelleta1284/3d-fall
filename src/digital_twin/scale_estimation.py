from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ScalePrior:
    name: str
    height_m: float
    tol_m: float
    weight: float


DEFAULT_PRIORS = [
    ScalePrior("ceiling", 2.6, 0.3, 2.0),
    ScalePrior("door", 2.03, 0.2, 1.5),
    ScalePrior("counter", 0.91, 0.15, 1.2),
    ScalePrior("table", 0.75, 0.12, 1.0),
    ScalePrior("bed", 0.5, 0.18, 1.0),
    ScalePrior("sofa_seat", 0.45, 0.12, 0.8),
    ScalePrior("window_sill", 1.0, 0.2, 0.8),
    ScalePrior("chair_seat", 0.45, 0.12, 0.6),
    ScalePrior("toilet_seat", 0.43, 0.1, 0.6),
    ScalePrior("tub_rim", 0.55, 0.12, 0.5),
    ScalePrior("coffee_table", 0.45, 0.1, 0.4),
    ScalePrior("light_switch", 1.2, 0.2, 0.3),
]


def _load_ply_ascii(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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


def load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            raise RuntimeError("Mesh is empty")
        return np.asarray(mesh.vertices)
    except Exception:
        vertices, _ = _load_ply_ascii(mesh_path)
        return vertices


def estimate_scale_from_mesh(
    vertices: np.ndarray,
    priors: List[ScalePrior] = None,
    ceiling_target_m: float = 2.6,
) -> Dict:
    if priors is None:
        priors = DEFAULT_PRIORS

    z = vertices[:, 2]
    floor_z = float(np.percentile(z, 2))
    ceil_z = float(np.percentile(z, 98))
    height_span = max(ceil_z - floor_z, 1e-6)

    heights = z - floor_z
    heights = heights[heights > 0.05]
    if heights.size == 0:
        scale = ceiling_target_m / height_span
        return {
            "scale": scale,
            "method": "ceiling_fallback",
            "score": 0.0,
            "height_span": height_span,
        }

    hist, edges = np.histogram(heights, bins=80)
    max_count = hist.max() if hist.size else 0
    peak_bins = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > max_count * 0.1:
            peak_bins.append(i)

    peaks = []
    for i in peak_bins:
        peak_h = (edges[i] + edges[i + 1]) / 2
        peaks.append(float(peak_h))
    if not peaks:
        scale = ceiling_target_m / height_span
        return {
            "scale": scale,
            "method": "ceiling_fallback",
            "score": 0.0,
            "height_span": height_span,
        }

    candidates = []
    for h in peaks:
        for prior in priors:
            if h > 1e-6:
                candidates.append(prior.height_m / h)

    best = None
    best_score = -1.0
    for s in candidates:
        score = 0.0
        matched = []
        for h in peaks:
            scaled = h * s
            for prior in priors:
                if abs(scaled - prior.height_m) <= prior.tol_m:
                    score += prior.weight
                    matched.append(prior.name)
        # bonus for overall ceiling height within plausible range
        scaled_height = height_span * s
        if 2.3 <= scaled_height <= 3.0:
            score += 1.0
        if score > best_score:
            best_score = score
            best = (s, matched, scaled_height)

    if best is None:
        scale = ceiling_target_m / height_span
        return {
            "scale": scale,
            "method": "ceiling_fallback",
            "score": 0.0,
            "height_span": height_span,
        }

    scale, matched, scaled_height = best
    return {
        "scale": scale,
        "method": "prior_fit",
        "score": float(best_score),
        "height_span": height_span,
        "scaled_height": float(scaled_height),
        "matched_priors": list(dict.fromkeys(matched)),
    }


def scale_mesh(mesh_path: Path, out_path: Path, scale: float) -> None:
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            raise RuntimeError("Mesh is empty")
        mesh.scale(scale, center=mesh.get_center())
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        return
    except Exception:
        vertices, faces = _load_ply_ascii(mesh_path)
        vertices *= float(scale)
        _write_ply_ascii(out_path, vertices, faces)


def _write_ply_ascii(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
