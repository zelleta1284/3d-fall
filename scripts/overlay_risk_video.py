#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
import torch
import torchvision
import torchvision.transforms.functional as TF

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


def _apply_cinematic_grade(frame: np.ndarray, vignette: np.ndarray, contrast: float, saturation: float) -> np.ndarray:
    graded = frame.astype(np.float32) / 255.0
    graded = np.clip((graded - 0.5) * contrast + 0.5, 0.0, 1.0)
    hsv = cv2.cvtColor((graded * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    graded = np.clip(graded * vignette, 0.0, 1.0)
    return (graded * 255).astype(np.uint8)


def _build_vignette(width: int, height: int, strength: float) -> np.ndarray:
    if strength <= 0:
        return np.ones((height, width, 1), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    vignette = 1.0 - strength * np.clip(radius, 0.0, 1.0)
    return vignette[..., None].astype(np.float32)


def _render_minimap(
    trip: Optional[np.ndarray],
    slip: Optional[np.ndarray],
    heat: Optional[np.ndarray],
    size: int,
    colors: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    if trip is None and slip is None and heat is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    base = np.zeros_like(trip if trip is not None else slip if slip is not None else heat, dtype=np.float32)
    if trip is not None:
        t = trip.astype(np.float32)
        if t.max() > 0:
            base = np.maximum(base, t / t.max())
    if slip is not None:
        s = slip.astype(np.float32)
        if s.max() > 0:
            base = np.maximum(base, s / s.max())
    if heat is not None:
        h = heat.astype(np.float32)
        if h.max() > 0:
            base = np.maximum(base, h / h.max())
    base = np.clip(base, 0.0, 1.0)
    map_img = np.zeros(base.shape + (3,), dtype=np.float32)
    if trip is not None:
        t = trip.astype(np.float32)
        if t.max() > 0:
            map_img += (t / t.max())[..., None] * (np.array(colors["trip"], dtype=np.float32) / 255.0)
    if slip is not None:
        s = slip.astype(np.float32)
        if s.max() > 0:
            map_img += (s / s.max())[..., None] * (np.array(colors["slip"], dtype=np.float32) / 255.0)
    if heat is not None:
        h = heat.astype(np.float32)
        if h.max() > 0:
            heat_colors = plt.get_cmap("inferno")(np.clip(h / h.max(), 0, 1))[:, :, :3]
            map_img = np.maximum(map_img, heat_colors.astype(np.float32))
    map_img = np.clip(map_img, 0.0, 1.0)
    map_img = (map_img * 255).astype(np.uint8)
    map_img = cv2.resize(map_img, (size, size), interpolation=cv2.INTER_NEAREST)
    return map_img


_COCO_CLASSES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _load_detector() -> torch.nn.Module:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model


def _detect_objects(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    score_thresh: float,
    max_dets: int,
    class_filter: Optional[set],
) -> List[Dict[str, object]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(rgb)
    with torch.no_grad():
        outputs = model([tensor])[0]
    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    dets: List[Dict[str, object]] = []
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        name = _COCO_CLASSES[int(label)] if int(label) < len(_COCO_CLASSES) else str(label)
        if class_filter is not None and name not in class_filter:
            continue
        dets.append({"box": box.astype(int), "label": name, "score": float(score)})
        if len(dets) >= max_dets:
            break
    return dets


def _rug_heuristic(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    h, w = frame_bgr.shape[:2]
    y0 = int(h * 0.45)
    roi = frame_bgr[y0:h, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.dilate(edges, k, iterations=1)
    ys, xs = np.where(edges > 0)
    if ys.size < 300:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    # require a reasonably large region
    if (x_max - x_min) * (y_max - y_min) < 0.03 * (w * (h - y0)):
        return None
    return (x_min, y_min + y0, x_max, y_max + y0)


def _hardwood_heuristic(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    h, w = frame_bgr.shape[:2]
    y0 = int(h * 0.5)
    roi = frame_bgr[y0:h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    mask = (h_chan >= 8) & (h_chan <= 30) & (s_chan >= 40) & (v_chan >= 40)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if ratio < 0.12:
        return None
    return (0, y0, w - 1, h - 1)


def _draw_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label: str,
    alpha_fill: float = 0.12,
) -> None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(frame.shape[1] - 1, x1))
    x2 = max(0, min(frame.shape[1] - 1, x2))
    y1 = max(0, min(frame.shape[0] - 1, y1))
    y2 = max(0, min(frame.shape[0] - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha_fill, frame, 1.0 - alpha_fill, 0, dst=frame)
    # draw only corners to avoid full box look
    corner = max(8, int(0.08 * min(x2 - x1, y2 - y1)))
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 2)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 2)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, 2)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _ot_message_for(label: str) -> Optional[str]:
    label = label.lower()
    if label in {"table", "dining table"}:
        return "Table: pad edges/move."
    if label in {"obstacle"}:
        return "Clear walking path."
    if label in {"trip"}:
        return "Trip risk: clear clutter."
    if label in {"slip"}:
        return "Slip risk: add grip."
    return None


def _draw_ot_callouts(
    frame: np.ndarray,
    callouts: List[Tuple[str, Tuple[int, int]]],
    color: Tuple[int, int, int],
    max_items: int,
) -> None:
    if not callouts:
        return
    x0, y0 = 12, 30
    box_w, box_h = 240, 22
    shown = 0
    for text, anchor in callouts:
        if shown >= max_items:
            break
        y = y0 + shown * (box_h + 8)
        # Background box (black) + white text
        cv2.rectangle(frame, (x0, y - box_h + 2), (x0 + box_w, y + 2), color, thickness=-1)
        cv2.putText(frame, text, (x0 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Leader line to anchor point
        ax, ay = anchor
        ax = int(max(0, min(frame.shape[1] - 1, ax)))
        ay = int(max(0, min(frame.shape[0] - 1, ay)))
        cv2.line(frame, (x0 + box_w, y - 8), (ax, ay), color, 1, cv2.LINE_AA)
        shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay the risk heatmap on the original video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--workdir", required=True, help="Workdir that produced run_all outputs")
    parser.add_argument("--output", required=True, help="Path for overlay video")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate used by run_all")
    parser.add_argument("--alpha", type=float, default=1.0, help="Overlay alpha")
    parser.add_argument("--max-points", type=int, default=20000, help="Max grid points to project each frame")
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
    parser.add_argument("--depth-min-keep", type=float, default=0.1, help="Minimum fraction of points to keep after depth occlusion")
    parser.add_argument("--min-v-ratio", type=float, default=0.0, help="Only draw overlay below this vertical ratio")
    parser.add_argument("--shaded-only", dest="shaded_only", action="store_true", help="Render shaded regions only (no dots)")
    parser.add_argument("--no-shaded-only", dest="shaded_only", action="store_false", help="Render dots/points")
    parser.set_defaults(shaded_only=True)
    parser.add_argument("--fade-seconds", type=float, default=1.0, help="Overlay fade-in duration (seconds)")
    parser.add_argument("--split-seconds", type=float, default=2.0, help="Split-screen intro duration (seconds)")
    parser.add_argument("--minimap-size", type=int, default=180, help="Mini-map inset size (pixels)")
    parser.add_argument("--minimap-alpha", type=float, default=0.85, help="Mini-map alpha")
    parser.add_argument("--pulse-speed", type=float, default=1.2, help="Hotspot pulse speed (Hz)")
    parser.add_argument("--vignette-strength", type=float, default=0.25, help="Cinematic vignette strength")
    parser.add_argument("--contrast", type=float, default=1.04, help="Cinematic contrast")
    parser.add_argument("--saturation", type=float, default=1.06, help="Cinematic saturation")
    parser.add_argument("--floor-mask-min-coverage", type=float, default=0.005, help="Minimum fraction of pixels for floor mask to be used")
    parser.add_argument("--floor-mask-min-keep", type=float, default=0.03, help="Minimum retained fraction after floor masking")
    parser.add_argument("--overlay-min-coverage", type=float, default=0.001, help="Minimum fraction of pixels for shaded overlay; otherwise relax masking")
    parser.add_argument("--skip-occlusion-for-risk", action="store_true", help="Skip depth occlusion for trip/slip/hotspot layers")
    parser.add_argument("--floor-fallback-alpha", type=float, default=0.35, help="Alpha for floor-wide fallback shading")
    parser.add_argument("--bottom-fallback", type=float, default=0.45, help="Bottom-frame fallback ratio for floor mask")
    parser.add_argument("--full-floor-heat", dest="full_floor_heat", action="store_true", help="Overlay full-floor heat shading")
    parser.add_argument("--no-full-floor-heat", dest="full_floor_heat", action="store_false", help="Disable full-floor heat shading")
    parser.add_argument("--full-floor-heat-alpha", type=float, default=0.45, help="Alpha for full-floor heat shading")
    parser.add_argument("--full-floor-heat-top", type=float, default=0.35, help="Top ratio for full-floor heat band")
    parser.add_argument("--path-only", dest="path_only", action="store_true", help="Restrict overlays to walking path corridor")
    parser.add_argument("--no-path-only", dest="path_only", action="store_false", help="Do not restrict overlays to walking path corridor")
    parser.add_argument("--path-width", type=int, default=20, help="Half-width (px) of walking path corridor")
    parser.add_argument("--detect-objects", action="store_true", help="Draw object detection boxes")
    parser.add_argument("--detect-every", type=int, default=5, help="Run object detection every N frames")
    parser.add_argument("--detect-score", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--detect-max", type=int, default=20, help="Maximum detections per frame")
    parser.add_argument(
        "--detect-classes",
        type=str,
        default="dining table",
        help="Comma-separated class filter (COCO names). Empty = all",
    )
    parser.add_argument("--detect-rug", action="store_true", help="Add a rug heuristic box")
    parser.add_argument("--detect-hardwood", action="store_true", help="Add hardwood floor box")
    parser.add_argument("--callouts", dest="callouts", action="store_true", help="Draw callout markers instead of large overlays")
    parser.add_argument("--no-callouts", dest="callouts", action="store_false", help="Disable callout markers")
    parser.add_argument("--callout-max", type=int, default=120, help="Max callout points per layer")
    parser.add_argument("--callout-labels", dest="callout_labels", action="store_true", help="Draw labels for top callouts")
    parser.add_argument("--no-callout-labels", dest="callout_labels", action="store_false", help="Disable callout labels")
    parser.add_argument("--ot-annotate", dest="ot_annotate", action="store_true", help="Add OT-style annotation callouts")
    parser.add_argument("--no-ot-annotate", dest="ot_annotate", action="store_false", help="Disable OT-style annotations")
    parser.add_argument("--ot-max", type=int, default=3, help="Max OT callouts per frame")
    parser.add_argument("--ot-duration", type=float, default=1.5, help="Seconds to keep OT callouts on screen")
    parser.add_argument("--risk-badges", dest="risk_badges", action="store_true", help="Show trip/slip badges")
    parser.add_argument("--no-risk-badges", dest="risk_badges", action="store_false", help="Hide trip/slip badges")
    parser.add_argument("--include-all-trip-slip", dest="include_all_trip_slip", action="store_true", help="Render all nonzero trip/slip cells")
    parser.add_argument("--no-include-all-trip-slip", dest="include_all_trip_slip", action="store_false", help="Allow quantile sampling for trip/slip")
    parser.add_argument("--no-minimap", dest="show_minimap", action="store_false", help="Disable mini-map inset")
    parser.add_argument("--no-split", dest="show_split", action="store_false", help="Disable split-screen intro")
    parser.add_argument("--no-grade", dest="show_grade", action="store_false", help="Disable cinematic grade/vignette")
    parser.set_defaults(
        show_minimap=True,
        show_split=True,
        show_grade=True,
        include_all_trip_slip=True,
        skip_occlusion_for_risk=True,
        full_floor_heat=False,
        path_only=False,
        callouts=True,
        callout_labels=True,
        ot_annotate=True,
        risk_badges=False,
    )
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
    floor_stride = max(1, int(floor_pts.shape[0] / 6000))
    floor_mask_pts = floor_pts[::floor_stride]

    components_path = risk_dir / "risk_components.npz"
    component_colors: Dict[str, Tuple[int, int, int]] = {
        "obstacle": (0, 60, 255),    # tables / edges (red)
        "trip": (0, 160, 255),       # trip (orange)
        "slip": (255, 130, 40),      # slip (cyan)
        "turn": (0, 255, 120),       # turn (green)
        "physics": (255, 0, 255),    # physics (magenta)
    }
    semantic_color_overrides: Dict[str, Tuple[int, int, int]] = {
        "semantic_trip": (180, 0, 255),      # rugs: purple
        "semantic_slip": (255, 200, 0),      # slick floor: teal
        "semantic_obstacle": (0, 0, 220),    # tables: deep red
    }
    component_layers: List[Dict[str, np.ndarray]] = []
    mini_trip = None
    mini_slip = None
    mini_heat = None

    if args.show_components and components_path.exists():
        comp_data = np.load(components_path)
        if "trip" in comp_data:
            mini_trip = comp_data["trip"].astype(np.float32)
        if "slip" in comp_data:
            mini_slip = comp_data["slip"].astype(np.float32)
        per_layer = max(50, args.max_points // max(len(component_colors), 1))
        slip_layers: List[Dict[str, np.ndarray]] = []
        for name, color in component_colors.items():
            if name not in comp_data:
                continue
            arr = comp_data[name].ravel()
            if not np.any(arr > 0):
                continue
            limit = max(np.percentile(arr, 95), arr.max(), 1e-6)
            norm_vals = arr / limit
            if args.include_all_trip_slip and name in {"trip", "slip", "obstacle"}:
                idx = np.where(arr > 0)[0]
            else:
                thresh = np.quantile(norm_vals, args.component_quantile) if np.any(norm_vals > 0) else 1.0
                idx = np.where(norm_vals >= thresh)[0]
                if idx.size > per_layer:
                    idx = np.random.choice(idx, size=per_layer, replace=False)
            layer = {
                "name": name,
                "points": floor_pts[idx],
                "values": np.clip(norm_vals[idx], 0.0, 1.0),
                "color": np.array(color, dtype=np.int32),
                "mode": "component",
            }
            if name == "slip":
                slip_layers.append(layer)
            else:
                component_layers.append(layer)
        component_layers.extend(slip_layers)

    # Add semantic hazard layers directly (helps rugs/tables show up even when risk is diffuse).
    semantic_path = workdir / "semantic_hazards.npz"
    if args.show_components and semantic_path.exists():
        sem = np.load(semantic_path)
        per_layer = max(1200, args.max_points)
        semantic_slip_layers: List[Dict[str, np.ndarray]] = []
        for name, color in component_colors.items():
            if name not in sem:
                continue
            arr = sem[name]
            if name in {"trip", "slip"}:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                arr = cv2.dilate(arr.astype(np.float32), kernel, iterations=1)
            arr = arr.ravel()
            if not np.any(arr > 0):
                continue
            limit = max(np.percentile(arr, 95), arr.max(), 1e-6)
            norm_vals = arr / limit
            if args.include_all_trip_slip and name in {"trip", "slip", "obstacle"}:
                idx = np.where(arr > 0)[0]
            else:
                idx = np.where(arr > 0)[0]
                if idx.size > per_layer:
                    idx = idx[np.argsort(norm_vals[idx])[::-1][:per_layer]]
            layer = {
                "name": f"semantic_{name}",
                "points": floor_pts[idx],
                "values": np.clip(norm_vals[idx], 0.0, 1.0),
                "color": np.array(color, dtype=np.int32),
                "mode": "component",
            }
            if name == "slip":
                semantic_slip_layers.append(layer)
            else:
                component_layers.append(layer)
        component_layers.extend(semantic_slip_layers)

    if args.show_heat:
        heat_flat = heatmap.ravel()
        heat_max = heat_flat.max() if heat_flat.size else 0.0
        if heat_max <= 0:
            raise RuntimeError("Heatmap all zeros; nothing to overlay.")
        norm_heat = heat_flat / heat_max
        mini_heat = heatmap.astype(np.float32)
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

    # Hotspot-only layer from risk summary (if available).
    risk_summary_path = risk_dir / "risk_summary.json"
    if risk_summary_path.exists():
        try:
            summary = yaml.safe_load(risk_summary_path.read_text(encoding="utf-8")) or {}
            hotspots = summary.get("hotspots", [])
            if hotspots:
                pts = []
                vals = []
                for hspot in hotspots:
                    pos = hspot.get("position")
                    score = hspot.get("score", 1.0)
                    if isinstance(pos, list) and len(pos) >= 2:
                        pts.append([float(pos[0]), float(pos[1]), float(floor_z)])
                        vals.append(float(score))
                if pts:
                    pts = np.array(pts, dtype=np.float32)
                    pts = _unscale(pts)
                    vals = np.array(vals, dtype=np.float32)
                    if vals.size and vals.max() > 0:
                        vals = vals / vals.max()
                    component_layers.append(
                        {
                            "name": "hotspot",
                            "points": pts,
                            "values": vals,
                            "color": np.array((0, 0, 255), dtype=np.int32),
                            "mode": "component",
                        }
                    )
        except Exception:
            pass

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

    path_points_world: List[np.ndarray] = []
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
        path_points_world.append(grid_cells_to_world(path))
        add_layer(
            "turn",
            path,
            np.clip(risk, 0.0, 1.0),
            component_colors["turn"],
        )

    if mini_trip is None:
        mini_trip = heatmap.astype(np.float32)
    minimap = _render_minimap(mini_trip, mini_slip, mini_heat if args.show_heat else None, args.minimap_size, component_colors)
    # Precompute a dense floor heat image (trip + slip) for full-floor shading.
    floor_heat = None
    if mini_trip is not None or mini_slip is not None:
        base = np.zeros_like(mini_trip if mini_trip is not None else mini_slip, dtype=np.float32)
        if mini_trip is not None:
            base += mini_trip.astype(np.float32)
        if mini_slip is not None:
            base += mini_slip.astype(np.float32)
        if base.max() <= 0 and semantic_path.exists():
            sem = np.load(semantic_path)
            if "trip" in sem:
                base += sem["trip"].astype(np.float32)
            if "slip" in sem:
                base += sem["slip"].astype(np.float32)
        if base.max() > 0:
            base = base / base.max()
            floor_heat = (plt.get_cmap("inferno")(np.clip(base, 0, 1))[:, :, :3] * 255).astype(np.uint8)

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

    vignette = _build_vignette(width, height, args.vignette_strength)
    split_frames = int(round(max(args.split_seconds, 0.0) * video_fps))
    fade_frames = int(round(max(args.fade_seconds, 0.0) * video_fps))

    last_camera: Optional[Dict[str, np.ndarray]] = None
    last_pose_name: Optional[str] = None
    last_depth: Optional[np.ndarray] = None
    detector = None
    last_dets: List[Dict[str, object]] = []
    ot_callouts: List[Tuple[str, Tuple[int, int]]] = []
    ot_expiry: List[int] = []
    class_filter: Optional[set] = None
    if args.detect_objects:
        detector = _load_detector()
        detect_classes = [c.strip().lower() for c in args.detect_classes.split(",") if c.strip()]
        class_filter = set(detect_classes) if detect_classes else None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frame = frame.copy()
        if args.show_grade:
            frame = _apply_cinematic_grade(frame, vignette, args.contrast, args.saturation)
        pose_idx = min(frame_idx // frame_interval, len(available_frames) - 1)
        pose_name = available_frames[pose_idx]
        camera = images.get(pose_name)
        if camera is None:
            camera = last_camera
        else:
            last_camera = camera
        overlay = frame.copy()
        if camera:
            floor_mask = None
            proj_floor, valid_floor, _ = _project_points(floor_mask_pts, camera, cameras)
            if proj_floor.size:
                proj_floor = proj_floor[np.isfinite(proj_floor).all(axis=1)]
                if args.min_v_ratio > 0:
                    v_min = int(height * args.min_v_ratio)
                    proj_floor = proj_floor[proj_floor[:, 1] >= v_min]
                mask = np.zeros((height, width), dtype=np.uint8)
                for (u, v) in proj_floor.astype(np.int32):
                    if 0 <= u < width and 0 <= v < height:
                        cv2.rectangle(mask, (u - 6, v - 6), (u + 6, v + 6), 255, thickness=-1)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
                mask = cv2.dilate(mask, kernel, iterations=1)
                if np.count_nonzero(mask) / float(height * width) >= args.floor_mask_min_coverage:
                    floor_mask = mask
            if floor_mask is None:
                fallback_start = int(height * args.bottom_fallback)
                if fallback_start < height:
                    floor_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.rectangle(floor_mask, (0, fallback_start), (width - 1, height - 1), 255, thickness=-1)
            path_mask = None
            if args.path_only and path_points_world:
                path_pts = np.vstack(path_points_world)
                proj_path, valid_path, _ = _project_points(path_pts, camera, cameras)
                if proj_path.size:
                    proj_path = proj_path[np.isfinite(proj_path).all(axis=1)]
                    mask = np.zeros((height, width), dtype=np.uint8)
                    for (u, v) in proj_path.astype(np.int32):
                        if 0 <= u < width and 0 <= v < height:
                            cv2.rectangle(
                                mask,
                                (u - args.path_width, v - args.path_width),
                                (u + args.path_width, v + args.path_width),
                                255,
                                thickness=-1,
                            )
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    if np.count_nonzero(mask) / float(height * width) >= 0.0005:
                        path_mask = mask

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

            risk_any_mask = None
            trip_mask_accum = None
            slip_mask_accum = None
            hazard_callouts: List[Tuple[str, Tuple[int, int]]] = []
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
                proj_unoccluded = proj.copy()
                values_unoccluded = values.copy()
                depths_unoccluded = depths.copy()
                # Discard points above a conservative image band (avoid wall overlays).
                if args.min_v_ratio > 0:
                    v_min = int(height * args.min_v_ratio)
                    keep = proj[:, 1] >= v_min
                    if not np.any(keep):
                        continue
                    proj = proj[keep]
                    values = values[keep]
                    depths = depths[keep]

                if depth_m is not None and not (args.skip_occlusion_for_risk and layer.get("mode") != "heat"):
                    orig_len = len(values)
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
                    if np.any(visible):
                        kept = int(np.count_nonzero(visible))
                        if orig_len > 0 and kept / float(orig_len) < args.depth_min_keep:
                            proj = proj_unoccluded
                            values = values_unoccluded
                        else:
                            proj = proj[visible]
                            values = values[visible]
                    else:
                        proj = proj_unoccluded
                        values = values_unoccluded
                proj = np.nan_to_num(proj, nan=-1e6, posinf=-1e6, neginf=-1e6)
                pts = proj.astype(np.int32)
                if pts.size == 0:
                    continue

                name = layer.get("name", "")
                is_semantic = name.startswith("semantic_")
                is_slip = "slip" in name
                is_trip = "trip" in name
                is_obstacle = "obstacle" in name
                if layer.get("mode") == "heat":
                    if not args.show_heat:
                        continue
                    colors = _color_map(values)
                    for (u, v), color, value in zip(pts, colors, values):
                        if 0 <= u < width and 0 <= v < height:
                            radius = max(args.point_radius_min, int(args.point_radius_min + value * (args.point_radius_max - args.point_radius_min)))
                            cv2.circle(overlay, (u, v), radius, tuple(int(c) for c in color), thickness=-1)
                else:
                    # Only shade trip/slip/hotspot layers; skip glare/other components.
                    if not (is_trip or is_slip or is_obstacle or name == "hotspot" or name.startswith("semantic_")):
                        continue
                    base_color = layer.get("color", np.array((0, 0, 255), dtype=np.int32))
                    override = semantic_color_overrides.get(name)
                    if override is not None:
                        base_color = np.array(override, dtype=np.int32)
                    if args.callouts:
                        # Draw small callout markers instead of large shaded regions.
                        order = np.argsort(values)[::-1]
                        if order.size > args.callout_max:
                            order = order[: args.callout_max]
                        label_name = name
                        if name == "semantic_trip":
                            label_name = "rug"
                        elif name == "semantic_obstacle":
                            label_name = "table"
                        elif name == "semantic_slip":
                            label_name = "slip area"
                        elif name == "hotspot":
                            label_name = "hotspot"
                        elif name == "obstacle":
                            label_name = "obstacle"
                        elif name == "trip":
                            label_name = "trip"
                        elif name == "slip":
                            label_name = "slip"
                        for k, idx_pt in enumerate(order):
                            u, v = pts[idx_pt]
                            if 0 <= u < width and 0 <= v < height:
                                size = 5 if values[idx_pt] < 0.5 else 7
                                cv2.rectangle(
                                    overlay,
                                    (u - size, v - size),
                                    (u + size, v + size),
                                    tuple(int(c) for c in base_color),
                                    thickness=-1,
                                )
                                cv2.circle(
                                    overlay,
                                    (u, v),
                                    size + 2,
                                    tuple(int(c) for c in base_color),
                                    thickness=1,
                                )
                                if args.callout_labels and k < 6:
                                    text = label_name
                                    cv2.putText(
                                        overlay,
                                        text,
                                        (u + 8, max(10, v - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45,
                                        tuple(int(c) for c in base_color),
                                        1,
                                        cv2.LINE_AA,
                                    )
                        if args.ot_annotate:
                            msg_label = label_name
                            if label_name == "hotspot":
                                msg_label = "trip"
                            msg = _ot_message_for(msg_label)
                            if msg:
                                idx_best = order[0] if order.size else None
                                if idx_best is not None:
                                    u, v = pts[idx_best]
                                    hazard_callouts.append((msg, (int(u), int(v))))
                        continue
                    # Hard color bands: draw separate masks by value buckets.
                    low_mask = np.zeros((height, width), dtype=np.uint8)
                    mid_mask = np.zeros((height, width), dtype=np.uint8)
                    high_mask = np.zeros((height, width), dtype=np.uint8)
                    for (u, v), value in zip(pts, values):
                        if 0 <= u < width and 0 <= v < height:
                            r = max(10, int(12 + value * 18))
                            x0, y0 = max(0, u - r), max(0, v - r)
                            x1, y1 = min(width - 1, u + r), min(height - 1, v + r)
                            if value >= 0.66:
                                cv2.rectangle(high_mask, (x0, y0), (x1, y1), 255, thickness=-1)
                            elif value >= 0.33:
                                cv2.rectangle(mid_mask, (x0, y0), (x1, y1), 255, thickness=-1)
                            else:
                                cv2.rectangle(low_mask, (x0, y0), (x1, y1), 255, thickness=-1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                    low_mask = cv2.dilate(low_mask, kernel, iterations=1)
                    mid_mask = cv2.dilate(mid_mask, kernel, iterations=1)
                    high_mask = cv2.dilate(high_mask, kernel, iterations=1)
                    raw_mask = cv2.bitwise_or(cv2.bitwise_or(low_mask, mid_mask), high_mask)
                    if path_mask is not None:
                        raw_count = float(np.count_nonzero(raw_mask))
                        masked = cv2.bitwise_and(raw_mask, path_mask)
                        masked_count = float(np.count_nonzero(masked))
                        if raw_count > 0 and masked_count / raw_count >= 0.01:
                            raw_mask = masked
                            low_mask = cv2.bitwise_and(low_mask, masked)
                            mid_mask = cv2.bitwise_and(mid_mask, masked)
                            high_mask = cv2.bitwise_and(high_mask, masked)
                    elif floor_mask is not None and not is_semantic:
                        raw_count = float(np.count_nonzero(raw_mask))
                        low_mask = cv2.bitwise_and(low_mask, floor_mask)
                        mid_mask = cv2.bitwise_and(mid_mask, floor_mask)
                        high_mask = cv2.bitwise_and(high_mask, floor_mask)
                        masked = cv2.bitwise_or(cv2.bitwise_or(low_mask, mid_mask), high_mask)
                        masked_count = float(np.count_nonzero(masked))
                        if raw_count > 0 and masked_count / raw_count >= args.floor_mask_min_keep:
                            raw_mask = masked
                            low_mask = cv2.bitwise_and(low_mask, masked)
                            mid_mask = cv2.bitwise_and(mid_mask, masked)
                            high_mask = cv2.bitwise_and(high_mask, masked)
                    # If masking wiped out too much, fall back to unmasked overlay.
                    if np.count_nonzero(raw_mask) / float(height * width) < args.overlay_min_coverage:
                        raw_mask = cv2.bitwise_or(cv2.bitwise_or(low_mask, mid_mask), high_mask)
                    low_mask = cv2.GaussianBlur(low_mask, (0, 0), 3)
                    mid_mask = cv2.GaussianBlur(mid_mask, (0, 0), 3)
                    high_mask = cv2.GaussianBlur(high_mask, (0, 0), 3)
                    fill_color = np.array(base_color, dtype=np.float32)
                    base_alpha = 0.45 if is_trip else 0.5
                    alpha_low = base_alpha
                    alpha_mid = base_alpha + 0.15
                    alpha_high = base_alpha + 0.25
                    if name == "hotspot":
                        t = frame_idx / max(video_fps, 1.0)
                        pulse = 0.6 + 0.4 * np.sin(2.0 * np.pi * args.pulse_speed * t)
                        alpha_low *= float(np.clip(pulse, 0.5, 1.0))
                        alpha_mid *= float(np.clip(pulse, 0.5, 1.0))
                        alpha_high *= float(np.clip(pulse, 0.5, 1.0))
                    for band_mask, alpha in ((low_mask, alpha_low), (mid_mask, alpha_mid), (high_mask, alpha_high)):
                        if np.any(band_mask > 0):
                            overlay[band_mask > 0] = (
                                overlay[band_mask > 0].astype(np.float32) * (1.0 - alpha)
                                + fill_color * alpha
                            ).astype(np.uint8)
                    if is_trip:
                        trip_mask_accum = raw_mask if trip_mask_accum is None else cv2.bitwise_or(trip_mask_accum, raw_mask)
                    if is_slip:
                        slip_mask_accum = raw_mask if slip_mask_accum is None else cv2.bitwise_or(slip_mask_accum, raw_mask)
                    if risk_any_mask is None:
                        risk_any_mask = raw_mask.copy()
                    else:
                        risk_any_mask = cv2.bitwise_or(risk_any_mask, raw_mask)
            if not args.callouts and not args.path_only and floor_mask is not None and risk_any_mask is not None:
                # If the overlay coverage is still tiny, softly shade the full floor to ensure visibility.
                coverage = float(np.count_nonzero(risk_any_mask)) / float(height * width)
                if coverage < 0.12:
                    fallback_mask = floor_mask
                    overlay[fallback_mask > 0] = (
                        overlay[fallback_mask > 0].astype(np.float32) * (1.0 - args.floor_fallback_alpha)
                        + np.array(component_colors["trip"], dtype=np.float32) * args.floor_fallback_alpha
                    ).astype(np.uint8)
            fade = 1.0
            if fade_frames > 0:
                fade = min(1.0, frame_idx / float(fade_frames))
            frame = cv2.addWeighted(overlay, args.alpha * fade, frame, 1.0 - args.alpha * fade, 0)
            if args.legend and component_layers:
                legend_items = [
                    ("Trip", component_colors["trip"]),
                    ("Slip", component_colors["slip"]),
                    ("Hotspot", (0, 0, 255)),
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
            if args.show_minimap and minimap is not None:
                inset = minimap.copy()
                inset_h, inset_w = inset.shape[:2]
                pad = 10
                x1 = width - inset_w - pad
                y1 = pad
                if x1 >= 0 and y1 + inset_h <= height:
                    roi = frame[y1:y1 + inset_h, x1:x1 + inset_w]
                    blended = cv2.addWeighted(inset, args.minimap_alpha, roi, 1.0 - args.minimap_alpha, 0)
                    frame[y1:y1 + inset_h, x1:x1 + inset_w] = blended
                    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x1 + inset_w + 2, y1 + inset_h + 2), (255, 255, 255), 1)
            if args.risk_badges:
                # Add trip/slip risk badges to make hazard types explicit.
                badge_x, badge_y = 12, height - 60
                badge_w, badge_h = 130, 22
                if trip_mask_accum is not None:
                    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), component_colors["trip"], thickness=-1)
                    cv2.putText(frame, "Trip risk", (badge_x + 6, badge_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    badge_y -= 26
                if slip_mask_accum is not None:
                    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), component_colors["slip"], thickness=-1)
                    cv2.putText(frame, "Slip risk", (badge_x + 6, badge_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if args.full_floor_heat and floor_heat is not None:
                y_start = int(height * args.full_floor_heat_top)
                if y_start < height:
                    heat_band = cv2.resize(floor_heat, (width, height - y_start), interpolation=cv2.INTER_NEAREST)
                    heat_roi = frame[y_start:height, 0:width]
                    if floor_mask is not None:
                        mask = floor_mask[y_start:height, 0:width]
                        blended = cv2.addWeighted(heat_band, args.full_floor_heat_alpha, heat_roi, 1.0 - args.full_floor_heat_alpha, 0)
                        heat_roi[mask > 0] = blended[mask > 0]
                    else:
                        blended = cv2.addWeighted(heat_band, args.full_floor_heat_alpha, heat_roi, 1.0 - args.full_floor_heat_alpha, 0)
                        frame[y_start:height, 0:width] = blended
            det_callouts: List[Tuple[str, Tuple[int, int]]] = []
            if args.detect_objects and detector is not None:
                if frame_idx % max(args.detect_every, 1) == 0:
                    last_dets = _detect_objects(
                        detector,
                        raw_frame,
                        score_thresh=args.detect_score,
                        max_dets=args.detect_max,
                        class_filter=class_filter,
                    )
                    if args.detect_rug:
                        rug_box = _rug_heuristic(raw_frame)
                        if rug_box:
                            last_dets.append({"box": rug_box, "label": "rug", "score": 1.0})
                    if args.detect_hardwood:
                        hardwood_box = _hardwood_heuristic(raw_frame)
                        if hardwood_box:
                            last_dets.append({"box": hardwood_box, "label": "hardwood floor", "score": 1.0})
                for det in last_dets:
                    box = det["box"]
                    label = str(det["label"])
                    # Normalize labels
                    if label == "dining table":
                        label = "table"
                    if label == "couch":
                        label = "sofa"
                    if label == "tv":
                        label = "tv"
                    color = (0, 0, 255)
                    if label in {"rug", "hardwood floor"}:
                        # Skip visual boxes for rug/hardwood per request.
                        continue
                    _draw_box(frame, tuple(int(v) for v in box), color, label)
                    if label == "table":
                        x1, y1, x2, y2 = [int(v) for v in box]
                        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                        for (cx, cy) in corners:
                            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                            cv2.putText(
                                frame,
                                "corner",
                                (cx + 6, cy - 6 if cy > 10 else cy + 14),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA,
                            )
                    msg = _ot_message_for(label)
                    if msg:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        det_callouts.append((msg, (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)))
            if args.ot_annotate:
                period = max(1, int(args.ot_duration * video_fps))
                if frame_idx % period == 0:
                    ot_callouts = hazard_callouts + det_callouts
                    ot_expiry = [frame_idx + period - 1] * len(ot_callouts)
                active = [
                    (callout, anchor)
                    for (callout, anchor), exp in zip(ot_callouts, ot_expiry)
                    if frame_idx <= exp
                ]
                _draw_ot_callouts(frame, active, (0, 0, 0), args.ot_max)
        if args.show_split and frame_idx < split_frames:
            split_x = width // 2
            combined = raw_frame.copy()
            combined[:, split_x:] = frame[:, split_x:]
            cv2.line(combined, (split_x, 0), (split_x, height), (255, 255, 255), 2)
            frame = combined
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Overlay video written to: {args.output}")


if __name__ == "__main__":
    main()
