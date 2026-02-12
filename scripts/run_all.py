#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from digital_twin.pipeline import run_pipeline
from digital_twin.simulate_risk import build_grids, load_mesh_vertices, simulate_risk
from digital_twin.window_detection import detect_window_planes
from digital_twin.reporting import compute_hotspots, save_report_json
from digital_twin.scale_estimation import estimate_scale_from_mesh, load_mesh_vertices as load_scale_vertices, scale_mesh


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _label_output(path: Path, workdir: Path) -> str:
    try:
        rel = path.relative_to(workdir)
        return f"<workdir>/{rel.as_posix()}"
    except ValueError:
        return str(path)


def _scale_mesh(mesh_path: Path, distance_m: float, picked_path: Path | None, out_path: Path) -> Path:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D is required for interactive scaling but is not installed.") from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty.")

    if picked_path is None:
        print("Pick two points in the mesh viewer, then press Q.")
        o3d.visualization.draw_geometries_with_editing([mesh])
        picked_path = Path("picked_points.json")

    if not picked_path.exists():
        raise RuntimeError(f"Picked points file not found: {picked_path}")

    data = json.loads(picked_path.read_text(encoding="utf-8"))
    indices = data.get("picked_points", [])
    if len(indices) < 2:
        raise RuntimeError("Need at least two picked points.")

    vertices = np.asarray(mesh.vertices)
    p0 = vertices[int(indices[0])]
    p1 = vertices[int(indices[1])]
    current_dist = float(np.linalg.norm(p0 - p1))
    if current_dist <= 1e-6:
        raise RuntimeError("Picked points are too close; cannot scale.")

    scale = distance_m / current_dist
    mesh.scale(scale, center=mesh.get_center())
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"Scale factor: {scale:.4f}")
    print(f"Scaled mesh written to: {out_path}")
    return out_path


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _insert_windows(cfg: Dict[str, Any], windows: list[dict]) -> Dict[str, Any]:
    lighting = cfg.get("lighting")
    if not isinstance(lighting, dict):
        lighting = {
            "latitude": 0.0,
            "longitude": 0.0,
            "timezone_offset_hours": 0,
            "datetime_iso": "",
            "glare_weight": 1.0,
            "ambient_weight": 0.3,
        }
    lighting["windows"] = windows
    cfg["lighting"] = lighting
    return cfg


def _make_report(
    heatmap_path: Path,
    config_path: Path,
    mesh_path: Path,
    out_dir: Path,
) -> None:
    heat = np.load(heatmap_path)
    cfg = _load_config(config_path)

    vertices = load_mesh_vertices(mesh_path)
    obstacle_grid, _, min_xy, _ = build_grids(
        vertices,
        float(cfg.get("room", {}).get("grid_size_m", 0.05)),
        float(cfg.get("room", {}).get("obstacle_height_m", 0.2)),
    )

    grid_size = float(cfg.get("room", {}).get("grid_size_m", 0.05))
    hotspots = compute_hotspots(heat, min_xy, grid_size, k=5)

    metadata = {
        "grid_size_m": grid_size,
        "min_xy": [float(min_xy[0]), float(min_xy[1])],
        "paths": cfg.get("paths", []),
        "biomechanics": cfg.get("biomechanics", {}),
        "lighting": cfg.get("lighting", {}),
        "physics": cfg.get("physics", {}),
        "patient": cfg.get("patient", {}),
        "scale_result": cfg.get("scale_result", {}),
    }

    interp_path = heatmap_path.parent / "room_interpretation.json"
    if interp_path.exists():
        try:
            metadata["room_interpretation"] = yaml.safe_load(interp_path.read_text(encoding="utf-8"))
        except Exception:
            metadata["room_interpretation"] = {}

    _ensure_dir(out_dir)
    json_path = out_dir / "report.json"
    save_report_json(json_path, metadata, hotspots)

    try:
        import matplotlib.pyplot as plt

        max_val = np.percentile(heat, 99) if np.any(heat) else 1.0
        viz = np.clip(heat / max_val, 0, 1)

        plt.figure(figsize=(8.5, 11))
        plt.subplot(2, 1, 1)
        plt.imshow(viz, cmap="hot")
        plt.title("SureStep.ai Fall Risk Heatmap")
        plt.axis("off")

        plt.subplot(2, 1, 2)
        plt.axis("off")
        lines = [
            "Top Hotspots:",
            *[
                f"- ({h.position[0]:.2f}, {h.position[1]:.2f}) score={h.score:.2f}"
                for h in hotspots
            ],
        ]
        plt.text(0.01, 0.98, "\n".join(lines), va="top", fontsize=10)

        pdf_path = out_dir / "report.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"PDF report written to: {pdf_path}")
    except Exception as exc:
        print(f"PDF generation skipped: {exc}")

    print(f"JSON report written to: {json_path}")


def _risk_context_text(risk_summary: Dict[str, Any]) -> str:
    dominant = risk_summary.get("dominant_factors", [])
    coverage = risk_summary.get("coverage", {})
    hotspots = risk_summary.get("hotspots", [])

    if dominant:
        top = ", ".join(
            f"{d['factor']} ({int(d.get('share', 0) * 100)}% of total)"
            for d in dominant[:3]
        )
    else:
        top = "no clear dominant hazards"

    hotspot_text = ""
    if hotspots:
        coords = hotspots[0]["position"]
        hotspot_text = f" Top hotspot near ({coords[0]:.2f}, {coords[1]:.2f})."

    above_p95 = coverage.get("above_p95", 0.0)
    above_p99 = coverage.get("above_p99", 0.0)
    coverage_text = (
        f" {above_p95*100:.1f}% of the floor exceeds the 95th percentile risk "
        f"and {above_p99*100:.1f}% exceeds the 99th percentile."
        if coverage
        else ""
    )

    return f"Dominant hazards: {top}.{hotspot_text}{coverage_text}"


def _risk_summary(heat: np.ndarray, min_xy: np.ndarray, grid_size: float) -> Dict[str, Any]:
    if heat.size == 0:
        return {"overall_risk_score": 0.0, "hotspots": [], "context_text": "No risk data captured."}

    max_val = np.percentile(heat, 99) if np.any(heat) else 1.0
    overall = float(np.clip(np.mean(heat) / max(max_val, 1e-6), 0.0, 1.0))
    p95 = float(np.percentile(heat, 95))
    p99 = float(np.percentile(heat, 99))
    hotspots = compute_hotspots(heat, min_xy, grid_size, k=5)
    summary = {
        "overall_risk_score": overall,
        "p95": p95,
        "p99": p99,
        "hotspots": [
            {"position": [h.position[0], h.position[1]], "score": h.score} for h in hotspots
        ],
    }
    summary["context_text"] = _risk_context_text(summary)
    return summary


def _suggest_dme(room_name: str | None, patient: Dict[str, Any], interpretation: Dict[str, Any]) -> List[Dict[str, str]]:
    room = (room_name or "").lower()
    suggestions: List[Dict[str, str]] = []

    def add(item: str, reason: str) -> None:
        suggestions.append({"item": item, "reason": reason})

    if room == "bathroom":
        add("Grab bars (toilet + shower)", "Bathroom is a high-risk slip area.")
        add("Non-slip shower mat", "Reduces slip risk on wet surfaces.")
        add("Raised toilet seat", "Improves transfer stability.")
        add("Shower chair", "Supports balance during bathing.")
    elif room == "bedroom":
        add("Bed rail", "Supports safer transfers in/out of bed.")
        add("Night light or motion light", "Improves visibility for night trips.")
        add("Remove/secure loose rugs", "Reduces trip risk near bed path.")
    elif room == "kitchen":
        add("Anti-slip floor mat with beveled edges", "Reduces slip and trip risk.")
        add("Reacher/grabber tool", "Avoids overreaching and instability.")
        add("Declutter walk paths", "Reduces obstacle/trip risk.")
    else:
        add("Secure rugs or remove them", "Common fall source in living areas.")
        add("Increase ambient lighting", "Improves visibility and contrast.")
        add("Reposition low furniture", "Reduces collision/turning risk.")

    if patient.get("assistive_aid") is False and patient.get("fall_last_6_months") is True:
        add("Assistive aid assessment (cane/walker)", "Recent fall history without aid.")

    if patient.get("can_get_out_of_bed") is False:
        add("Bed transfer assist device", "Improves stability during transfers.")

    return suggestions


def _risk_narrative(risk_summary: Dict[str, Any]) -> str:
    overall = risk_summary.get("overall_risk_score", 0.0)
    p95 = risk_summary.get("p95", 0.0)
    p99 = risk_summary.get("p99", 0.0)
    coverage = risk_summary.get("coverage", {})
    above_p95 = coverage.get("above_p95", 0.0)
    above_p99 = coverage.get("above_p99", 0.0)
    dominant = risk_summary.get("dominant_factors", [])

    if overall < 0.25:
        level = "low"
    elif overall < 0.55:
        level = "moderate"
    else:
        level = "high"

    dominant_text = ", ".join([f"{d['factor']} ({int(d['share']*100)}%)" for d in dominant]) if dominant else "unknown factors"

    return (
        f"Overall risk in this room is {level} (score {overall:.2f}). "
        f"Risk is concentrated in small areas: top 5% of floor cells have risk >= {p95:.2f}, "
        f"top 1% >= {p99:.2f}. Approximately {above_p95*100:.1f}% of cells exceed p95 and "
        f"{above_p99*100:.1f}% exceed p99. "
        f"Dominant contributors: {dominant_text}."
    )


def _risk_explanation_text(risk_summary: Dict[str, Any]) -> str:
    return (
        "How to read this: overall_risk_score compares the average floor risk to the worst hotspots "
        "(0 = very low overall, 1 = average as risky as the worst spot). "
        "p95 and p99 are thresholds for the highest-risk 5% and 1% of floor cells. "
        "coverage.above_p95/above_p99 tell what share of the room falls into those high-risk bands. "
        "components break total risk into causes (obstacle, trip, slip, turn, glare, physics). "
        "dominant_factors lists the top contributors, and hotspots are the specific coordinates of highest risk. "
        "Semantic hazards (if enabled) inject object-detection cues into the obstacle/trip/turn components."
    )


def _process_room(
    args: argparse.Namespace,
    video_path: Path,
    config_path: Path,
    workdir: Path,
    intake_payload: Dict[str, Any] | None,
    room_name: str | None,
) -> None:
    _ensure_dir(workdir)

    mesh_path = run_pipeline(
        video_path=video_path,
        work_dir=workdir,
        fps=args.fps,
        median_depth_m=args.median_depth_m,
        run_colmap_flag=True,
    )

    final_mesh = mesh_path
    scale_result = None
    if args.scale_distance:
        scaled = workdir / "mesh_scaled.ply"
        picked = Path(args.picked) if args.picked else None
        final_mesh = _scale_mesh(mesh_path, args.scale_distance, picked, scaled)
        scale_result = {
            "method": "known_distance",
            "distance_m": args.scale_distance,
        }
    elif args.auto_scale:
        try:
            vertices = load_scale_vertices(mesh_path)
            scale_info = estimate_scale_from_mesh(vertices)
            scaled = workdir / "mesh_scaled_auto.ply"
            scale_mesh(mesh_path, scaled, scale_info["scale"])
            final_mesh = scaled
            scale_result = scale_info
        except Exception as exc:
            print(f"Auto-scale skipped: {exc}")

    config_out = workdir / "config_used.yaml"
    cfg = _load_config(config_path)
    if intake_payload and isinstance(intake_payload, dict):
        cfg.setdefault("patient", {})
        cfg["patient"].update(intake_payload)
    if room_name:
        cfg.setdefault("patient", {})
        cfg["patient"]["room_name"] = room_name
    if args.intake:
        intake_path = Path(args.intake)
        intake = _load_config(intake_path)
        if isinstance(intake, dict):
            cfg.setdefault("patient", {})
            cfg["patient"].update(intake)

    if args.auto_windows:
        candidates = detect_window_planes(final_mesh)
        windows = [
            {
                "center": [c.center[0], c.center[1], c.center[2]],
                "normal": [c.normal[0], c.normal[1], c.normal[2]],
                "width": c.width,
                "height": c.height,
                "transmittance": 0.7,
            }
            for c in candidates
        ]
        cfg = _insert_windows(cfg, windows)
        print(f"Detected {len(windows)} window candidates.")

    if scale_result:
        cfg["scale_result"] = scale_result

    _save_config(config_out, cfg)

    semantic_path = None
    semantic_summary = None
    if not args.no_semantic:
        try:
            from digital_twin.semantic_hazards import SemanticConfig, compute_semantic_hazards

            semantic_path = workdir / "semantic_hazards.npz"
            semantic_summary = workdir / "semantic_hazards.json"
            sem_cfg = SemanticConfig(
                score_threshold=args.semantic_score_threshold,
                mask_threshold=args.semantic_mask_threshold,
                frame_stride=args.semantic_frame_stride,
                pixel_stride=args.semantic_pixel_stride,
                low_profile_height_m=args.semantic_low_height,
                rug_min_height_m=args.semantic_rug_min_height,
                rug_max_height_m=args.semantic_rug_max_height,
                rug_gradient_max_m=args.semantic_rug_gradient_max,
                rug_weight=args.semantic_rug_weight,
                small_object_area_ratio=args.semantic_small_object_area,
                small_object_trip_boost=args.semantic_small_object_trip_boost,
            )
            compute_semantic_hazards(
                workdir=workdir,
                mesh_path=final_mesh,
                grid_size=float(cfg.get("room", {}).get("grid_size_m", 0.05)),
                obstacle_height_m=float(cfg.get("room", {}).get("obstacle_height_m", 0.2)),
                out_path=semantic_path,
                summary_path=semantic_summary,
                config=sem_cfg,
            )
            cfg.setdefault("risk", {})
            cfg["risk"]["semantic_hazards_path"] = str(semantic_path)
            _save_config(config_out, cfg)
        except Exception as exc:
            print(f"Semantic hazards skipped: {exc}")

    risk_dir = workdir / "risk"
    simulate_risk(final_mesh, config_out, risk_dir)

    heatmap_path = risk_dir / "risk_heatmap.npy"
    report_dir = workdir / "report"
    _make_report(heatmap_path, config_out, final_mesh, report_dir)

    if not args.no_mesh_video:
        from subprocess import run

        render_script = ROOT / "scripts" / "render_mesh_video.py"
        mesh_video = workdir / "mesh_preview.mp4"
        run(
            [
                sys.executable,
                str(render_script),
                "--mesh",
                str(final_mesh),
                "--out",
                str(mesh_video),
                "--wireframe",
            ],
            check=True,
        )

    # Room-level JSON output
    heat = np.load(heatmap_path)
    cfg = _load_config(config_out)
    grid_size = float(cfg.get("room", {}).get("grid_size_m", 0.05))
    vertices = load_mesh_vertices(final_mesh)
    _, _, min_xy, _ = build_grids(
        vertices,
        grid_size,
        float(cfg.get("room", {}).get("obstacle_height_m", 0.2)),
    )
    interp_path = risk_dir / "room_interpretation.json"
    patient_path = risk_dir / "patient_inference.json"
    interpretation = _load_config(interp_path) if interp_path.exists() else {}
    patient_inference = _load_config(patient_path) if patient_path.exists() else {}

    patient_input = cfg.get("patient", {})
    risk_summary_path = risk_dir / "risk_summary.json"
    if risk_summary_path.exists():
        risk_summary = _load_config(risk_summary_path)
    else:
        risk_summary = _risk_summary(heat, min_xy, grid_size)

    risk_summary.setdefault("context_text", _risk_context_text(risk_summary))

    risk_summary["narrative"] = _risk_narrative(risk_summary)
    risk_summary["explanation_text"] = _risk_explanation_text(risk_summary)

    def _label(p: Path) -> str:
        return _label_output(p, workdir)

    room_output = {
        "patient_input": patient_input,
        "patient_inferences": patient_inference,
        "room_name": patient_input.get("room_name") if isinstance(patient_input, dict) else room_name,
        "room_interpretation": interpretation.get("room", interpretation),
        "risk_summary": risk_summary,
        "explanation_text": risk_summary.get("explanation_text", ""),
        "mitigations": _suggest_dme(room_name, patient_input if isinstance(patient_input, dict) else {}, interpretation),
        "outputs": {
            "mesh_ply": _label(final_mesh),
            "mesh_preview_mp4": _label(workdir / "mesh_preview.mp4"),
            "heatmap_png": _label(risk_dir / "risk_heatmap.png"),
            "heatmap_npy": _label(heatmap_path),
            "room_interpretation_json": _label(interp_path),
            "patient_inference_json": _label(patient_path),
            "risk_summary_json": _label(risk_summary_path),
            "report_json": _label(report_dir / "report.json"),
        },
    }
    if semantic_path and semantic_path.exists():
        room_output["outputs"]["semantic_hazards_npz"] = _label(semantic_path)
    if semantic_summary and semantic_summary.exists():
        room_output["outputs"]["semantic_hazards_json"] = _label(semantic_summary)

    room_output_path = workdir / "room_output.json"
    with room_output_path.open("w", encoding="utf-8") as f:
        json.dump(room_output, f, indent=2)

    print("Done. Outputs:")
    print(f"- Room: {room_name or 'unknown'}")
    print(f"- Mesh: {final_mesh}")
    print(f"- Config used: {config_out}")
    print(f"- Heatmap: {risk_dir / 'risk_heatmap.png'}")
    print(f"- Report: {report_dir}")
    print(f"- Room JSON: {room_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SureStep.ai end-to-end pipeline")
    parser.add_argument("--video", help="Path to input .mp4")
    parser.add_argument("--workdir", required=True, help="Working directory for outputs")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--keragon", help="Path to Keragon POST payload (JSON/YAML)")
    parser.add_argument("--room", help="Room name to select from Keragon payload")
    parser.add_argument("--intake", help="Path to intake JSON/YAML")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    parser.add_argument("--median-depth-m", type=float, default=2.5)
    parser.add_argument("--scale-distance", type=float, default=None, help="Known distance in meters")
    parser.add_argument("--picked", help="picked_points.json path for scaling")
    parser.add_argument("--auto-windows", action="store_true", help="Auto-detect windows and update config copy")
    parser.add_argument("--auto-scale", dest="auto_scale", action="store_true", default=True, help="Auto-scale mesh using priors (default)")
    parser.add_argument("--no-auto-scale", dest="auto_scale", action="store_false", help="Disable auto-scale priors")
    parser.add_argument("--no-mesh-video", action="store_true", help="Skip mesh preview video")
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic hazard detection")
    parser.add_argument("--semantic-score-threshold", type=float, default=0.45)
    parser.add_argument("--semantic-mask-threshold", type=float, default=0.4)
    parser.add_argument("--semantic-frame-stride", type=int, default=2)
    parser.add_argument("--semantic-pixel-stride", type=int, default=4)
    parser.add_argument("--semantic-low-height", type=float, default=0.12)
    parser.add_argument("--semantic-rug-min-height", type=float, default=0.01)
    parser.add_argument("--semantic-rug-max-height", type=float, default=0.08)
    parser.add_argument("--semantic-rug-gradient-max", type=float, default=0.04)
    parser.add_argument("--semantic-rug-weight", type=float, default=0.75)
    parser.add_argument("--semantic-small-object-area", type=float, default=0.015)
    parser.add_argument("--semantic-small-object-trip-boost", type=float, default=1.4)
    args = parser.parse_args()

    if args.keragon:
        payload = _load_config(Path(args.keragon))
        if not isinstance(payload, dict):
            raise RuntimeError("Keragon payload must be a JSON/YAML object")
        rooms = payload.get("rooms", [])
        intake_payload = payload.get("intake") or payload.get("patient")

        selected_rooms = []
        if args.room:
            for room in rooms:
                if room.get("name") == args.room:
                    selected_rooms.append(room)
                    break
        else:
            selected_rooms = rooms

        if not selected_rooms:
            raise RuntimeError("No rooms found in Keragon payload")

        base_workdir = Path(args.workdir)
        for idx, room in enumerate(selected_rooms, start=1):
            room_name = room.get("name") or f"room_{idx}"
            video = room.get("video_path") or room.get("video") or room.get("video_url")
            config = room.get("config_path") or args.config
            if not video:
                raise RuntimeError(f"Missing video for room: {room_name}")
            if not config:
                raise RuntimeError(f"Missing config_path for room: {room_name}")
            _process_room(
                args=args,
                video_path=Path(video),
                config_path=Path(config),
                workdir=base_workdir / room_name,
                intake_payload=intake_payload if isinstance(intake_payload, dict) else None,
                room_name=room_name,
            )
        return

    if not args.video:
        raise RuntimeError("--video is required unless --keragon provides room videos")
    if not args.config:
        raise RuntimeError("--config is required unless --keragon provides config_path")

    _process_room(
        args=args,
        video_path=Path(args.video),
        config_path=Path(args.config),
        workdir=Path(args.workdir),
        intake_payload=None,
        room_name=args.room,
    )


if __name__ == "__main__":
    main()
