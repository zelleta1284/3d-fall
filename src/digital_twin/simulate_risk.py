import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .lighting import LightSpec, LightingSpec, WindowSpec, glare_map
from .physics import PhysicsSpec, physics_collision_risk


@dataclass
class PathSpec:
    start: Tuple[float, float]
    goal: Tuple[float, float]
    count: int


@dataclass
class GaitSpec:
    step_length: float
    turn_radius: float
    lateral_std: float
    speed_mps: float


@dataclass
class BiomechanicsSpec:
    foot_clearance_m: float
    cane: bool
    shuffle_bias: float
    reaction_time_s: float


@dataclass
class RiskWeights:
    obstacle: float
    turn: float
    glare: float
    trip: float
    slip: float
    physics: float


@dataclass
class RiskSpec:
    obstacle_distance_m: float
    weights: RiskWeights


@dataclass
class RoomSpec:
    grid_size_m: float
    obstacle_height_m: float
    default_friction: float
    friction_zones: List[Dict]


@dataclass
class SimConfig:
    room: RoomSpec
    paths: List[PathSpec]
    gait: GaitSpec
    biomechanics: BiomechanicsSpec
    lighting: Optional[LightingSpec]
    physics: PhysicsSpec
    risk: RiskSpec


def _safe_float(val: Optional[float], default: float) -> float:
    return float(val) if val is not None else float(default)


def load_config(path: Path) -> SimConfig:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = [
        PathSpec(tuple(p["start"]), tuple(p["goal"]), int(p.get("count", 2000)))
        for p in cfg.get("paths", [])
    ]

    gait_cfg = cfg.get("gait", {})
    gait = GaitSpec(
        step_length=_safe_float(gait_cfg.get("step_length"), 0.5),
        turn_radius=_safe_float(gait_cfg.get("turn_radius"), 0.6),
        lateral_std=_safe_float(gait_cfg.get("lateral_std"), 0.05),
        speed_mps=_safe_float(gait_cfg.get("speed_mps"), 0.7),
    )

    bio_cfg = cfg.get("biomechanics", {})
    biomechanics = BiomechanicsSpec(
        foot_clearance_m=_safe_float(bio_cfg.get("foot_clearance_m"), 0.03),
        cane=bool(bio_cfg.get("cane", False)),
        shuffle_bias=_safe_float(bio_cfg.get("shuffle_bias"), 0.2),
        reaction_time_s=_safe_float(bio_cfg.get("reaction_time_s"), 0.7),
    )

    room_cfg = cfg.get("room", {})
    room = RoomSpec(
        grid_size_m=_safe_float(room_cfg.get("grid_size_m"), 0.05),
        obstacle_height_m=_safe_float(room_cfg.get("obstacle_height_m"), 0.2),
        default_friction=_safe_float(room_cfg.get("default_friction"), 0.6),
        friction_zones=room_cfg.get("friction_zones", []),
    )

    lighting_cfg = cfg.get("lighting")
    lighting = None
    if lighting_cfg:
        windows = [
            WindowSpec(
                center=tuple(w["center"]),
                normal=tuple(w["normal"]),
                width=float(w["width"]),
                height=float(w["height"]),
                transmittance=float(w.get("transmittance", 0.7)),
            )
            for w in lighting_cfg.get("windows", [])
        ]
        lights = [
            LightSpec(
                position=tuple(l["position"]),
                intensity=float(l.get("intensity", 1.0)),
                range_m=float(l.get("range_m", 3.0)),
            )
            for l in lighting_cfg.get("lights", [])
        ]
        lighting = LightingSpec(
            latitude=float(lighting_cfg["latitude"]),
            longitude=float(lighting_cfg["longitude"]),
            timezone_offset_hours=float(lighting_cfg.get("timezone_offset_hours", 0.0)),
            datetime_iso=str(lighting_cfg.get("datetime_iso", "")),
            windows=windows,
            lights=lights,
            glare_weight=float(lighting_cfg.get("glare_weight", 1.0)),
            ambient_weight=float(lighting_cfg.get("ambient_weight", 0.3)),
        )

    physics_cfg = cfg.get("physics", {})
    physics = PhysicsSpec(
        enabled=bool(physics_cfg.get("enabled", False)),
        timestep=_safe_float(physics_cfg.get("timestep"), 0.02),
        steps_per_point=int(physics_cfg.get("steps_per_point", 5)),
        body_radius=_safe_float(physics_cfg.get("body_radius"), 0.2),
        body_height=_safe_float(physics_cfg.get("body_height"), 1.2),
        mass_kg=_safe_float(physics_cfg.get("mass_kg"), 70.0),
        floor_friction=_safe_float(physics_cfg.get("floor_friction"), room.default_friction),
        use_gui=bool(physics_cfg.get("use_gui", False)),
    )

    risk_cfg = cfg.get("risk", {})
    weights_cfg = risk_cfg.get("weights", {})
    weights = RiskWeights(
        obstacle=_safe_float(weights_cfg.get("obstacle", 1.0), 1.0),
        turn=_safe_float(weights_cfg.get("turn", 0.5), 0.5),
        glare=_safe_float(weights_cfg.get("glare", 0.8), 0.8),
        trip=_safe_float(weights_cfg.get("trip", 1.0), 1.0),
        slip=_safe_float(weights_cfg.get("slip", 0.8), 0.8),
        physics=_safe_float(weights_cfg.get("physics", 1.0), 1.0),
    )
    risk = RiskSpec(
        obstacle_distance_m=_safe_float(risk_cfg.get("obstacle_distance_m"), 0.4),
        weights=weights,
    )

    return SimConfig(
        room=room,
        paths=paths,
        gait=gait,
        biomechanics=biomechanics,
        lighting=lighting,
        physics=physics,
        risk=risk,
    )


def load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError("Mesh is empty; check reconstruction output.")
    return np.asarray(mesh.vertices)


def build_grids(vertices: np.ndarray, grid_size: float, obstacle_height: float):
    min_xyz = np.percentile(vertices, 2, axis=0)
    max_xyz = np.percentile(vertices, 98, axis=0)
    floor_z = min_xyz[2]

    min_xy = min_xyz[:2]
    max_xy = max_xyz[:2]
    size = ((max_xy - min_xy) / grid_size).astype(int) + 1

    height_grid = np.full((size[1], size[0]), floor_z, dtype=np.float32)
    idx = ((vertices[:, :2] - min_xy) / grid_size).astype(int)
    idx = np.clip(idx, 0, np.array([size[0] - 1, size[1] - 1]))
    for (x, y), z in zip(idx, vertices[:, 2]):
        if z > height_grid[y, x]:
            height_grid[y, x] = z

    obstacle_grid = (height_grid > floor_z + obstacle_height).astype(np.uint8)
    return obstacle_grid, height_grid, min_xy, floor_z


def distance_transform(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    from collections import deque

    q = deque()
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 1:
                dist[y, x] = 0.0
                q.append((y, x))

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        y, x = q.popleft()
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((ny, nx))
    return dist


def to_grid(point_xy: Tuple[float, float], min_xy: np.ndarray, grid_size: float) -> Tuple[int, int]:
    idx = ((np.array(point_xy) - min_xy) / grid_size).astype(int)
    return int(idx[1]), int(idx[0])


def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    import heapq

    h, w = grid.shape

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for dy, dx in dirs:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0:
                neighbor = (ny, nx)
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, 1e9):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score = tentative + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []


def compute_turn_risk(path: List[Tuple[int, int]]) -> np.ndarray:
    if len(path) < 3:
        return np.zeros(len(path), dtype=np.float32)
    risk = np.zeros(len(path), dtype=np.float32)
    for i in range(1, len(path) - 1):
        y0, x0 = path[i - 1]
        y1, x1 = path[i]
        y2, x2 = path[i + 1]
        v1 = np.array([y1 - y0, x1 - x0], dtype=np.float32)
        v2 = np.array([y2 - y1, x2 - x1], dtype=np.float32)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angle = math.acos(float(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        risk[i] = angle / math.pi
    return risk


def build_friction_grid(room: RoomSpec, grid_shape: Tuple[int, int], min_xy: np.ndarray) -> np.ndarray:
    friction = np.full(grid_shape, room.default_friction, dtype=np.float32)
    for zone in room.friction_zones:
        rect = zone.get("rect")
        mu = float(zone.get("mu", room.default_friction))
        if not rect:
            continue
        x0, y0, x1, y1 = rect
        min_idx = ((np.array([x0, y0]) - min_xy) / room.grid_size_m).astype(int)
        max_idx = ((np.array([x1, y1]) - min_xy) / room.grid_size_m).astype(int)
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, np.array([grid_shape[1] - 1, grid_shape[0] - 1]))
        friction[min_idx[1] : max_idx[1] + 1, min_idx[0] : max_idx[0] + 1] = mu
    return friction


def simulate_risk(mesh_path: Path, config_path: Path, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    cfg = load_config(config_path)
    vertices = load_mesh_vertices(mesh_path)
    obstacle_grid, height_grid, min_xy, floor_z = build_grids(
        vertices, cfg.room.grid_size_m, cfg.room.obstacle_height_m
    )

    dist_grid = distance_transform(obstacle_grid) * cfg.room.grid_size_m
    dist_grid = np.where(np.isfinite(dist_grid), dist_grid, 0)
    friction_grid = build_friction_grid(cfg.room, obstacle_grid.shape, min_xy)

    glare = np.zeros_like(dist_grid, dtype=np.float32)
    if cfg.lighting:
        glare = glare_map(obstacle_grid.shape, min_xy, cfg.room.grid_size_m, floor_z, cfg.lighting)

    heat = np.zeros_like(dist_grid, dtype=np.float32)
    physics_paths: List[List[Tuple[int, int]]] = []

    clearance = cfg.biomechanics.foot_clearance_m * (1.0 - 0.5 * cfg.biomechanics.shuffle_bias)
    stability = 0.9 if cfg.biomechanics.cane else 1.0

    for path_spec in cfg.paths:
        start = to_grid(path_spec.start, min_xy, cfg.room.grid_size_m)
        goal = to_grid(path_spec.goal, min_xy, cfg.room.grid_size_m)
        base_path = a_star(obstacle_grid, start, goal)
        if not base_path:
            continue

        physics_paths.append(base_path)
        base_path = np.array(base_path)
        turn_risk = compute_turn_risk(base_path.tolist())

        for _ in range(path_spec.count):
            jitter = np.random.normal(0, cfg.gait.lateral_std / cfg.room.grid_size_m, size=base_path.shape)
            jitter[:, 0] = 0
            jittered = base_path + jitter
            jittered = np.clip(jittered, [0, 0], np.array(obstacle_grid.shape) - 1)
            jittered = jittered.astype(int)

            for i, (y, x) in enumerate(jittered):
                if obstacle_grid[y, x] == 1:
                    continue

                dist = dist_grid[y, x]
                obstacle_risk = max(
                    0.0,
                    (cfg.risk.obstacle_distance_m - dist) / max(cfg.risk.obstacle_distance_m, 1e-6),
                )

                height = height_grid[y, x] - floor_z
                trip_risk = 1.0 if height > clearance else 0.0
                trip_risk *= 1.0 + cfg.biomechanics.shuffle_bias

                mu = friction_grid[y, x]
                a_lat = (cfg.gait.speed_mps ** 2) / max(cfg.gait.turn_radius, 1e-3)
                slip_ratio = max(0.0, (a_lat - mu * 9.81) / max(mu * 9.81, 1e-6))
                slip_risk = slip_ratio * (1.0 + turn_risk[i])

                heat[y, x] += (
                    cfg.risk.weights.obstacle * obstacle_risk
                    + cfg.risk.weights.turn * turn_risk[i]
                    + cfg.risk.weights.trip * trip_risk
                    + cfg.risk.weights.slip * slip_risk
                ) * stability

    if cfg.physics.enabled:
        physics_risk = physics_collision_risk(
            str(mesh_path),
            obstacle_grid.shape,
            min_xy,
            cfg.room.grid_size_m,
            floor_z,
            physics_paths,
            cfg.physics,
        )
        heat += cfg.risk.weights.physics * physics_risk

    heat += cfg.risk.weights.glare * glare

    out_dir.mkdir(parents=True, exist_ok=True)
    heat_path = out_dir / "risk_heatmap.npy"
    np.save(heat_path, heat)

    # Normalize for visualization
    max_val = np.percentile(heat, 99) if np.any(heat) else 1.0
    viz = np.clip(heat / max_val, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(viz, cmap="hot")
    plt.title("Fall Risk Heatmap")
    plt.axis("off")
    img_path = out_dir / "risk_heatmap.png"
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return img_path
