import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class WindowSpec:
    center: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    width: float
    height: float
    transmittance: float = 0.7


@dataclass
class LightSpec:
    position: Tuple[float, float, float]
    intensity: float
    range_m: float


@dataclass
class LightingSpec:
    latitude: float
    longitude: float
    timezone_offset_hours: float
    datetime_iso: str
    windows: List[WindowSpec]
    lights: List[LightSpec]
    glare_weight: float = 1.0
    ambient_weight: float = 0.3


def _sun_position(dt: datetime, latitude: float, longitude: float, tz_offset_hours: float):
    # NOAA approximate solar position (radians)
    day = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    time_offset = eqtime + 4.0 * longitude - 60.0 * tz_offset_hours
    tst = hour * 60.0 + time_offset
    ha = math.radians(tst / 4.0 - 180.0)

    lat_rad = math.radians(latitude)
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    cos_zenith = min(1.0, max(-1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation = math.pi / 2.0 - zenith

    sin_az = -math.sin(ha) * math.cos(decl)
    cos_az = (math.sin(decl) - math.sin(lat_rad) * math.cos(zenith)) / (math.cos(lat_rad) * math.sin(zenith) + 1e-6)
    azimuth = math.atan2(sin_az, cos_az)

    return elevation, azimuth


def sun_direction(dt: datetime, latitude: float, longitude: float, tz_offset_hours: float) -> np.ndarray:
    elevation, azimuth = _sun_position(dt, latitude, longitude, tz_offset_hours)
    # Direction from sun to ground (light direction)
    if elevation <= 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Azimuth measured clockwise from north; convert to x (east), y (north)
    x = math.sin(azimuth) * math.cos(elevation)
    y = math.cos(azimuth) * math.cos(elevation)
    z = math.sin(elevation)
    # Light direction points downward
    return np.array([-x, -y, -z], dtype=np.float32)


def _window_corners(window: WindowSpec) -> np.ndarray:
    center = np.array(window.center, dtype=np.float32)
    normal = np.array(window.normal, dtype=np.float32)
    normal /= np.linalg.norm(normal) + 1e-6

    # Build orthonormal basis for window plane
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(up, normal)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent = np.cross(up, normal)
    tangent /= np.linalg.norm(tangent) + 1e-6
    bitangent = np.cross(normal, tangent)

    half_w = window.width * 0.5
    half_h = window.height * 0.5

    return np.array(
        [
            center + tangent * half_w + bitangent * half_h,
            center - tangent * half_w + bitangent * half_h,
            center - tangent * half_w - bitangent * half_h,
            center + tangent * half_w - bitangent * half_h,
        ],
        dtype=np.float32,
    )


def glare_map(
    grid_shape: Tuple[int, int],
    min_xy: np.ndarray,
    grid_size: float,
    floor_z: float,
    spec: LightingSpec,
) -> np.ndarray:
    glare = np.zeros(grid_shape, dtype=np.float32)

    try:
        dt = datetime.fromisoformat(spec.datetime_iso)
    except ValueError:
        return glare

    sun_dir = sun_direction(dt, spec.latitude, spec.longitude, spec.timezone_offset_hours)
    if np.allclose(sun_dir, 0.0):
        return glare

    # Sun direction should point into the room (negative z)
    if sun_dir[2] >= 0:
        return glare

    for window in spec.windows:
        normal = np.array(window.normal, dtype=np.float32)
        normal /= np.linalg.norm(normal) + 1e-6
        # Window normal points inward; sun entering if light direction aligns with normal
        if np.dot(sun_dir, normal) <= 0:
            continue

        corners = _window_corners(window)
        t = (floor_z - corners[:, 2]) / (sun_dir[2] + 1e-6)
        projected = corners + sun_dir * t[:, None]
        poly = projected[:, :2]

        # Build mask for polygon
        min_poly = poly.min(axis=0)
        max_poly = poly.max(axis=0)
        min_idx = ((min_poly - min_xy) / grid_size).astype(int)
        max_idx = ((max_poly - min_xy) / grid_size).astype(int)
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, np.array([grid_shape[1] - 1, grid_shape[0] - 1]))

        if min_idx[0] >= max_idx[0] or min_idx[1] >= max_idx[1]:
            continue

        xs = np.arange(min_idx[0], max_idx[0] + 1)
        ys = np.arange(min_idx[1], max_idx[1] + 1)
        xv, yv = np.meshgrid(xs, ys)
        points = np.column_stack(
            [
                min_xy[0] + xv.ravel() * grid_size,
                min_xy[1] + yv.ravel() * grid_size,
            ]
        )
        mask = MplPath(poly).contains_points(points)
        mask = mask.reshape(xv.shape)
        glare_intensity = window.transmittance * abs(sun_dir[2])
        glare[min_idx[1] : max_idx[1] + 1, min_idx[0] : max_idx[0] + 1][mask] += glare_intensity

    # Add artificial lights as soft brightness on floor
    for light in spec.lights:
        pos = np.array(light.position, dtype=np.float32)
        max_range = max(light.range_m, 1e-3)
        for y in range(grid_shape[0]):
            for x in range(grid_shape[1]):
                world_x = min_xy[0] + x * grid_size
                world_y = min_xy[1] + y * grid_size
                dist = np.linalg.norm(np.array([world_x, world_y, floor_z]) - pos)
                if dist <= max_range:
                    glare[y, x] += spec.ambient_weight * light.intensity / (dist * dist + 1e-3)

    return glare * spec.glare_weight
