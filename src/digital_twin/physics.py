from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PhysicsSpec:
    enabled: bool
    timestep: float
    steps_per_point: int
    body_radius: float
    body_height: float
    mass_kg: float
    floor_friction: float
    use_gui: bool


def _is_floor_contact(contact, floor_z: float) -> bool:
    # contact position on body
    pos = contact[5]
    normal = contact[7]
    if pos[2] <= floor_z + 0.05 and normal[2] > 0.8:
        return True
    return False


def physics_collision_risk(
    mesh_path: str,
    grid_shape: Tuple[int, int],
    min_xy: np.ndarray,
    grid_size: float,
    floor_z: float,
    paths: List[List[Tuple[int, int]]],
    spec: PhysicsSpec,
) -> np.ndarray:
    import pybullet as p

    if not spec.enabled:
        return np.zeros(grid_shape, dtype=np.float32)

    connection = p.GUI if spec.use_gui else p.DIRECT
    cid = p.connect(connection)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(spec.timestep)

    mesh_collision = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    mesh_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=mesh_collision)
    p.changeDynamics(mesh_id, -1, lateralFriction=spec.floor_friction)

    capsule_collision = p.createCollisionShape(p.GEOM_CAPSULE, radius=spec.body_radius, height=spec.body_height)
    body_id = p.createMultiBody(baseMass=spec.mass_kg, baseCollisionShapeIndex=capsule_collision)
    p.changeDynamics(body_id, -1, lateralFriction=spec.floor_friction)

    risk = np.zeros(grid_shape, dtype=np.float32)

    for path in paths:
        for y, x in path:
            world_x = min_xy[0] + x * grid_size
            world_y = min_xy[1] + y * grid_size
            world_z = floor_z + spec.body_height * 0.5 + spec.body_radius

            p.resetBasePositionAndOrientation(body_id, [world_x, world_y, world_z], [0, 0, 0, 1])
            for _ in range(spec.steps_per_point):
                p.stepSimulation()

            contacts = p.getContactPoints(bodyA=body_id, bodyB=mesh_id)
            collision = False
            for c in contacts:
                if not _is_floor_contact(c, floor_z):
                    collision = True
                    break

            if collision:
                risk[y, x] += 1.0

    p.disconnect(cid)
    return risk
