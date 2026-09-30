"""CARLA (Unreal: left-handed, X forward, Y right, Z up, metres) <-> renderer (right-handed, Y up).

Mapping S: (x, y, z)_UE -> (y, z, -x). Transforms: M = S * M_UE * S^T.
Shared with tools/carla_export/carla_to_scene.py.
"""
from __future__ import annotations

import math

import numpy as np

S = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def ue_point(p) -> np.ndarray:
    """UE point (x, y, z) -> renderer point."""
    return S @ np.asarray(p, dtype=np.float64)


def ue_matrix(m4) -> np.ndarray:
    """4x4 UE matrix (as carla.Transform.get_matrix()) -> 3x4 renderer matrix."""
    m = np.asarray(m4, dtype=np.float64)
    out = np.zeros((3, 4))
    out[:, :3] = S @ m[:3, :3] @ S.T
    out[:, 3] = S @ m[:3, 3]
    return out


def ue_transform_matrix(location, rotation) -> np.ndarray:
    """(x, y, z) metres + (pitch, yaw, roll) degrees -> 4x4 UE matrix (carla.Transform.get_matrix)."""
    x, y, z = location
    pitch, yaw, roll = (math.radians(a) for a in rotation)
    cy, sy, cr, sr, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
                     [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
                     [sp, -cp * sr, cp * cr, z],
                     [0.0, 0.0, 0.0, 1.0]])


def box_instance_matrix(ue_m4, extent, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Renderer matrix placing a unit box (size [1,1,1]) on a UE oriented box.

    ue_m4: UE transform of the box frame; extent: half sizes along UE local axes; center: box centre
    in that frame. The unit box is scaled to (2*ey, 2*ez, 2*ex) in renderer local axes.
    """
    m = np.asarray(ue_m4, dtype=np.float64)
    c = np.eye(4)
    c[:3, 3] = center
    world = ue_matrix(m @ c)
    ex, ey, ez = extent
    scale = np.diag([2 * ey, 2 * ez, 2 * ex])
    world[:, :3] = world[:, :3] @ scale
    return world


def sun_direction(altitude_deg: float, azimuth_deg: float) -> np.ndarray:
    """CARLA weather sun angles -> unit vector towards the sun (renderer frame).
    Azimuth measured in the UE XY plane from +X towards +Y (check against your CARLA version)."""
    alt, az = math.radians(altitude_deg), math.radians(azimuth_deg)
    return ue_point([math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)])


def loc(v) -> tuple:
    """carla.Location / Vector3D (or any object with x, y, z) -> tuple."""
    return (float(v.x), float(v.y), float(v.z))


def rot(r) -> tuple:
    return (float(r.pitch), float(r.yaw), float(r.roll))
