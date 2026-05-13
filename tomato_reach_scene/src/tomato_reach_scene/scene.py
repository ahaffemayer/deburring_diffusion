"""Dataclasses describing a generated scene + its provenance.

A ``Scene`` is the unit the sampler emits and the exporters consume.
It carries:
- A dict of primitives (shelf + 5 scene objects) each with full pose
  in robot base frame and full collision geometry.
- The grasp pose for the gripper TCP (panda_hand_tcp) in robot base
  frame.
- The sampling parameters that produced it (for manifests / debugging).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

PrimitiveType = Literal["box", "cylinder"]


@dataclass(frozen=True)
class ObjectPlacement:
    """One collision primitive in the world (i.e., in robot base frame).

    Attributes
    ----------
    name
        Stable identifier, e.g. ``"A_rice_noodle"`` or
        ``"shelf_slab_middle"``.
    type
        Either ``"box"`` or ``"cylinder"``.
    dims_m
        For ``box``: ``(x, y, z)`` full extents.
        For ``cylinder``: ``(radius, height, 0.0)`` — third slot
        unused but kept for tuple uniformity.
    pose_xyz_m
        Centre translation ``(x, y, z)`` in robot base frame, metres.
    pose_quat_xyzw
        Rotation quaternion in robot base frame, ``(qx, qy, qz, qw)``.
    """
    name: str
    type: PrimitiveType
    dims_m: tuple[float, float, float]
    pose_xyz_m: tuple[float, float, float]
    pose_quat_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneParams:
    """The sampled scalars that produced one scene.

    Position parameters are stored both as Cartesian (x, y) **in the
    composite frame** and as derived polar (distance from origin),
    because both are useful for downstream analysis.
    """
    k_mm: float
    hard_side: Literal["left", "right", "neither"]
    e_xy_mm: tuple[float, float]   # E's centre in composite frame, mm
    c_xy_mm: tuple[float, float]   # C's centre in composite frame, mm
    e_yaw_deg: float
    dist_e_mm: float
    dist_c_mm: float


@dataclass(frozen=True)
class Scene:
    """A fully realised scene ready for export.

    Attributes
    ----------
    primitives
        Ordered mapping of name → :class:`ObjectPlacement`. Includes the
        7 shelf parts plus the 5 scene objects (A, B, C, D, E).
    grasp_pose_xyz_m, grasp_pose_quat_xyzw
        The 6-DoF goal pose for ``panda_hand_tcp`` in robot base frame.
        For the canonical top-down grasp this is B's top face centre
        plus a small approach offset.
    params
        Sampling parameters that produced this scene.
    """
    primitives: Mapping[str, ObjectPlacement]
    grasp_pose_xyz_m: tuple[float, float, float]
    grasp_pose_quat_xyzw: tuple[float, float, float, float]
    params: SceneParams

    def grasp_pose_matrix(self) -> np.ndarray:
        """Return the grasp pose as a 4x4 SE(3) matrix."""
        from numpy import asarray
        from numpy.linalg import norm
        q = asarray(self.grasp_pose_quat_xyzw, dtype=np.float64)
        q = q / norm(q)
        qx, qy, qz, qw = q
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = self.grasp_pose_xyz_m
        return T
