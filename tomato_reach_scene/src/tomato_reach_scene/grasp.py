"""Canonical top-down grasp pose for the tomato (B).

The grasp pose is the 6-DoF goal pose for the gripper TCP (`panda_hand_tcp`
in the Panda URDF) in the **robot base frame**. For a Franka with the
standard hand, the TCP sits between the two fingers, ~10.3 cm forward of
``panda_hand``'s mounting flange.

For v1 we emit a single canonical grasp per scene:
- Position: B's top face centre + ``approach_offset_m`` along base +z
  (gripper approaches from above and stops with the TCP slightly above
  B's top face, ready to descend and close).
- Orientation: gripper z-axis pointing in base -z (so fingers close
  around B as it descends). Gripper x-axis aligned with base +x (a
  deterministic choice — symmetric for a cylindrical target).

Future extensions (out of v1 scope):
- Sample grasp ``tilt`` (deviation of approach axis from base -z) and
  ``yaw`` (rotation of approach about base +z) for grasp diversity.
- Add side / pinch grasps for non-symmetric targets.
"""
from __future__ import annotations

import numpy as np

from .constants import composite_pose_in_base


def _matrix_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (qx, qy, qz, qw) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def canonical_top_down_grasp(
    b_center_xyz_composite_m: tuple[float, float, float],
    b_top_z_composite_m: float,
    approach_offset_m: float = 0.005,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compute the canonical top-down grasp pose for B (tomato).

    Parameters
    ----------
    b_center_xyz_composite_m
        B's centre in the composite frame, metres.
    b_top_z_composite_m
        z coordinate of B's top face in the composite frame, metres.
    approach_offset_m
        Distance above B's top face where the TCP stops. Default 5 mm.

    Returns
    -------
    (xyz_base_m, quat_xyzw_base)
        Translation in metres and quaternion ``(qx, qy, qz, qw)``, both
        in the **robot base frame**. Suitable for emitting directly to
        a planner.
    """
    # Position in composite frame.
    grasp_xyz_composite = np.array([
        b_center_xyz_composite_m[0],
        b_center_xyz_composite_m[1],
        b_top_z_composite_m + approach_offset_m,
    ], dtype=np.float64)

    # Top-down orientation in composite (and base, since composite_R = I):
    # gripper z = base -z, gripper x = base +x, gripper y = base -y.
    # In matrix form, columns = gripper x, y, z expressed in base.
    R_grasp_in_base = np.array([
        [1.0,  0.0,  0.0],   # gripper x in base = base x
        [0.0, -1.0,  0.0],   # gripper y in base = -base y
        [0.0,  0.0, -1.0],   # gripper z in base = -base z
    ], dtype=np.float64)

    # Composite frame is axis-aligned with base, so coordinates transfer
    # by a simple translation offset.
    B_T_S = composite_pose_in_base()
    grasp_xyz_base = B_T_S[:3, :3] @ grasp_xyz_composite + B_T_S[:3, 3]
    quat = _matrix_to_quat_xyzw(R_grasp_in_base)
    return tuple(float(x) for x in grasp_xyz_base), quat
