"""Frozen geometric constants for the tomato-reach scene.

All values come from physical measurements of the lab setup and from
the calibrated `B_T_ekenabben` captured on the calibration rig.
Updating these requires re-calibration; see `tools/freeze_constants.py`
(not yet shipped) for the canonical refresh procedure.

Coordinate convention
---------------------
The **composite frame** is anchored at the geometric center of the
EKENÄBBEN shelf's 2nd (middle) slab top surface, with axes **aligned
with the robot base frame**:

- composite +x = base +x = robot forward (the reach direction)
- composite +y = base +y = robot left
- composite +z = base +z = up

So `composite_T_base = identity` (rotation), translation only.

The robot base frame (`panda_link0`) is the **world frame** in all
exported STORM yamls — the robot itself sits at world origin.
"""
from __future__ import annotations

import numpy as np

# ----- object dimensions (in metres) -----

# A — rice noodle (cylinder, standing upright)
A_DIAM_M: float = 0.11297
A_HEIGHT_M: float = 0.120

# B — tomato (cylinder, sits on top of A)
B_DIAM_M: float = 0.057
B_HEIGHT_M: float = 0.041

# C — aqua bottle (cylinder, standing upright)
C_DIAM_M: float = 0.060
C_HEIGHT_M: float = 0.166

# D — soup can (cylinder, standing upright)
D_DIAM_M: float = 0.06637
D_HEIGHT_M: float = 0.10045

# E — cocopops box (rectangular, tall orientation: smallest face on slab)
# Footprint 82.44 × 38.12 mm; height (vertical) 106.17 mm.
# half_extents (x, y, z) in primitive frame:
E_HALF_EXTENTS_M: tuple[float, float, float] = (0.04122, 0.01906, 0.053085)

# Derived radii (used in collision checks)
R_A: float = A_DIAM_M / 2
R_B: float = B_DIAM_M / 2
R_C: float = C_DIAM_M / 2
R_D: float = D_DIAM_M / 2

# ----- shelf geometry (from assets/objects/ekenabben.yaml) -----
#
# In the EKENÄBBEN class frame the slab is 700 mm × 355 mm × 15 mm with
# half-extents [0.3499 (width), 0.1774 (depth), 0.0075]. But the
# composite frame is aligned with the **robot base frame**, and the
# shelf is registered such that its width direction maps to base ±y
# and its depth direction maps to base ±x. So in the composite frame:
#
#   composite +x  =  base +x  =  slab DEPTH direction (short, ~355 mm)
#   composite +y  =  base +y  =  slab WIDTH direction (long, ~700 mm)
#
# Hence:
SLAB_X_HALF_M: float = 0.1773899644613266   # composite x = slab depth half (~177 mm)
SLAB_Y_HALF_M: float = 0.3498820960521698   # composite y = slab width half (~350 mm)
SLAB_HALF_THICKNESS_M: float = 0.0075

# ----- composite frame in robot base frame (frozen from calibration) -----

# Translation: geometric center of the 2nd slab top surface, in
# `panda_link0`. Captured from B_T_ekenabben (state/frames/ekenabben_A.yaml)
# composed with shelf_slab_middle's pose in ekenabben_A's class file.
# Re-derived on every package release.
COMPOSITE_ORIGIN_IN_BASE_M: np.ndarray = np.array([
    0.6484874437186451,
    -0.01843463808028556,
    0.5665821997358784,
], dtype=np.float64)

# Rotation: identity (composite axes aligned with base axes).
COMPOSITE_R_IN_BASE: np.ndarray = np.eye(3, dtype=np.float64)


def composite_pose_in_base() -> np.ndarray:
    """Return 4x4 SE(3): composite frame expressed in robot base frame."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = COMPOSITE_R_IN_BASE
    T[:3, 3] = COMPOSITE_ORIGIN_IN_BASE_M
    return T


# ----- shelf primitives in robot base frame (frozen) -----
#
# All 7 EKENÄBBEN parts (3 slabs + 4 columns) in the SAME shelf instance
# we operate on. Poses derived from B_T_ekenabben composed with each
# primitive's pose in the ekenabben class file.

SHELF_PRIMITIVES_BASE: list[dict] = [
    # name, type, dims_m (full extents for boxes; (r, height) for cylinders),
    # pose_xyzw_base = (x, y, z, qx, qy, qz, qw)
    {
        "name": "shelf_slab_lower",
        "type": "box",
        "dims_m": (0.6997641921043396, 0.3547799289226532, 0.015),
        "pose_xyz_m": (0.6544271911635838, -0.019463267032892195, 0.2186520338496774),
        "pose_quat_xyzw": (-0.007091173733542534, -0.0049743256358721075, 0.7086356966190333, 0.7055213822534411),
    },
    {
        "name": "shelf_slab_middle",
        "type": "box",
        "dims_m": (0.6997641921043396, 0.3547799289226532, 0.015),
        "pose_xyz_m": (0.6486154619976988, -0.018456807930017083, 0.5590833251657852),
        "pose_quat_xyzw": (-0.007091173733542534, -0.0049743256358721075, 0.7086356966190333, 0.7055213822534411),
    },
    {
        "name": "shelf_slab_upper",
        "type": "box",
        "dims_m": (0.6997641921043396, 0.3547799289226532, 0.015),
        "pose_xyz_m": (0.6430160770164226, -0.01748712200280833, 0.8870762177013399),
        "pose_quat_xyzw": (-0.007091173733542534, -0.0049743256358721075, 0.7086356966190333, 0.7055213822534411),
    },
    {
        "name": "shelf_column_front_left",
        "type": "box",
        "dims_m": (0.035, 0.035, 0.8536290178308263),
        "pose_xyz_m": (0.8109952939053107, -0.3502984336048284, 0.5004944606735612),
        "pose_quat_xyzw": (-0.007091173733542534, -0.0049743256358721075, 0.7086356966190333, 0.7055213822534411),
    },
    {
        "name": "shelf_column_front_right",
        "type": "box",
        "dims_m": (0.035, 0.035, 0.8536290178308263),
        "pose_xyz_m": (0.8080846678970074, 0.31443278864254115, 0.4984795443684008),
        "pose_quat_xyzw": (-0.007091173733542534, -0.0049743256358721075, 0.7086356966190333, 0.7055213822534411),
    },
    {
        "name": "shelf_column_back_left",
        "type": "box",
        "dims_m": (0.035, 0.035, 0.8536290178308263),
        "pose_xyz_m": (0.49127418167550974, -0.3517149127393157, 0.4950404764719381),
        "pose_quat_xyzw": (-0.007091173733542534, -0.0049743256358721075, 0.7086356966190333, 0.7055213822534411),
    },
    {
        "name": "shelf_column_back_right",
        "type": "box",
        "dims_m": (0.035, 0.035, 0.8536290178308263),
        "pose_xyz_m": (0.48836355566720646, 0.31301630950805376, 0.4930255601667778),
        "pose_quat_xyzw": (-0.007091173733542534, -0.0049743256358721075, 0.7086356966190333, 0.7055213822534411),
    },
]

# ----- bounds for k (gap between A and D) -----
#
# Derived geometrically from A's diameter + slab depth (see README §3.1).
K_MAX_MM: float = 128.84  # = 2 * (SLAB_X_HALF_M*1000 - 2*R_A*1000) = 2*(177.39 - 112.97)
K_MIN_MM: float = 10.0    # operator floor — visually distinguishable gap

# ----- r_swing (clearance radius) derived from Panda URDF -----
#
# r_swing = R_A + hand_half_thickness + safety_margin
#         = 56.485 + 50 + ~18.5 ≈ 125 mm
# (Safety margin absorbs the small variation as A's radius shifts —
# this defaults to 125 mm for stability across A-diameter swaps so
# scene-difficulty knobs stay calibrated. Override via SceneConfig
# if you want it strictly r_A + 65 mm.)
# See docs/experiments/tomato_reach_scene.md §3.2 for the derivation.
R_SWING_DEFAULT_MM: float = 125.0
