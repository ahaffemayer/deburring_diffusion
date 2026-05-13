"""Geometric helpers: SDF queries used by the sampler's rejection rules."""
from __future__ import annotations

import math
from typing import Tuple


def box_circle_distance(
    box_center_xy: Tuple[float, float],
    hx: float,
    hy: float,
    yaw: float,
    circle_center_xy: Tuple[float, float],
    circle_r: float,
) -> float:
    """Signed distance between a yawed-axis-aligned box and a circle in 2D.

    Positive = clearance between the box surface and the circle surface.
    Negative = overlap by that amount.

    The box has half-extents ``(hx, hy)`` in its own frame, rotated by
    ``yaw`` radians counterclockwise. The circle is axis-aligned.
    """
    dx = circle_center_xy[0] - box_center_xy[0]
    dy = circle_center_xy[1] - box_center_xy[1]
    c, s = math.cos(yaw), math.sin(yaw)
    # Circle's centre in the box's local frame (inverse rotation).
    dx_local = c * dx + s * dy
    dy_local = -s * dx + c * dy
    closest_x = max(-hx, min(hx, dx_local))
    closest_y = max(-hy, min(hy, dy_local))
    return math.hypot(dx_local - closest_x, dy_local - closest_y) - circle_r


def box_aabb_extent(hx: float, hy: float, yaw: float) -> Tuple[float, float]:
    """Half-extents of the axis-aligned bounding box of a yawed 2D box."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return hx * c + hy * s, hx * s + hy * c
