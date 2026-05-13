"""Scene exporters.

Two on-disk world formats are supported; pick the one your downstream
planner expects:

- :func:`to_storm_yaml` / :func:`write_batch` — **STORM / cuRobo V1**
  (≤ 0.7.x). ``world_model.coll_objs.{cube,cylinder,sphere}`` schema;
  pose quaternion in xyzw order.
- :func:`to_curobo_v2_yaml` / :func:`write_batch_v2` — **cuRobo V2**
  (≥ 0.8.0). Bare ``cuboid:`` / ``cylinder:`` at root; pose quaternion
  in wxyz order.

Both writers consume the same :class:`tomato_reach_scene.Scene` —
choose at the export step which format to emit.
"""
from __future__ import annotations

from .curobo_v2 import to_curobo_v2_yaml, write_batch_v2
from .storm_yaml import to_storm_yaml, write_batch

__all__ = [
    "to_curobo_v2_yaml",
    "to_storm_yaml",
    "write_batch",
    "write_batch_v2",
]
