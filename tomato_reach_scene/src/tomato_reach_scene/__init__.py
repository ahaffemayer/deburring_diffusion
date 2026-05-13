"""Procedural scene generator for the tomato-reach experiment.

Produces STORM/cuRobo-compatible world yamls + grasp poses for a Franka
Panda + EKENABBEN shelf rig. All poses are in the robot base frame
(panda_link0). See README.md for conventions.

Public API
----------
- ``SceneConfig`` — knobs (k, r_swing, distance bounds, E yaw mode, seed).
- ``SceneSampler`` — sample one or many scenes.
- ``Scene`` — dataclass: primitives, grasp pose, params.
- ``ObjectPlacement`` — one primitive's geometry + pose in base.
- ``SceneParams`` — sampled scalars per scene.

Exporters
---------
- ``tomato_reach_scene.exporters.to_storm_yaml(scene, path)`` — STORM/cuRobo world yaml.

Constants
---------
- ``tomato_reach_scene.constants`` — frozen geometric facts (object dims,
  composite frame in base, shelf primitives, home joint config).
"""
from __future__ import annotations

from .scene import ObjectPlacement, Scene, SceneParams
from .sampler import SceneConfig, SceneSampler

__all__ = [
    "ObjectPlacement",
    "Scene",
    "SceneConfig",
    "SceneParams",
    "SceneSampler",
]
__version__ = "0.1.1"
