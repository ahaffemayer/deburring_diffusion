"""cuRobo V2 (≥ 0.8.0) world yaml writer.

V2's scene yaml schema differs from the legacy V1 / STORM format in three
small but load-bearing ways:

1. No top-level ``world_model.coll_objs`` envelope — primitives sit at
   the root of the document.
2. Boxes are keyed under ``cuboid:`` (not ``cube:``).
3. Pose quaternions use **``wxyz`` order** (qw first), not the V1
   ``xyzw`` order.

Schema:

```yaml
cuboid:
  <name>:
    dims:  [x, y, z]                                    # full extents (m)
    pose:  [x, y, z, qw, qx, qy, qz]                    # robot base frame
  …
cylinder:
  <name>:
    radius: <r>                                         # m
    height: <h>                                         # m
    pose:   [x, y, z, qw, qx, qy, qz]                   # robot base frame
  …
```

Spheres are not emitted by the tomato-reach scene generator (no
spherical primitives in the scene), but the V2 sphere schema is
``{position: [x, y, z], radius: <r>}`` — no pose quaternion since
spheres are isotropic.

Reference: ``curobo.types.Pose.from_list`` (cuRobo v0.8.0) defaults to
``q_xyzw=False`` → the slice ``pose[3:]`` is interpreted as
``[qw, qx, qy, qz]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from ..scene import Scene


def to_curobo_v2_yaml(scene: Scene, path: Path | str) -> None:
    """Serialise one :class:`Scene` to a cuRobo V2 world yaml.

    Overwrites ``path`` if it exists.
    """
    cuboids: dict[str, dict] = {}
    cylinders: dict[str, dict] = {}
    for name, p in scene.primitives.items():
        # V2 expects [x, y, z, qw, qx, qy, qz]; our scene carries qxyzw.
        qx, qy, qz, qw = p.pose_quat_xyzw
        pose_v2 = [*p.pose_xyz_m, qw, qx, qy, qz]
        if p.type == "box":
            cuboids[name] = {"dims": list(p.dims_m), "pose": pose_v2}
        elif p.type == "cylinder":
            cylinders[name] = {
                "radius": p.dims_m[0],
                "height": p.dims_m[1],
                "pose": pose_v2,
            }
        else:  # pragma: no cover - exhaustive by type
            raise ValueError(f"unsupported primitive type {p.type!r} on {name!r}")

    world: dict[str, dict] = {}
    if cuboids:
        world["cuboid"] = cuboids
    if cylinders:
        world["cylinder"] = cylinders

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(world, f, sort_keys=False)


def write_batch_v2(
    scenes: Iterable[Scene],
    out_dir: Path | str,
    *,
    name_prefix: str = "variant",
    write_manifest: bool = True,
) -> Path:
    """Write a batch of scenes to ``out_dir`` in cuRobo V2 format + manifest.

    Mirrors :func:`tomato_reach_scene.exporters.storm_yaml.write_batch`
    but emits V2-schema yamls instead of V1 STORM. Manifest schema is
    the same.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_list = list(scenes)
    manifest_entries = []
    for i, scene in enumerate(scenes_list):
        stem = f"{name_prefix}_{i:02d}"
        to_curobo_v2_yaml(scene, out_dir / f"{stem}.yaml")
        manifest_entries.append({
            "name": stem,
            "world_file": f"{stem}.yaml",
            "k_mm": scene.params.k_mm,
            "hard_side": scene.params.hard_side,
            "E": {
                "x_mm": scene.params.e_xy_mm[0],
                "y_mm": scene.params.e_xy_mm[1],
                "dist_mm": scene.params.dist_e_mm,
                "yaw_deg": scene.params.e_yaw_deg,
            },
            "C": {
                "x_mm": scene.params.c_xy_mm[0],
                "y_mm": scene.params.c_xy_mm[1],
                "dist_mm": scene.params.dist_c_mm,
            },
            "grasp_pose_base": {
                "translation_m": list(scene.grasp_pose_xyz_m),
                "quaternion_xyzw": list(scene.grasp_pose_quat_xyzw),
                "reference_link": "panda_hand_tcp",
            },
        })

    if write_manifest:
        manifest = {
            "description": (
                "Procedurally-generated tomato-reach scenes. World yamls "
                "in cuRobo V2 format (cuboid:/cylinder:/sphere: at root; "
                "pose quaternion in wxyz order); poses in panda_link0 "
                "frame; grasp pose references the panda_hand_tcp link. "
                "Grasp pose quaternion in this manifest is xyzw to match "
                "the in-memory Scene; reorder to wxyz if your consumer "
                "expects it."
            ),
            "format": "curobo_v2",
            "count": len(scenes_list),
            "variants": manifest_entries,
        }
        with open(out_dir / "manifest.yaml", "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)

    return out_dir
