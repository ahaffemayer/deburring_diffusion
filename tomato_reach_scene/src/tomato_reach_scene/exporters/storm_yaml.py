"""STORM / cuRobo world yaml writer.

Schema:

```yaml
world_model:
  coll_objs:
    cube:
      <name>:
        dims:  [x, y, z]                                    # full extents (m)
        pose:  [x, y, z, qx, qy, qz, qw]                    # robot base frame
      …
    cylinder:
      <name>:
        radius: <r>                                         # m
        height: <h>                                         # m
        pose:   [x, y, z, qx, qy, qz, qw]                   # robot base frame
      …
    sphere: {}
```

All poses are in the **robot base frame** (``panda_link0``). The
sphere bucket is emitted empty for schema completeness.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from ..scene import Scene


def to_storm_yaml(scene: Scene, path: Path | str) -> None:
    """Serialise one :class:`Scene` to a STORM-format world yaml.

    Overwrites ``path`` if it exists.
    """
    cubes: dict[str, dict] = {}
    cylinders: dict[str, dict] = {}
    for name, p in scene.primitives.items():
        pose = list(p.pose_xyz_m) + list(p.pose_quat_xyzw)
        if p.type == "box":
            cubes[name] = {"dims": list(p.dims_m), "pose": pose}
        elif p.type == "cylinder":
            cylinders[name] = {
                "radius": p.dims_m[0],
                "height": p.dims_m[1],
                "pose": pose,
            }
        else:  # pragma: no cover - exhaustive by type
            raise ValueError(f"unsupported primitive type {p.type!r} on {name!r}")

    world = {
        "world_model": {
            "coll_objs": {
                "cube": cubes,
                "cylinder": cylinders,
                "sphere": {},
            }
        }
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(world, f, sort_keys=False)


def write_batch(
    scenes: Iterable[Scene],
    out_dir: Path | str,
    *,
    name_prefix: str = "variant",
    write_manifest: bool = True,
) -> Path:
    """Write a batch of scenes to ``out_dir`` plus a manifest.

    Layout::

        out_dir/
          variant_00.yaml             # STORM world per scene
          variant_01.yaml
          ...
          manifest.yaml               # per-scene parameters + grasp poses

    Returns
    -------
    Path
        ``out_dir`` (as a Path) for chaining.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_list = list(scenes)
    manifest_entries = []
    for i, scene in enumerate(scenes_list):
        stem = f"{name_prefix}_{i:02d}"
        to_storm_yaml(scene, out_dir / f"{stem}.yaml")
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
                "in STORM format; poses in panda_link0 frame; grasp pose "
                "references the panda_hand_tcp link. The `hard_side` "
                "label is determined by clearance from the composite "
                "origin: the close obstacle is within "
                "[d_min, r_swing - clearance_buffer]; the far obstacle "
                "is within [r_swing + clearance_buffer, d_max]. A buffer "
                "around r_swing is excluded so that 'hard' and 'easy' "
                "are unambiguous in every scene."
            ),
            "count": len(scenes_list),
            "variants": manifest_entries,
        }
        with open(out_dir / "manifest.yaml", "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)

    return out_dir
