# tomato-reach-scene

Procedural scene generator for the tomato-reach experiment on a Franka
Panda + EKENÄBBEN shelf rig. Produces STORM-format world yamls plus
per-scene grasp poses, ready to drive a motion-planning pipeline.

Designed for use **alongside the bundled Panda URDF** so the trajectories
your planner emits are valid on the calibrating lab's real hardware.

## Quickstart

```python
from tomato_reach_scene import SceneConfig, SceneSampler
from tomato_reach_scene.exporters import to_storm_yaml, write_batch

# Sample one scene
cfg = SceneConfig(k_mm=60.0, seed=42)
sampler = SceneSampler(cfg)
scene = sampler.sample()

# Inspect
print(scene.params)
# SceneParams(k_mm=60.0, hard_side='right', e_xy_mm=(...), c_xy_mm=(...),
#             e_yaw_deg=..., dist_e_mm=..., dist_c_mm=...)

print(scene.grasp_pose_xyz_m)        # (x, y, z) in panda_link0 frame
print(scene.grasp_pose_quat_xyzw)    # (qx, qy, qz, qw)

# Pick your downstream format (both consume the same Scene object):
from tomato_reach_scene.exporters import (
    to_storm_yaml,        # STORM / cuRobo V1 (≤ v0.7.x) format
    to_curobo_v2_yaml,    # cuRobo V2 (≥ v0.8.0) format
)

to_storm_yaml(scene, "world_v1.yaml")
to_curobo_v2_yaml(scene, "world_v2.yaml")

# Batch versions of either:
from tomato_reach_scene.exporters import write_batch, write_batch_v2
scenes = sampler.sample_batch(n=100)
write_batch(scenes,    out_dir="./scenes_v1_2026-05-12")
write_batch_v2(scenes, out_dir="./scenes_v2_2026-05-12")
# Each writes {variant_NN.yaml, manifest.yaml}.
```

## Installation

```bash
git clone <this-repo> && cd tomato-reach-scene
pip install -e .
```

Dependencies: `numpy`, `pyyaml`. Python ≥ 3.10.

## What you get per scene

Each `Scene` has three load-bearing parts:

### 1. Collision primitives (`scene.primitives`)

A dict of 12 `ObjectPlacement` entries:

| Group | Names | Count |
|---|---|---|
| Shelf (EKENÄBBEN) | `shelf_slab_{lower,middle,upper}`, `shelf_column_{front,back}_{left,right}` | 7 |
| Scene objects | `A_rice_noodle`, `B_tomato`, `C_aqua_bottle`, `D_soup_can`, `E_cocopops` | 5 |

Each entry carries:
- `type`: `"box"` or `"cylinder"`.
- `dims_m`: full extents (for boxes) or `(radius, height, 0)` (for cylinders).
- `pose_xyz_m`: centre translation in **robot base frame** (`panda_link0`), metres.
- `pose_quat_xyzw`: rotation quaternion in robot base frame.

### 2. Grasp pose (`scene.grasp_pose_xyz_m`, `scene.grasp_pose_quat_xyzw`)

6-DoF goal pose for the gripper TCP (`panda_hand_tcp` link) in the
robot base frame. Canonical top-down for v1:
- Position: above B's (tomato) top face by `grasp_approach_offset_m` (5 mm default).
- Orientation: gripper z-axis pointing in base −z (top-down approach);
  gripper x-axis aligned with base +x.

### 3. Sampling parameters (`scene.params`)

The scalars that produced this scene. Useful for manifest joins / debugging:
- `k_mm`: gap between A and D (fixed by config).
- `hard_side`: `"left"` or `"right"` — which obstacle is the close one.
- `e_xy_mm`, `c_xy_mm`: E and C centres in the **composite frame** (mm).
- `e_yaw_deg`: E's rotation about its z-axis.
- `dist_e_mm`, `dist_c_mm`: Euclidean distances from composite origin.

## Coordinate conventions

| Convention | Value |
|---|---|
| World frame | `panda_link0` (the robot's base) |
| Robot's position in that frame | identity (the world *is* `panda_link0`) |
| All `pose_xyz_m` in `Scene` | robot base frame |
| Grasp pose's reference link | `panda_hand_tcp` (standard Franka TCP) |
| Joint config order | `panda_joint1` … `panda_joint7` (see `robot/home_config.yaml`) |

**Composite frame** (an internal helper, exposed if you need it):
the scene primitives sit at the geometric center of the EKENÄBBEN's
2nd shelf-slab top surface, with axes aligned to the robot base.
Translation is hard-coded as a frozen constant captured from the
calibrating lab's rig (`COMPOSITE_ORIGIN_IN_BASE_M` in `constants.py`).

## Bundled assets

- `robot/urdf/panda.urdf` — the URDF that matches the real rig. **Use
  this**, not a vanilla `franka_description` URDF, if you intend to
  send trajectories back to be executed.
- `robot/urdf/panda_collision.urdf` — collision-only URDF (lighter; for
  collision queries).
- `robot/meshes/` — visual + collision meshes referenced by the URDFs.
- `robot/home_config.yaml` — home / start joint configuration. The
  Franka standard "ready" pose used by the real-robot `go_home` routine.

## Algorithm summary

For full details see the source repo's
`docs/experiments/tomato_reach_scene.md`. Tl;dr:

1. `k` (gap A↔D) is **fixed** by `SceneConfig.k_mm`. A and D are placed
   on the composite x-axis straddling the origin (gap midpoint at origin).
2. Per scene, a **fair coin** picks `hard_side ∈ {left, right}`. The
   designated hard-side obstacle is sampled with distance ∈
   `[d_min, r_swing - clearance_buffer]`; the easy-side obstacle with
   distance ∈ `[r_swing + clearance_buffer, d_max]`. The buffer's
   forbidden zone around `r_swing` rules out scenes where both obstacles
   straddle the threshold (which would be pedagogically ambiguous).
   With the buffer in place, every scene has a clearly blocking hard
   side and a clearly clear easy side.
3. E (the cocopops box) has its yaw sampled uniformly in `[0°, 180°)`
   (or fixed via `SceneConfig.e_yaw_mode = "fixed"` + `e_yaw_deg=…`).
4. Rejection sampling enforces non-overlap (box ↔ circle SDF and
   circle ↔ circle for cylinder pairs) and the slab footprint.

Empirical: ~3-4 retries per accepted scene on the default knobs.

## Knobs

All on `SceneConfig`:

| Knob | Default | Notes |
|---|---:|---|
| `k_mm` | 60.0 | Range `[10, 128.8]` (geometric bound from slab depth and A's diameter). |
| `r_swing_mm` | 125.0 | Estimated clearance threshold around the central column. Derived from Panda URDF (r_A + hand half-thickness + safety margin). |
| `clearance_buffer_mm` | 15.0 | Buffer carved around `r_swing_mm` within which no obstacle centre may land. Ensures every scene has a *clearly* blocking obstacle on one side and a *clearly* clear obstacle on the other (no borderline-on-both-sides scenes). Set to 0 to disable. |
| `d_min_mm` | 100.0 | Must be ≤ `r_swing_mm - clearance_buffer_mm`. |
| `d_max_mm` | 300.0 | Must be ≥ `r_swing_mm + clearance_buffer_mm`; capped by slab extent in practice. |
| `both_easy_probability` | **0.33** | Probability that a scene has **no hard side** — both obstacles sampled from the easy band. The remaining `1 − p` go through the usual left/right coin flip. The default 0.33 gives a roughly even distribution across the three scene classes (`left`, `right`, `neither`) so the downstream policy sees all cases in similar proportion during training. Set to 0 if you want the old two-case behaviour. Scenes carry `hard_side: "neither"` in the manifest. |
| `e_yaw_mode` | `"random"` | Or `"fixed"` to pin E's orientation. |
| `e_yaw_deg` | 0.0 | Used only when mode is `"fixed"`. |
| `seed` | 0 | Same seed + same knobs ⇒ identical scene sequence. |
| `max_tries_per_scene` | 500 | Rejection-sampling retry budget. |
| `grasp_approach_offset_m` | 0.005 | metres above B's top face where the TCP stops. |

## Output yaml formats

The package supports **both cuRobo lineage formats**. Pick the one your
downstream planner consumes.

### V1 (STORM / cuRobo ≤ 0.7.x) — `to_storm_yaml`

```yaml
world_model:
  coll_objs:
    cube:
      shelf_slab_lower:
        dims: [0.7, 0.355, 0.015]
        pose: [x, y, z, qx, qy, qz, qw]   # quaternion order: xyzw
    cylinder:
      A_rice_noodle:
        radius: 0.056485
        height: 0.120
        pose: [x, y, z, qx, qy, qz, qw]
    sphere: {}
```

### V2 (cuRobo ≥ 0.8.0) — `to_curobo_v2_yaml`

```yaml
cuboid:                                   # NB: 'cuboid', not 'cube'
  shelf_slab_lower:
    dims: [0.7, 0.355, 0.015]
    pose: [x, y, z, qw, qx, qy, qz]       # quaternion order: wxyz
cylinder:
  A_rice_noodle:
    radius: 0.056485
    height: 0.120
    pose: [x, y, z, qw, qx, qy, qz]
```

Three differences:
- No top-level `world_model.coll_objs` wrapper.
- Boxes keyed under **`cuboid`**, not `cube`.
- Pose quaternion is **`wxyz`** order (qw first), not `xyzw`.

Both formats use the **robot base frame** (`panda_link0`) for all
poses. The in-memory `Scene` is format-agnostic; you choose at export.

## Versioning

The frozen calibration constants (`COMPOSITE_ORIGIN_IN_BASE_M`,
`SHELF_PRIMITIVES_BASE`) are captured from a specific calibration
session at the source repo. If the lab re-pins the shelf, a new
release of this package is needed. The version (in `pyproject.toml`)
tracks calibration generation.

## License

MIT.
