"""Example: sample one scene + dump it to a STORM yaml.

Run:
    python examples/generate_one.py [--seed N] [--k-mm K] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tomato_reach_scene import SceneConfig, SceneSampler
from tomato_reach_scene.exporters import to_storm_yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-mm", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=Path("world.yaml"))
    args = p.parse_args()

    sampler = SceneSampler(SceneConfig(seed=args.seed, k_mm=args.k_mm))
    scene = sampler.sample()

    print(f"hard_side: {scene.params.hard_side}")
    print(f"k_mm:      {scene.params.k_mm}")
    print(f"E:         {scene.params.e_xy_mm}  dist={scene.params.dist_e_mm:.1f} mm")
    print(f"C:         {scene.params.c_xy_mm}  dist={scene.params.dist_c_mm:.1f} mm")
    print(f"E_yaw:     {scene.params.e_yaw_deg:.1f} deg")
    print(f"grasp xyz: {scene.grasp_pose_xyz_m}")
    print(f"grasp quat: {scene.grasp_pose_quat_xyzw}")

    to_storm_yaml(scene, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
