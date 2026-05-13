"""Example: sample N scenes + dump them to a directory with a manifest.

Run:
    python examples/generate_batch.py --n 100 --out ./scenes
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tomato_reach_scene import SceneConfig, SceneSampler
from tomato_reach_scene.exporters import write_batch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-mm", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=Path("./scenes"))
    args = p.parse_args()

    sampler = SceneSampler(SceneConfig(seed=args.seed, k_mm=args.k_mm))
    scenes = sampler.sample_batch(n=args.n)
    out_dir = write_batch(scenes, args.out)
    print(f"wrote {args.n} scenes + manifest to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
