"""
Visualize sampled diffusion trajectories in MeshCat.

This script:
1. Loads a trained diffusion model
2. Samples trajectories for a given start configuration and goal
3. Visualizes the trajectories in MeshCat with the robot and environment
"""

import pathlib

import numpy as np
import pinocchio as pin

from deburring_diffusion.robot.traj_gen_utils import store_results
from deburring_diffusion.robot.visualization_utils import (
    load_diffusion_model,
    prepare_conditioning,
    sample_trajectories,
    setup_environment,
    setup_scene,
    visualize_trajectory,
)

# Configuration
CHECKPOINT_PATH = (
    # "/workspaces/deburring_diffusion/results/diffusion/"
    # # "lightning_logs/version_0/checkpoints/epoch=99-step=700.ckpt" This is for single goal single conf
    "/workspaces/deburring_diffusion/results/diffusion/lightning_logs/version_1/checkpoints/epoch=499-step=19000.ckpt"
)


MODEL_PATH = pathlib.Path("/workspaces/deburring_diffusion/models")
OBJ_FILE = MODEL_PATH / "pylone.obj"

Q_START = np.array([0.0, -0.4, 0.0, -0.2, 0.0, 1.57, 0.79])
TARGET_POSITION = np.array([0.35, 0.4, 0.7])
PYLONE_POSE = pin.XYZQUATToSE3([0.45, -0.116, 0.739, 0.0, 0.0, 0.0, 1.0])

N_SAMPLES = 5
SEQ_LENGTH = 50


def main() -> None:
    """Main execution function."""
    print("=" * 70)
    print("Diffusion Trajectory Visualization in MeshCat")
    print("=" * 70)

    # 1. Setup environment
    print("\n[1/6] Setting up environment...")
    rmodel, rdata, cmodel, cdata, vmodel, vdata = setup_environment()

    # 2. Setup target and conditioning
    print("[2/6] Preparing target and conditioning...")
    target_se3 = pin.SE3(np.eye(3), TARGET_POSITION)
    target_xyzquat = pin.SE3ToXYZQUAT(target_se3)

    Q_START = pin.randomConfiguration(rmodel)

    cond_dict = prepare_conditioning(
        q_start=Q_START,
        target_se3=target_se3,
    )

    # 3. Load diffusion model
    print("[3/6] Loading diffusion model...")
    model = load_diffusion_model(CHECKPOINT_PATH)

    # 4. Sample trajectories
    print(f"[4/6] Sampling {N_SAMPLES} trajectory(ies)...")
    sampled_trajs = sample_trajectories(
        model=model,
        cond_dict=cond_dict,
        n_samples=N_SAMPLES,
        seq_length=SEQ_LENGTH,
        configuration_size=rmodel.nq,
    )

    # 5. Setup MeshCat scene
    print("[5/6] Setting up MeshCat scene...")
    scene, robot = setup_scene(
        rmodel=rmodel,
        rdata=rdata,
        vmodel=vmodel,
        vdata=vdata,
        obj_file=OBJ_FILE,
        target_se3=target_se3,
        pylone_pose=PYLONE_POSE,
    )
    robot[:] = Q_START

    print(f"\n{'=' * 70}")
    print("MeshCat visualization ready!")
    print(f"{'=' * 70}")
    print("Robot: Panda")
    print(f"Start configuration: {Q_START}")
    print(f"Target position: {TARGET_POSITION}")
    print(f"Trajectory length: {SEQ_LENGTH} steps")
    print(f"Number of samples: {N_SAMPLES}")
    print(f"{'=' * 70}\n")

    # 6. Visualize each trajectory
    print("[6/6] Visualizing trajectories...")

    for i in range(N_SAMPLES):
        print(f"\n{'=' * 70}")
        print(f"Trajectory {i + 1}/{N_SAMPLES}")
        print(f"{'=' * 70}")

        # Convert to numpy array
        traj_np = sampled_trajs[i].cpu().numpy()

        # Store results (optional)
        xs = [traj_np[t] for t in range(traj_np.shape[0])]
        result = store_results(xs, target_xyzquat, rmodel)

        # Visualize
        visualize_trajectory(
            robot=robot,
            trajectory=traj_np,
            rmodel=rmodel,
            pause_between_steps=True,
        )

        if i < N_SAMPLES - 1:
            input("\nPress Enter to view next trajectory...")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
