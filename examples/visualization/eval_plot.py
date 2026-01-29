"""
Sample trajectories from a trained diffusion model for robot motion planning.

This script:
1. Loads a Panda robot model
2. Loads a pre-trained diffusion model
3. Samples trajectories conditioned on start configuration and goal pose
4. Visualizes and saves the results
"""

import pathlib

import numpy as np
import pinocchio as pin

from deburring_diffusion.robot.visualization_utils import (
    load_diffusion_model,
    plot_trajectories,
    prepare_conditioning,
    process_and_store_results,
    sample_trajectories,
    setup_environment,
)

# Configuration constants
CHECKPOINT_PATH = (
    # "/workspaces/deburring_diffusion/results/diffusion/"
    # "lightning_logs/version_0/checkpoints/epoch=99-step=700.ckpt"
    "/workspaces/deburring_diffusion/results/diffusion/lightning_logs/version_1/checkpoints/epoch=499-step=19000.ckpt"
)
N_SAMPLES = 10
SEQ_LENGTH = 50
Q_START = np.array([0.0, -0.4, 0.0, -0.2, 0.0, 1.57, 0.79])
TARGET_POSITION = np.array([0.35, 0.4, 0.7])


def main() -> None:
    """Main execution function."""
    # Setup paths
    script_dir = pathlib.Path(__file__).parent.parent
    output_dir = script_dir / "results" / "diffusion_eval"
    plot_path = script_dir / "results" / "traj_generator" / "trajectory_output.png"

    # 1. Setup environment
    print("Setting up environment...")
    rmodel, rdata, _, _, _, _ = setup_environment()

    # 2. Load diffusion model
    print("Loading diffusion model...")
    model = load_diffusion_model(CHECKPOINT_PATH)

    # 3. Prepare conditioning
    target_se3 = pin.SE3(np.eye(3), TARGET_POSITION)
    target_xyzquat = pin.SE3ToXYZQUAT(target_se3)
    cond_dict = prepare_conditioning(Q_START, target_se3)

    # 4. Sample trajectories
    sampled_trajs = sample_trajectories(
        model=model,
        cond_dict=cond_dict,
        n_samples=N_SAMPLES,
        seq_length=SEQ_LENGTH,
        configuration_size=rmodel.nq,
    )

    # 5. Process and store results
    print("Processing results...")
    results = process_and_store_results(
        sampled_trajs=sampled_trajs,
        target_xyzquat=target_xyzquat,
        rmodel=rmodel,
        output_dir=output_dir,
    )

    # 6. Visualize results
    print("Creating visualization...")
    plot_trajectories(
        results=results,
        rmodel=rmodel,
        sampled_trajs=sampled_trajs,
        output_path=plot_path,
    )

    print("Done!")


if __name__ == "__main__":
    main()
