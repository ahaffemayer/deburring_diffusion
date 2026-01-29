import pathlib
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import torch

from deburring_diffusion.diffusion.model import Model
from deburring_diffusion.robot.panda_env_loader import load_reduced_panda
from deburring_diffusion.robot.traj_gen_utils import (
    from_trajectory_to_ee_poses,
    store_results,
)


def setup_environment() -> tuple:
    """Load robot model and create data structures.

    Returns:
        tuple: (rmodel, rdata) - robot model and robot data
    """
    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    vdata = vmodel.createData()
    return rmodel, rdata, vmodel, vdata


def load_diffusion_model(checkpoint_path: str) -> Model:
    """Load and prepare the trained diffusion model.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Model: Loaded model in evaluation mode on GPU
    """
    model = Model.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()
    return model


def prepare_conditioning(
    q_start: np.ndarray, target_se3: pin.SE3
) -> Dict[str, torch.Tensor]:
    """Prepare conditioning tensor for diffusion sampling.

    Args:
        q_start: Initial joint configuration (7,)
        target_se3: Target end-effector pose

    Returns:
        Dictionary with 'q0' and 'goal' tensors on GPU
    """
    target_xyzquat = pin.SE3ToXYZQUAT(target_se3)

    cond_dict = {
        "q0": torch.from_numpy(q_start).float().unsqueeze(0).cuda(),
        "goal": torch.from_numpy(target_xyzquat).float().unsqueeze(0).cuda(),
    }

    return cond_dict


def sample_trajectories(
    model: Model,
    cond_dict: Dict[str, torch.Tensor],
    n_samples: int,
    seq_length: int,
    configuration_size: int,
) -> torch.Tensor:
    """Sample trajectories from the diffusion model.

    Args:
        model: Trained diffusion model
        cond_dict: Conditioning dictionary
        n_samples: Number of trajectories to generate
        seq_length: Length of each trajectory
        configuration_size: Size of robot configuration

    Returns:
        Sampled trajectories tensor
    """
    print(f"Sampling {n_samples} trajectories from diffusion model...")

    with torch.no_grad():
        sampled_trajs = model.sample(
            cond=cond_dict,
            bs=n_samples,
            seq_length=seq_length,
            configuration_size=configuration_size,
        )

    return sampled_trajs


def process_and_store_results(
    sampled_trajs: torch.Tensor,
    target_xyzquat: np.ndarray,
    rmodel: pin.Model,
    output_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    """Process sampled trajectories and store results.

    Args:
        sampled_trajs: Sampled trajectory tensor
        target_xyzquat: Target pose in [x, y, z, qx, qy, qz, qw] format
        rmodel: Robot model
        output_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = sampled_trajs.shape[0]
    results = []

    for i in range(n_samples):
        traj_np = sampled_trajs[i].cpu().numpy()
        xs = [traj_np[t] for t in range(traj_np.shape[0])]

        result = store_results(xs, target_xyzquat, rmodel)
        results.append(result)

    return results


def plot_trajectories(
    results: List[Dict[str, Any]],
    rmodel: pin.Model,
    sampled_trajs: torch.Tensor,
    output_path: pathlib.Path,
) -> None:
    """Plot and save end-effector trajectories.

    Args:
        results: List of trajectory results
        rmodel: Robot model
        sampled_trajs: Sampled trajectory tensor
        output_path: Path to save the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each trajectory
    for i, result in enumerate(results):
        traj_np = sampled_trajs[i].cpu().numpy()
        xs = [traj_np[t] for t in range(traj_np.shape[0])]

        ee_poses = from_trajectory_to_ee_poses(rmodel, xs)
        ee_xyz = np.array([pose.translation for pose in ee_poses])

        ax.plot(
            ee_xyz[:, 0],
            ee_xyz[:, 1],
            ee_xyz[:, 2],
            label=f"Trajectory {i + 1}" if len(results) > 1 else "Trajectory",
        )

    # Plot target
    target = np.array(results[0]["target"][:3])
    ax.scatter(
        target[0],
        target[1],
        target[2],
        marker="x",
        s=100,
        c="red",
        label="Target",
    )

    # Format plot
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-Effector Trajectories")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")
