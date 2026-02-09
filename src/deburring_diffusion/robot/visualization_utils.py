import pathlib
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import torch
from robomeshcat import Object, Robot, Scene

from deburring_diffusion.diffusion.model import Model
from deburring_diffusion.robot.panda_env_loader import load_fer
from deburring_diffusion.robot.traj_gen_utils import (
    from_trajectory_to_ee_poses,
    store_results,
)


def setup_environment() -> tuple:
    """Load robot model and create data structures.

    Returns:
        tuple: (rmodel, rdata) - robot model and robot data
    """
    rmodel, cmodel, vmodel = load_fer()
    rdata = rmodel.createData()
    vdata = vmodel.createData()
    cdata = cmodel.createData()
    return rmodel, rdata, cmodel, cdata, vmodel, vdata


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

        ee_poses = from_trajectory_to_ee_poses(
            rmodel, xs, "fer_ati_mini45_with_compact_camera_tool"
        )
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


def setup_scene(
    rmodel: pin.Model,
    rdata: pin.Data,
    vmodel: pin.GeometryModel,
    vdata: pin.GeometryData,
    obj_file: pathlib.Path,
    target_se3: pin.SE3,
    pylone_pose: pin.SE3,
) -> tuple[Scene, Robot]:
    """Setup MeshCat scene with robot and objects.

    Args:
        rmodel: Robot model
        rdata: Robot data
        vmodel: Visual model
        vdata: Visual data
        obj_file: Path to pylone mesh file
        target_se3: Target end-effector pose
        pylone_pose: Pylone object pose

    Returns:
        Tuple of (scene, robot)
    """
    # Create robot
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )

    # Create scene
    scene = Scene()
    scene.add_robot(robot)

    # Add pylone object
    pylone = Object.create_mesh(
        path_to_mesh=obj_file,
        name="robot/movable_obj",
        scale=1.0,
    )
    scene.add_object(pylone)
    pylone.pose = pylone_pose.homogeneous

    # Add environment primitives
    pylone_to_box = pin.SE3(np.eye(3), np.array([0.01, 0.015, -0.50]))
    base_box = Object.create_cuboid(
        lengths=[0.5, 0.6, 0.435],
        name="robot/base_box",
        color=[0.0, 0.5, 0.5],
    )
    scene.add_object(base_box)
    base_box.pose = (pylone_pose * pylone_to_box).homogeneous

    table = Object.create_cuboid(
        lengths=[2.0, 2.0, 0.2],
        name="robot/table",
        color=[0.5, 0.5, 0.0],
    )
    scene.add_object(table)
    table.pose[2, 3] = -0.15

    # Add goal sphere
    goal = Object.create_sphere(
        radius=0.02,
        name="goal_sphere",
        color=[0, 1, 0],  # Green with transparency
    )
    scene.add_object(goal)
    goal.pose = target_se3.homogeneous

    return scene, robot


def visualize_trajectory(
    robot: Robot,
    trajectory: np.ndarray,
    rmodel: pin.Model,
    pause_between_steps: bool = True,
) -> None:
    """Visualize a single trajectory in MeshCat.

    Args:
        robot: Robot object
        trajectory: Trajectory as numpy array (seq_length, config_size)
        rmodel: Robot model
        pause_between_steps: If True, pause after each step
    """
    print(f"\nVisualizing trajectory with {len(trajectory)} steps")
    print("Press Enter to advance through trajectory...")

    for step_idx, q in enumerate(trajectory):
        # Update robot configuration
        robot[:] = q[: rmodel.nq]

        if pause_between_steps:
            input(f"Step {step_idx + 1}/{len(trajectory)} - Press Enter to continue...")
