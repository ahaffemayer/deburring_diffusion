import pathlib
from typing import List

import hppfcl
import numpy as np
import pinocchio as pin
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)


def get_device_args():
    # choose CUDA if available, otherwise CPU
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return TensorDeviceType(device=dev)


tensor_args = get_device_args()


def parser_collision_model(collision_model) -> dict:
    """Returns a dict of the like:
        world_config = {
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],
            },
            "obs_1": {
                "dims": [0.2, 0.2, 0.4],
                "pose": [0.4, 0.0, 0.2, 1, 0, 0, 0],
            },
            "obs_2": {
                "dims": [0.15, 0.15, 0.3],
                "pose": [0.2, -0.15, 0.15, 1, 0, 0, 0],
            },
        },

    }

    Args:
        collision_model (_type_): _description_
    """
    world_config = {
        "cuboid": {},
    }

    for obj in collision_model.geometryObjects:
        if isinstance(obj.geometry, hppfcl.Box):
            name = obj.name
            pose_se3 = obj.placement
            xyzquat = pin.SE3ToXYZQUAT(pose_se3)
            translation = [float(v) for v in xyzquat[:3]]
            rotation = [float(v) for v in xyzquat[3:]]
            qx, qy, qz, qw = rotation  # Pinocchio uses (qx, qy, qz, qw) order
            world_config["cuboid"][name] = {
                "dims": [float(v) for v in (obj.geometry.halfSide * 2)],
                "pose": [
                    translation[0],
                    translation[1],
                    translation[2],
                    float(qw),
                    float(qx),
                    float(qy),
                    float(qz),
                ],
            }

    return world_config


def create_motion_gen_curobo(
    pylone_pose: pin.SE3 = pin.SE3.Identity(),
    obj_file: pathlib.Path = pathlib.Path(""),
    use_custom_panda: bool = False,
) -> MotionGen:
    """Create and return a curobo MotionGen instance for the Franka robot."""
    if isinstance(pylone_pose, pin.SE3):
        pylone_pose_quat = pin.SE3ToXYZQUAT(pylone_pose)  # (x, y, z, qx, qy, qz, qw)
        # Convert to (x, y, z, qw, qx, qy, qz)
        pylone_pose_list = [
            float(pylone_pose_quat[0]),
            float(pylone_pose_quat[1]),
            float(pylone_pose_quat[2]),
            float(pylone_pose_quat[6]),
            float(pylone_pose_quat[3]),
            float(pylone_pose_quat[4]),
            float(pylone_pose_quat[5]),
        ]
    else:
        raise ValueError("pylone_pose must be a pin.SE3 instance.")

    if not obj_file.exists():
        raise FileNotFoundError(f"Object file not found: {obj_file.as_posix()}")
    world_config = {
        "mesh": {
            "scene": {
                "pose": pylone_pose_list,
                "file_path": obj_file.as_posix(),
            }
        },
    }
    print("World config for Curobo MotionGen:", world_config)
    if not use_custom_panda:
        robot_file = "franka.yml"  # assumes this file is in your curobo robot configs
    else:
        package_dir = pathlib.Path(__file__).parent.parent.parent.parent
        robot_file = (package_dir / "models" / "fer" / "fer.yaml").as_posix()
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        tensor_args,
        interpolation_dt=0.02,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=4,
        num_ik_seeds=50,
        collision_activation_distance=0.01,
    )
    # --- create MotionGen and warm it up ---
    motion_gen = MotionGen(motion_gen_cfg)
    # motion_gen.warmup()  # warms GPU kernels, IK solvers, etc

    return motion_gen


def create_motion_gen_plan_config():
    reach_vec_weight = tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    pose_metric = PoseCostMetric(
        reach_partial_pose=False,
        reach_full_pose=True,  # do not enforce full pose
        reach_vec_weight=reach_vec_weight,
    )
    plan_cfg = MotionGenPlanConfig(
        max_attempts=3, timeout=6.0, pose_cost_metric=pose_metric
    )
    return plan_cfg


def plan_with_curobo(
    motion_gen: MotionGen,
    q_start: np.ndarray,
    target_pose: pin.SE3,
    plan_cfg: MotionGenPlanConfig,
) -> List[np.ndarray]:
    """Plan a trajectory using curobo's motion generation. Normally here we have a SE3 but only the position matters.
    1. Run create_motion_gen_curobo to create the motion_gen
    2. Run create_motion_gen_plan_config to create the plan_cfg
    3. Call this function with the motion_gen, start configuration, target pose, and plan_cfg
    Returns the interpolated trajectory as a list of numpy arrays."""

    q_start_torch = torch.tensor(
        q_start, device=tensor_args.device, dtype=torch.float32
    )
    start_state = JointState.from_position(q_start_torch.view(1, -1))

    goal = pin.SE3ToXYZQUAT(target_pose)
    goal_quat = goal[-4:]
    w, x, y, z = goal_quat[3], goal_quat[0], goal_quat[1], goal_quat[2]
    goal_pose_curobo = Pose(
        position=torch.tensor(
            target_pose.translation, device=tensor_args.device, dtype=torch.float32
        ).unsqueeze(0),
        quaternion=torch.tensor(
            [w, x, y, z], device=tensor_args.device, dtype=torch.float32
        ).unsqueeze(0),  # Different convention for quaternions
    )

    result = motion_gen.plan_single(start_state, goal_pose_curobo, plan_cfg)
    if bool(result.success):
        interp = result.get_interpolated_plan()  # trajectory object
        # show some shapes and the first few joint positions
        return interp
    else:
        raise RuntimeError("Curobo planning failed.")


def resample_trajectory(q_traj: np.ndarray, T: int) -> np.ndarray:
    """
    q_traj: ndarray of shape (N, dof)
    T: desired number of timesteps
    Returns an ndarray of shape (T, dof)
    """
    N = q_traj.shape[0]
    dof = q_traj.shape[1]

    if N == T:
        return q_traj.copy()

    # Time indices of original and new trajectories
    original_idx = np.linspace(0, 1, N)
    target_idx = np.linspace(0, 1, T)

    q_resampled = np.zeros((T, dof), dtype=q_traj.dtype)

    for j in range(dof):
        q_resampled[:, j] = np.interp(target_idx, original_idx, q_traj[:, j])

    # Ensure exact matching of start and end
    q_resampled[0] = q_traj[0]
    q_resampled[-1] = q_traj[-1]

    assert q_resampled.shape == (T, dof)
    return q_resampled
