from typing import Tuple, List, Dict
import json
import pathlib
import random

import torch
import numpy as np
import pinocchio as pin
import hppfcl
from robomeshcat import Scene, Object, Robot

from deburring_diffusion.robot.visualizer import (
    create_viewer,
    add_sphere_to_viewer,
)
from deburring_diffusion.robot.panda_env_loader import (
    load_reduced_panda,
    robot_links,
)
from deburring_diffusion.robot.curobo_utils import get_device_args

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.util_file import (
    join_path,
    load_yaml,
    get_robot_configs_path,
    get_world_configs_path,
)
from curobo.rollout.cost.pose_cost import PoseCostMetric


def create_motion_gen_plan_config():

    reach_vec_weight = tensor_args.to_device([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    reach_vec_weight = tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    pose_metric = PoseCostMetric(
        reach_partial_pose=True,
        reach_full_pose=False,  # do not enforce full pose
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
        ).unsqueeze(
            0
        ),  # Different convention for quaternions
    )

    result = motion_gen.plan_single(start_state, goal_pose_curobo, plan_cfg)
    if bool(result.success):
        interp = result.get_interpolated_plan()  # trajectory object
        # show some shapes and the first few joint positions
        return interp
    else:
        raise RuntimeError("Curobo planning failed.")


def create_motion_gen_curobo(
    cmodel: pin.GeometryModel,
    pylone_pose: pin.SE3 = pin.SE3.Identity(),
    obj_file: pathlib.Path = pathlib.Path(""),
) -> MotionGen:
    """Create and return a curobo MotionGen instance for the Franka robot."""
    if isinstance(pylone_pose, pin.SE3):
        pylone_pose_quat = pin.SE3ToXYZQUAT(pylone_pose) # (x, y, z, qx, qy, qz, qw)
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
    robot_file = "franka.yml"  # assumes this file is in your curobo robot configs
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        tensor_args,
        interpolation_dt=0.02,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=4,
        num_ik_seeds=50,
        collision_activation_distance=0.01
    )
    # --- create MotionGen and warm it up ---
    motion_gen = MotionGen(motion_gen_cfg)
    # motion_gen.warmup()  # warms GPU kernels, IK solvers, etc

    return motion_gen


def create_motion_gen_plan_config():

    reach_vec_weight = tensor_args.to_device([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    reach_vec_weight = tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    pose_metric = PoseCostMetric(
        reach_partial_pose=True,
        reach_full_pose=False,  # do not enforce full pose
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
        ).unsqueeze(
            0
        ),  # Different convention for quaternions
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


if __name__ == "__main__":
    tensor_args = get_device_args()
    print("Using device:", tensor_args.device)

    # Setting up the scene with the robot and the pylone object
    obj_path = pathlib.Path(__file__).parent.parent / "models"
    obj_file = obj_path / "pylone.obj"
    pylone_pose = pin.XYZQUATToSE3([0.45, -0.116, 0.739, 0.0, 0.0, 0.0, 1.0])
    target = pin.SE3(np.eye(3), np.array([0.35, -0.1, 0.7]))

    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    vdata = vmodel.createData()
    
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )
    scene = Scene()
    scene.add_robot(robot)
    o = Object.create_mesh(
        path_to_mesh=obj_file,
        name="robot/movable_obj",
        scale=1.0,
    )
    scene.add_object(o)
    o.pose = pylone_pose.homogeneous
    
    goal = Object.create_sphere(radius=0.02, name="goal_sphere", color=[0, 1, 0])
    scene.add_object(goal)
    goal.pose = target.homogeneous
    
    
    motion_gen = create_motion_gen_curobo(cmodel, pylone_pose=pylone_pose, obj_file=obj_file)
    plan_cfg = create_motion_gen_plan_config()

    inp = 0
    while inp != "ok":
        q_start = pin.randomConfiguration(rmodel)
        robot[:] = q_start
        inp = input("Type 'ok' to start planning: ")
    traj = plan_with_curobo(motion_gen, q_start, target, plan_cfg)

    resampled_traj = resample_trajectory(
        traj.position.cpu().numpy(), T=50
    )
    xs = [resampled_traj[t] for t in range(resampled_traj.shape[0])]
    
    

    
    for x in xs:
        q = x[: rmodel.nq]
        robot[:] = q
        input("Press Enter to continue...")
    
    