"""Standalone batched generator for tomato pick-place training data.

It generates:
- multiple tomato shelf scenes
- random place poses on the upper shelf
- random initial robot configurations
- multiple joint-space modes for each fixed scene/place target
- collision-aware cuRobo trajectories in real batches
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys
from dataclasses import asdict
from typing import Any, NamedTuple

import numpy as np
import pinocchio as pin
import torch
import yaml
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cylinder
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from deburring_diffusion.robot.curobo_utils import get_device_args, resample_trajectory


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOMATO_SCENE_ROOT = REPO_ROOT / "tomato_reach_scene"
TOMATO_SCENE_SRC = TOMATO_SCENE_ROOT / "src"
if TOMATO_SCENE_SRC.exists():
    sys.path.insert(0, TOMATO_SCENE_SRC.as_posix())

from tomato_reach_scene import Scene, SceneConfig, SceneSampler  # noqa: E402
from tomato_reach_scene import constants as tomato_constants  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "results" / "traj_generator"
DEFAULT_OUTPUT = OUTPUT_DIR / "tomato_pick_place_batched_training.json"
HOME_CONFIG = TOMATO_SCENE_ROOT / "robot" / "home_config.yaml"

OBJECT_TO_PICK = "B_tomato"
PLANNER_EE_LINK = "ee_link"
MODE_KEYFRAMES = ("q_pregrasp", "q_grasp_open", "q_lift", "qfinal")

JOINT_LOWER = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
JOINT_UPPER = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("*"),
    TimeElapsedColumn(),
    TextColumn("*"),
    TimeRemainingColumn(),
)


class ShelfTarget(NamedTuple):
    place_tcp_pose: pin.SE3
    place_object_pose: pin.SE3
    local_xy_m: tuple[float, float]
    yaw_rad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate batched multimodal tomato pick-place training trajectories."
    )
    parser.add_argument("--n-scenes", type=int, default=10)
    parser.add_argument("--places-per-scene", type=int, default=4)
    parser.add_argument("--modalities-per-place", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-mm", type=float, default=60.0)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)

    parser.add_argument("--trajectory-length", type=int, default=100)
    parser.add_argument("--min-mode-rms-rad", type=float, default=0.10)
    parser.add_argument("--max-candidates-per-target", type=int, default=64)
    parser.add_argument(
        "--start-noise-rad",
        type=float,
        default=0.35,
        help="Uniform perturbation around home for starts. Set <=0 for full joint-range random.",
    )

    parser.add_argument("--place-local-x-min-m", type=float, default=-0.12)
    parser.add_argument("--place-local-x-max-m", type=float, default=0.12)
    parser.add_argument("--place-local-y-min-m", type=float, default=0.04)
    parser.add_argument("--place-local-y-max-m", type=float, default=0.115)
    parser.add_argument("--shelf-margin-m", type=float, default=0.06)
    parser.add_argument("--place-clearance-m", type=float, default=0.0)

    parser.add_argument("--pregrasp-offset-m", type=float, default=0.10)
    parser.add_argument("--lift-m", type=float, default=0.10)
    parser.add_argument("--open-gripper-width-m", type=float, default=0.08)
    parser.add_argument("--gripper-squeeze-m", type=float, default=0.005)
    parser.add_argument("--close-steps", type=int, default=8)
    parser.add_argument("--open-steps", type=int, default=8)

    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--num-ik-seeds", type=int, default=128)
    parser.add_argument("--num-trajopt-seeds", type=int, default=8)
    parser.add_argument("--num-graph-seeds", type=int, default=8)
    parser.add_argument("--trajopt-tsteps", type=int, default=32)
    parser.add_argument("--interpolation-dt", type=float, default=0.02)
    parser.add_argument("--collision-activation-distance", type=float, default=0.01)
    parser.add_argument("--attached-sphere-radius", type=float, default=0.003)
    parser.add_argument("--disable-graph", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_scenes < 1:
        raise ValueError("--n-scenes must be at least 1")
    if args.places_per_scene < 1:
        raise ValueError("--places-per-scene must be at least 1")
    if args.modalities_per_place < 1:
        raise ValueError("--modalities-per-place must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.min_mode_rms_rad < 0:
        raise ValueError("--min-mode-rms-rad must be non-negative")
    if args.max_candidates_per_target < args.batch_size:
        raise ValueError("--max-candidates-per-target should be >= --batch-size")


def load_home_configuration() -> np.ndarray:
    with open(HOME_CONFIG) as f:
        data = yaml.safe_load(f)
    return np.asarray(data["q_home_rad"], dtype=np.float32)


def random_start(q_home: np.ndarray, rng: np.random.Generator, noise_rad: float) -> np.ndarray:
    margin = 0.05
    if noise_rad <= 0:
        return rng.uniform(JOINT_LOWER + margin, JOINT_UPPER - margin).astype(np.float32)
    q = q_home + rng.uniform(-noise_rad, noise_rad, size=q_home.shape)
    return np.clip(q, JOINT_LOWER + margin, JOINT_UPPER - margin).astype(np.float32)


def scene_to_world(scene: Scene, *, include_tomato: bool) -> dict[str, dict[str, Any]]:
    world: dict[str, dict[str, Any]] = {"cuboid": {}, "cylinder": {}}
    for name, primitive in scene.primitives.items():
        if name == OBJECT_TO_PICK and not include_tomato:
            continue

        qx, qy, qz, qw = primitive.pose_quat_xyzw
        pose_wxyz = [*primitive.pose_xyz_m, qw, qx, qy, qz]
        if primitive.type == "box":
            world["cuboid"][name] = {"dims": list(primitive.dims_m), "pose": pose_wxyz}
        elif primitive.type == "cylinder":
            world["cylinder"][name] = {
                "radius": primitive.dims_m[0],
                "height": primitive.dims_m[1],
                "pose": pose_wxyz,
            }
        else:
            raise ValueError(f"unsupported primitive type {primitive.type!r}")

    return {kind: objects for kind, objects in world.items() if objects}


def create_motion_gen(world: dict[str, dict[str, Any]], args: argparse.Namespace) -> MotionGen:
    tensor_args = get_device_args()
    cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        world,
        tensor_args,
        interpolation_dt=args.interpolation_dt,
        trajopt_tsteps=args.trajopt_tsteps,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        num_graph_seeds=args.num_graph_seeds,
        collision_activation_distance=args.collision_activation_distance,
        ee_link_name=PLANNER_EE_LINK,
    )
    return MotionGen(cfg)


def batch_planner_args(args: argparse.Namespace) -> argparse.Namespace:
    batch_args = copy.copy(args)
    batch_args.num_trajopt_seeds = 1
    batch_args.num_graph_seeds = 1
    batch_args.max_attempts = 1
    return batch_args


def position_only_config(args: argparse.Namespace) -> MotionGenPlanConfig:
    tensor_args = get_device_args()
    pose_metric = PoseCostMetric(
        reach_partial_pose=True,
        reach_vec_weight=tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    return MotionGenPlanConfig(
        max_attempts=args.max_attempts,
        timeout=args.timeout,
        pose_cost_metric=pose_metric,
        check_start_validity=False,
        enable_graph=not args.disable_graph,
    )


def joint_state(q: np.ndarray) -> JointState:
    tensor_args = get_device_args()
    q_tensor = torch.as_tensor(q, device=tensor_args.device, dtype=torch.float32)
    if q_tensor.ndim == 1:
        q_tensor = q_tensor.view(1, -1)
    return JointState.from_position(q_tensor)


def xyzquat_to_se3(xyzquat_xyzw: list[float] | tuple[float, ...] | np.ndarray) -> pin.SE3:
    return pin.XYZQUATToSE3(np.asarray(xyzquat_xyzw, dtype=np.float64))


def quat_xyzw_to_rotation(q_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    qx, qy, qz, qw = q_xyzw
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rotation_to_quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quat = pin.Quaternion(rotation)
    quat.normalize()
    return float(quat.x), float(quat.y), float(quat.z), float(quat.w)


def top_down_quat_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    yaw = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    top_down = np.diag([1.0, -1.0, -1.0])
    return rotation_to_quat_xyzw(yaw @ top_down)


def batched_pose(se3: pin.SE3, batch_size: int) -> Pose:
    tensor_args = get_device_args()
    xyzquat = pin.SE3ToXYZQUAT(se3)
    pose_wxyz = [
        float(xyzquat[0]),
        float(xyzquat[1]),
        float(xyzquat[2]),
        float(xyzquat[6]),
        float(xyzquat[3]),
        float(xyzquat[4]),
        float(xyzquat[5]),
    ]
    return Pose.from_batch_list([pose_wxyz for _ in range(batch_size)], tensor_args=tensor_args)


def curobo_pose_to_se3(pose: Pose) -> pin.SE3:
    xyz = pose.position[0].detach().cpu().numpy()
    quat_wxyz = pose.quaternion[0].detach().cpu().numpy()
    return xyzquat_to_se3([*xyz, quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def ee_pose(motion_gen: MotionGen, state: JointState) -> pin.SE3:
    return curobo_pose_to_se3(motion_gen.compute_kinematics(state).ee_pose)


def sample_shelf_target(scene: Scene, rng: np.random.Generator, args: argparse.Namespace) -> ShelfTarget:
    shelf = scene.primitives["shelf_slab_upper"]
    tomato = scene.primitives[OBJECT_TO_PICK]
    rotation = quat_xyzw_to_rotation(shelf.pose_quat_xyzw)
    shelf_center = np.asarray(shelf.pose_xyz_m, dtype=np.float64)
    top_center = shelf_center + rotation[:, 2] * (shelf.dims_m[2] / 2.0)

    margin = max(args.shelf_margin_m, tomato.dims_m[0] + 0.01)
    half_x = max(0.0, shelf.dims_m[0] / 2.0 - margin)
    half_y = max(0.0, shelf.dims_m[1] / 2.0 - margin)
    x_min = max(-half_x, args.place_local_x_min_m)
    x_max = min(half_x, args.place_local_x_max_m)
    y_min = max(-half_y, args.place_local_y_min_m)
    y_max = min(half_y, args.place_local_y_max_m)
    if x_min > x_max or y_min > y_max:
        raise ValueError("random shelf place bounds are outside the upper shelf")

    local_x = float(rng.uniform(x_min, x_max))
    local_y = float(rng.uniform(y_min, y_max))
    object_center = (
        top_center
        + rotation[:, 0] * local_x
        + rotation[:, 1] * local_y
        + rotation[:, 2] * (args.place_clearance_m + tomato.dims_m[1] / 2.0)
    )
    yaw_rad = float(rng.uniform(-math.pi, math.pi))
    quat = top_down_quat_xyzw(yaw_rad)
    return ShelfTarget(
        place_tcp_pose=xyzquat_to_se3([*object_center, *quat]),
        place_object_pose=pin.SE3(np.eye(3), object_center),
        local_xy_m=(local_x, local_y),
        yaw_rad=yaw_rad,
    )


def grasp_poses(scene: Scene, args: argparse.Namespace) -> tuple[pin.SE3, pin.SE3, pin.SE3]:
    tomato = scene.primitives[OBJECT_TO_PICK]
    grasp = xyzquat_to_se3([*tomato.pose_xyz_m, *top_down_quat_xyzw(0.0)])
    xyzquat = pin.SE3ToXYZQUAT(grasp)
    pregrasp = xyzquat_to_se3(
        [grasp.translation[0], grasp.translation[1], grasp.translation[2] + args.pregrasp_offset_m, *xyzquat[3:]]
    )
    lift = xyzquat_to_se3(
        [grasp.translation[0], grasp.translation[1], grasp.translation[2] + args.lift_m, *xyzquat[3:]]
    )
    return grasp, pregrasp, lift


def tomato_at_scene_pose(scene: Scene) -> Cylinder:
    tomato = scene.primitives[OBJECT_TO_PICK]
    qx, qy, qz, qw = tomato.pose_quat_xyzw
    return Cylinder(
        name=f"{OBJECT_TO_PICK}_attached",
        radius=tomato.dims_m[0],
        height=tomato.dims_m[1],
        pose=[*tomato.pose_xyz_m, qw, qx, qy, qz],
    )


def tomato_at_lift_pose(
    scene: Scene,
    motion_gen: MotionGen,
    grasp_state: JointState,
    lift_state: JointState,
) -> Cylinder:
    tomato = scene.primitives[OBJECT_TO_PICK]
    world_t_grasp_ee = ee_pose(motion_gen, grasp_state)
    world_t_lift_ee = ee_pose(motion_gen, lift_state)
    world_t_tomato = xyzquat_to_se3([*tomato.pose_xyz_m, *tomato.pose_quat_xyzw])
    lift_t_tomato = world_t_lift_ee * (world_t_grasp_ee.inverse() * world_t_tomato)
    xyzquat = pin.SE3ToXYZQUAT(lift_t_tomato)
    return Cylinder(
        name=f"{OBJECT_TO_PICK}_attached",
        radius=tomato.dims_m[0],
        height=tomato.dims_m[1],
        pose=[
            float(xyzquat[0]),
            float(xyzquat[1]),
            float(xyzquat[2]),
            float(xyzquat[6]),
            float(xyzquat[3]),
            float(xyzquat[4]),
            float(xyzquat[5]),
        ],
    )


def phase_lengths(total: int) -> tuple[int, int, int, int]:
    pregrasp = max(2, int(round(total * 0.25)))
    grasp = max(2, int(round(total * 0.20)))
    lift = max(2, int(round(total * 0.15)))
    place = max(2, total - pregrasp - grasp - lift + 3)
    return pregrasp, grasp, lift, place


def numpy_traj(path: JointState) -> np.ndarray:
    arr = path.position.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def result_paths_or_fallback(
    result: Any,
    batch_count: int,
    starts: np.ndarray,
    phase_name: str,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    try:
        paths = result.get_paths()
        if len(paths) == batch_count:
            return [numpy_traj(path) for path in paths]
    except Exception as exc:
        if args.verbose:
            print(f"{phase_name} batch get_paths failed: {exc}")

    if batch_count == 1 and result.interpolated_plan is not None:
        try:
            return [numpy_traj(result.get_interpolated_plan())]
        except Exception as exc:
            if args.verbose:
                print(f"{phase_name} single-lane interpolated path failed: {exc}")

    return [np.asarray([starts[i]], dtype=np.float32) for i in range(batch_count)]


def closed_gripper_width(scene: Scene, args: argparse.Namespace) -> float:
    tomato = scene.primitives[OBJECT_TO_PICK]
    return max(0.0, 2.0 * tomato.dims_m[0] - args.gripper_squeeze_m)


def append_gripper(arm_q: np.ndarray, width_m: float) -> np.ndarray:
    fingers = np.full((arm_q.shape[0], 2), width_m / 2.0, dtype=arm_q.dtype)
    return np.hstack([arm_q, fingers])


def add_gripper_motion(
    arm_q: np.ndarray,
    grasp_index: int,
    open_width: float,
    closed_width: float,
    close_steps: int,
    open_steps: int,
) -> tuple[np.ndarray, list[float], dict[str, int]]:
    before_close = append_gripper(arm_q[: grasp_index + 1], open_width)
    close_rows = np.vstack(
        [
            np.concatenate([arm_q[grasp_index], [width / 2.0, width / 2.0]])
            for width in np.linspace(open_width, closed_width, close_steps + 1)[1:]
        ]
    )
    after_close = append_gripper(arm_q[grasp_index + 1 :], closed_width)
    open_rows = np.vstack(
        [
            np.concatenate([arm_q[-1], [width / 2.0, width / 2.0]])
            for width in np.linspace(closed_width, open_width, open_steps + 1)[1:]
        ]
    )
    full = np.vstack([before_close, close_rows, after_close, open_rows])
    bounds = {
        "close_start_index": grasp_index + 1,
        "close_end_index": grasp_index + close_steps,
        "open_start_index": full.shape[0] - open_steps,
        "open_end_index": full.shape[0] - 1,
    }
    return full, (full[:, -2] + full[:, -1]).astype(float).tolist(), bounds


def build_result(
    scene: Scene,
    args: argparse.Namespace,
    poses: tuple[pin.SE3, pin.SE3, pin.SE3, pin.SE3, pin.SE3],
    phase_trajs: dict[str, np.ndarray],
) -> dict[str, Any]:
    grasp_pose, pregrasp_pose, lift_pose, place_tcp_pose, place_object_pose = poses
    pre_len, grasp_len, lift_len, place_len = phase_lengths(args.trajectory_length)
    pre = resample_trajectory(phase_trajs["pregrasp"], T=pre_len)
    grasp = resample_trajectory(phase_trajs["grasp"], T=grasp_len)
    lift = resample_trajectory(phase_trajs["lift"], T=lift_len)
    place = resample_trajectory(phase_trajs["place"], T=place_len)
    arm = np.vstack([pre, grasp[1:], lift[1:], place[1:]])

    pregrasp_index = pre_len - 1
    grasp_index = pre_len + grasp_len - 2
    lift_index = pre_len + grasp_len + lift_len - 3
    closed_width = closed_gripper_width(scene, args)
    trajectory, gripper_width, grip_bounds = add_gripper_motion(
        arm,
        grasp_index,
        args.open_gripper_width_m,
        closed_width,
        args.close_steps,
        args.open_steps,
    )
    close_offset = max(1, args.close_steps)

    return {
        "target": {
            "object": OBJECT_TO_PICK,
            "pregrasp_tcp_xyzquat_xyzw": pin.SE3ToXYZQUAT(pregrasp_pose).tolist(),
            "grasp_tcp_xyzquat_xyzw": pin.SE3ToXYZQUAT(grasp_pose).tolist(),
            "lift_tcp_xyzquat_xyzw": pin.SE3ToXYZQUAT(lift_pose).tolist(),
            "place_tcp_xyzquat_xyzw": pin.SE3ToXYZQUAT(place_tcp_pose).tolist(),
            "place_object_xyzquat_xyzw": pin.SE3ToXYZQUAT(place_object_pose).tolist(),
            "place_surface": "shelf_slab_upper",
            "place_clearance_m": args.place_clearance_m,
        },
        "q0": trajectory[0].astype(float).tolist(),
        "q_pregrasp": np.concatenate([phase_trajs["pregrasp"][-1], [args.open_gripper_width_m / 2] * 2]).astype(float).tolist(),
        "q_grasp_open": np.concatenate([phase_trajs["grasp"][-1], [args.open_gripper_width_m / 2] * 2]).astype(float).tolist(),
        "q_grasp_closed": np.concatenate([phase_trajs["grasp"][-1], [closed_width / 2] * 2]).astype(float).tolist(),
        "q_lift": np.concatenate([phase_trajs["lift"][-1], [closed_width / 2] * 2]).astype(float).tolist(),
        "qfinal": trajectory[-1].astype(float).tolist(),
        "arm_trajectory": arm.astype(float).tolist(),
        "trajectory": trajectory.astype(float).tolist(),
        "gripper_width": gripper_width,
        "phase_boundaries": {
            "pregrasp_index": pregrasp_index,
            "grasp_index": grasp_index,
            **grip_bounds,
            "lift_index": lift_index + close_offset,
            "place_start_index": lift_index + close_offset,
            "place_index": trajectory.shape[0] - 1 - args.open_steps,
        },
    }


def plan_batch_phase(
    name: str,
    motion_gen: MotionGen,
    starts: np.ndarray,
    goal: pin.SE3,
    cfg: MotionGenPlanConfig,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[np.ndarray]]:
    batch_count = starts.shape[0]
    success = np.zeros(batch_count, dtype=bool)
    paths: list[np.ndarray | None] = [None] * batch_count
    last_status: Any = "not_run"

    for _ in range(max(1, args.max_attempts)):
        if bool(np.all(success)):
            break
        result = motion_gen.plan_batch(joint_state(starts), batched_pose(goal, batch_count), cfg)
        last_status = getattr(result, "status", "unknown")
        attempt_success = (
            result.success.detach().view(-1).cpu().numpy().astype(bool)
            if result.success is not None
            else np.zeros(batch_count, dtype=bool)
        )
        if attempt_success.shape[0] != batch_count:
            raise RuntimeError(f"{name} returned {attempt_success.shape[0]} lanes for {batch_count}")

        attempt_paths = result_paths_or_fallback(result, batch_count, starts, name, args)

        for i in range(batch_count):
            paths[i] = attempt_paths[i]
            success[i] = success[i] or attempt_success[i]

    if args.verbose and not bool(np.all(success)):
        print(f"{name} batch success: {int(success.sum())}/{batch_count}; last_status={last_status}")
    return success, [p if p is not None else np.asarray([starts[i]]) for i, p in enumerate(paths)]


def keep_success(
    phase_name: str,
    lanes: list[int],
    success: np.ndarray,
    paths: list[np.ndarray],
    phase_trajs: dict[int, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> list[int]:
    kept: list[int] = []
    for local_i, lane in enumerate(lanes):
        if success[local_i]:
            phase_trajs[lane][phase_name] = paths[local_i]
            kept.append(lane)
        elif args.verbose:
            print(f"batch lane {lane} failed during {phase_name}")
    return kept


def phase_end_q(phase_trajs: dict[int, dict[str, np.ndarray]], lanes: list[int], phase: str) -> np.ndarray:
    return np.stack([phase_trajs[lane][phase][-1] for lane in lanes]).astype(np.float32)


def plan_target_batch(
    scene: Scene,
    target: ShelfTarget,
    q_starts: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, Any]]]:
    batch_args = batch_planner_args(args)
    pick_world = scene_to_world(scene, include_tomato=True)
    carry_world = scene_to_world(scene, include_tomato=False)
    grasp_pose, pregrasp_pose, lift_pose = grasp_poses(scene, args)
    phase_trajs: dict[int, dict[str, np.ndarray]] = {i: {} for i in range(q_starts.shape[0])}
    lanes = list(phase_trajs)

    pregrasp_gen = create_motion_gen(pick_world, batch_args)
    success, paths = plan_batch_phase(
        "pregrasp",
        pregrasp_gen,
        q_starts,
        pregrasp_pose,
        position_only_config(batch_args),
        args,
    )
    lanes = keep_success("pregrasp", lanes, success, paths, phase_trajs, args)
    if not lanes:
        return []

    grasp_gen = create_motion_gen(carry_world, batch_args)
    success, paths = plan_batch_phase(
        "grasp",
        grasp_gen,
        phase_end_q(phase_trajs, lanes, "pregrasp"),
        grasp_pose,
        position_only_config(batch_args),
        args,
    )
    lanes = keep_success("grasp", lanes, success, paths, phase_trajs, args)
    if not lanes:
        return []

    lift_gen = create_motion_gen(carry_world, batch_args)
    lift_ref_lane = lanes[0]
    attached = lift_gen.attach_external_objects_to_robot(
        joint_state(phase_trajs[lift_ref_lane]["grasp"][-1]),
        [tomato_at_scene_pose(scene)],
        surface_sphere_radius=args.attached_sphere_radius,
    )
    if not attached:
        raise RuntimeError("failed to attach tomato before lift")
    success, paths = plan_batch_phase(
        "lift",
        lift_gen,
        phase_end_q(phase_trajs, lanes, "grasp"),
        lift_pose,
        position_only_config(batch_args),
        args,
    )
    lanes = keep_success("lift", lanes, success, paths, phase_trajs, args)
    if not lanes:
        return []

    place_gen = create_motion_gen(carry_world, batch_args)
    place_ref_lane = lanes[0]
    attached = place_gen.attach_external_objects_to_robot(
        joint_state(phase_trajs[place_ref_lane]["lift"][-1]),
        [
            tomato_at_lift_pose(
                scene,
                lift_gen,
                joint_state(phase_trajs[place_ref_lane]["grasp"][-1]),
                joint_state(phase_trajs[place_ref_lane]["lift"][-1]),
            )
        ],
        surface_sphere_radius=args.attached_sphere_radius,
    )
    if not attached:
        raise RuntimeError("failed to attach tomato before place")
    success, paths = plan_batch_phase(
        "place",
        place_gen,
        phase_end_q(phase_trajs, lanes, "lift"),
        target.place_tcp_pose,
        position_only_config(batch_args),
        args,
    )
    lanes = keep_success("place", lanes, success, paths, phase_trajs, args)

    results: list[tuple[int, dict[str, Any]]] = []
    poses = (grasp_pose, pregrasp_pose, lift_pose, target.place_tcp_pose, target.place_object_pose)
    for lane in lanes:
        result = build_result(scene, args, poses, phase_trajs[lane])
        result["batch_info"] = {
            "batched": True,
            "requested_batch_size": int(q_starts.shape[0]),
            "lane_index": lane,
            "effective_num_trajopt_seeds": batch_args.num_trajopt_seeds,
            "effective_num_graph_seeds": batch_args.num_graph_seeds,
            "graph_enabled": not batch_args.disable_graph,
            "lift_attachment_reference_lane": lift_ref_lane,
            "place_attachment_reference_lane": place_ref_lane,
        }
        results.append((lane, result))
    return results


def mode_signature(result: dict[str, Any]) -> np.ndarray:
    return np.concatenate([np.asarray(result[name], dtype=np.float64)[:7] for name in MODE_KEYFRAMES])


def accept_mode(
    candidate: dict[str, Any],
    accepted: list[dict[str, Any]],
    min_rms_rad: float,
) -> tuple[bool, float | None]:
    if not accepted:
        return True, None
    distances = [
        float(np.sqrt(np.mean(np.square(mode_signature(candidate) - mode_signature(prev)))))
        for prev in accepted
    ]
    nearest = min(distances)
    return nearest >= min_rms_rad, nearest


def add_mode_info(result: dict[str, Any], min_rms_rad: float, nearest: float | None) -> None:
    result["mode_info"] = {
        "keyframes": list(MODE_KEYFRAMES),
        "signature": mode_signature(result).astype(float).tolist(),
        "distance": "rms_arm_joint_distance_rad",
        "min_required_rms_rad": min_rms_rad,
        "nearest_previous_mode_rms_rad": nearest,
    }


def scene_metadata(scene: Scene) -> dict[str, Any]:
    return {
        "params": asdict(scene.params),
        "primitives": {
            name: {
                "type": primitive.type,
                "dims_m": list(primitive.dims_m),
                "pose_xyz_m": list(primitive.pose_xyz_m),
                "pose_quat_xyzw": list(primitive.pose_quat_xyzw),
            }
            for name, primitive in scene.primitives.items()
        },
    }


def dataset_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "description": "Batched multimodal tomato pick-place trajectories for training.",
        "joint_order": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        "arm_joint_order": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        "scene_package_root": TOMATO_SCENE_ROOT.as_posix(),
        "scene_config": {
            "seed": args.seed,
            "k_mm": args.k_mm,
            "composite_origin_in_base_m": tomato_constants.COMPOSITE_ORIGIN_IN_BASE_M.tolist(),
        },
        "planner": {
            "robot_config": "franka.yml",
            "planner_ee_link": PLANNER_EE_LINK,
            "collision_checker_type": "PRIMITIVE",
            "batch_size": args.batch_size,
            "n_scenes": args.n_scenes,
            "places_per_scene": args.places_per_scene,
            "modalities_per_place": args.modalities_per_place,
            "trajectory_length": args.trajectory_length,
            "min_mode_rms_rad": args.min_mode_rms_rad,
            "mode_keyframes": list(MODE_KEYFRAMES),
            "random_start_noise_rad": args.start_noise_rad,
            "random_place_local_x_bounds_m": [args.place_local_x_min_m, args.place_local_x_max_m],
            "random_place_local_y_bounds_m": [args.place_local_y_min_m, args.place_local_y_max_m],
            "place_clearance_m": args.place_clearance_m,
        },
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    print("Using device:", get_device_args().device)
    print(
        f"Generating {args.n_scenes} scenes x {args.places_per_scene} place poses x "
        f"{args.modalities_per_place} modes with batch size {args.batch_size}"
    )

    q_home = load_home_configuration()
    rng = np.random.default_rng(args.seed)
    sampler = SceneSampler(SceneConfig(seed=args.seed, k_mm=args.k_mm))
    total = args.n_scenes * args.places_per_scene * args.modalities_per_place
    dataset: dict[str, Any] = {"metadata": dataset_metadata(args), "trajectories": []}

    with progress:
        task = progress.add_task("  * Batched training trajectories", total=total)
        for scene_index in range(args.n_scenes):
            scene = sampler.sample()
            scene_info = scene_metadata(scene)

            for target_index in range(args.places_per_scene):
                target = sample_shelf_target(scene, rng, args)
                accepted: list[dict[str, Any]] = []
                candidate_count = 0
                batch_attempt = 0

                while (
                    len(accepted) < args.modalities_per_place
                    and candidate_count < args.max_candidates_per_target
                ):
                    batch_count = min(args.batch_size, args.max_candidates_per_target - candidate_count)
                    starts = np.stack(
                        [random_start(q_home, rng, args.start_noise_rad) for _ in range(batch_count)]
                    )
                    candidate_count += batch_count

                    try:
                        batch_results = plan_target_batch(scene, target, starts, args)
                    except Exception as exc:
                        batch_results = []
                        if args.verbose:
                            print(
                                f"scene {scene_index}, target {target_index}, "
                                f"batch {batch_attempt} failed: {exc}"
                            )

                    for lane, result in batch_results:
                        if len(accepted) >= args.modalities_per_place:
                            break
                        accepted_mode, nearest = accept_mode(
                            result,
                            accepted,
                            args.min_mode_rms_rad,
                        )
                        add_mode_info(result, args.min_mode_rms_rad, nearest)
                        if not accepted_mode:
                            if args.verbose:
                                print(
                                    f"scene {scene_index}, target {target_index}, lane {lane} "
                                    f"rejected: nearest mode RMS {nearest:.4f} rad"
                                )
                            continue

                        result["scene_index"] = scene_index
                        result["target_index"] = target_index
                        result["modality_index"] = len(accepted)
                        result["trajectory_index"] = len(dataset["trajectories"])
                        result["scene"] = copy.deepcopy(scene_info)
                        result["target"]["sampled_place_local_xy_m"] = list(target.local_xy_m)
                        result["target"]["sampled_place_yaw_rad"] = target.yaw_rad
                        result["batch_info"]["batch_attempt_index"] = batch_attempt
                        result["batch_info"]["candidate_count_before_batch"] = (
                            candidate_count - batch_count
                        )
                        result["batch_info"]["lane_index"] = lane

                        accepted.append(result)
                        dataset["trajectories"].append(result)
                        progress.update(task, advance=1)

                    batch_attempt += 1

                missing = args.modalities_per_place - len(accepted)
                if missing > 0:
                    if args.verbose:
                        print(
                            f"scene {scene_index}, target {target_index}: accepted "
                            f"{len(accepted)}/{args.modalities_per_place} modes from "
                            f"{candidate_count} candidates"
                        )
                    progress.update(task, advance=missing)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset['trajectories'])} trajectories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
