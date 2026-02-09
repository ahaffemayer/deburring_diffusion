import json
import pathlib

import numpy as np
import pinocchio as pin
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from deburring_diffusion.robot.curobo_utils import (
    create_motion_gen_curobo,
    create_motion_gen_plan_config,
    get_device_args,
    plan_with_curobo,
    resample_trajectory,
)
from deburring_diffusion.robot.panda_env_loader import (
    load_fer,
)
from deburring_diffusion.robot.traj_gen_utils import store_results
from deburring_diffusion.robot.visualization_utils import setup_scene

# ---------------------------
# Progress Bar Configuration
# ---------------------------
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("\u2022"),
    TimeElapsedColumn(),
    TextColumn("\u2022"),
    TimeRemainingColumn(),
)

# Configuration
MODEL_PATH = pathlib.Path(__file__).parent.parent / "models"
OBJ_FILE = MODEL_PATH / "pylone.obj"
OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "traj_generator"

# Scene configuration
PYLONE_POSE = pin.XYZQUATToSE3([0.45, -0.116, 0.739, 0.0, 0.0, 0.0, 1.0])
TARGET_POSITION = np.array([0.35, 0.4, 0.7])
TARGET_SE3 = pin.SE3(np.eye(3), TARGET_POSITION)

# Generation parameters
VERBOSE = False  # For the debug
N_TRAJECTORIES_PER_TARGET = 50
N_TARGET = 1000
TRAJECTORY_LENGTH = 50  # Resampled length
VISUALIZE_TRAJECTORIES = False  # Set to True for interactive visualization
MAX_TRIES = 15
x_lim = np.array([-0.7, 0.7])
y_lim = np.array([-0.7, 0.7])
z_lim = np.array([-0.3, 1])


def generate_reachable_target(
    x_lim: np.ndarray, y_lim: np.ndarray, z_lim: np.ndarray
) -> pin.SE3:
    """
    x_lim, y_lim, z_lim: arrays like [min, max]
    """

    t = np.array(
        [
            np.random.uniform(x_lim[0], x_lim[1]),
            np.random.uniform(y_lim[0], y_lim[1]),
            np.random.uniform(z_lim[0], z_lim[1]),
        ]
    )

    R = pin.exp3(np.random.randn(3))

    return pin.SE3(R, t)


if __name__ == "__main__":
    tensor_args = get_device_args()
    print("Using device:", tensor_args.device)

    rmodel, cmodel, vmodel = load_fer()
    rdata = rmodel.createData()
    vdata = vmodel.createData()

    motion_gen = create_motion_gen_curobo(
        pylone_pose=PYLONE_POSE, obj_file=OBJ_FILE, use_custom_panda=True
    )
    plan_cfg = create_motion_gen_plan_config()

    results = []
    with progress:
        target_task = progress.add_task("  • Targets processed", total=N_TARGET)
        traj_task = progress.add_task(
            "  • Trajectories for current target",
            total=N_TRAJECTORIES_PER_TARGET,
        )

        for j in range(N_TARGET):
            TARGET_SE3 = generate_reachable_target(x_lim, y_lim, z_lim)
            progress.reset(traj_task)
            progress.update(
                traj_task,
                description=f"  • Trajectories for target {j + 1}/{N_TARGET}",
            )

            consecutive_failures = 0
            success_count = 0

            for i in range(N_TRAJECTORIES_PER_TARGET):
                try:
                    q_start = pin.randomConfiguration(rmodel)
                    traj = plan_with_curobo(motion_gen, q_start, TARGET_SE3, plan_cfg)
                    resampled_traj = resample_trajectory(
                        traj.position.cpu().numpy(), T=TRAJECTORY_LENGTH
                    )
                    xs = [resampled_traj[t] for t in range(resampled_traj.shape[0])]

                    result = store_results(xs, pin.SE3ToXYZQUAT(TARGET_SE3), rmodel)
                    results.append(result)

                    if VISUALIZE_TRAJECTORIES:
                        scene, robot = setup_scene(
                            rmodel=rmodel,
                            rdata=rdata,
                            vmodel=vmodel,
                            vdata=vdata,
                            obj_file=OBJ_FILE,
                            target_se3=TARGET_SE3,
                            pylone_pose=PYLONE_POSE,
                        )
                        for x in xs:
                            robot[:] = x[: rmodel.nq]
                            input()

                    success_count += 1
                    consecutive_failures = 0
                    progress.update(traj_task, advance=1)

                except Exception as e:
                    if VERBOSE:
                        print(f"\n  Warning: Failed to generate trajectory {i}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_TRIES:
                        break  # target unreachable

            progress.update(target_task, advance=1)

    # Save results to a JSON file

    with open(OUTPUT_DIR / "multiple_configuration_multiple_poses.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Saved {len(results)} trajectories to {OUTPUT_DIR / 'multiple_configuration_multiple_poses.json'}"
    )
