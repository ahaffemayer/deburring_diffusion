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
N_TRAJECTORIES = 1000
TRAJECTORY_LENGTH = 50  # Resampled length
VISUALIZE_TRAJECTORIES = False  # Set to True for interactive visualization


if __name__ == "__main__":
    tensor_args = get_device_args()
    print("Using device:", tensor_args.device)

    rmodel, cmodel, vmodel = load_fer()
    rdata = rmodel.createData()
    vdata = vmodel.createData()

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
        robot[:] = pin.neutral(rmodel)

    motion_gen = create_motion_gen_curobo(
        pylone_pose=PYLONE_POSE, obj_file=OBJ_FILE, use_custom_panda=True
    )
    plan_cfg = create_motion_gen_plan_config()

    results = []
    with progress:
        traj_task = progress.add_task(
            "  • Trajectories generated", total=N_TRAJECTORIES
        )
        for i in range(N_TRAJECTORIES):
            try:
                q_start = pin.randomConfiguration(rmodel)
                traj = plan_with_curobo(motion_gen, q_start, TARGET_SE3, plan_cfg)
                resampled_traj = resample_trajectory(traj.position.cpu().numpy(), T=50)
                xs = [resampled_traj[t] for t in range(resampled_traj.shape[0])]

                result = store_results(xs, pin.SE3ToXYZQUAT(TARGET_SE3), rmodel)
                results.append(result)
                if VISUALIZE_TRAJECTORIES:
                    for x in xs:
                        q = x[: rmodel.nq]
                        robot[:] = q
                        input("Press Enter to continue...")
            except Exception as e:
                print(f"\n  Warning: Failed to generate trajectory {i}: {e}")
                continue
            progress.update(traj_task, advance=1)

    # Save results to a JSON file

    with open(OUTPUT_DIR / "multiple_configuration_single_single_pose.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Saved {len(results)} trajectories to {OUTPUT_DIR / 'multiple_configuration_single_single_pose.json'}"
    )
