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
from robomeshcat import Object, Robot, Scene

from deburring_diffusion.robot.curobo_utils import (
    create_motion_gen_curobo,
    create_motion_gen_plan_config,
    get_device_args,
    plan_with_curobo,
    resample_trajectory,
)
from deburring_diffusion.robot.panda_env_loader import (
    load_reduced_panda,
)
from deburring_diffusion.robot.traj_gen_utils import store_results

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


if __name__ == "__main__":
    tensor_args = get_device_args()
    print("Using device:", tensor_args.device)

    # Problem settings
    pylone_pose = pin.XYZQUATToSE3([0.45, -0.116, 0.739, 0.0, 0.0, 0.0, 1.0])
    target = pin.SE3(np.eye(3), np.array([0.35, 0.0, 0.7]))
    q_start = np.array(
        [
            0.0,
            -0.4,
            0.0,
            -0.2,
            0.0,
            1.57,
            0.79,
        ]
    )

    n_trajectories = 100

    # Output directory
    output_dir = pathlib.Path(__file__).parent.parent / "results" / "traj_generator"

    # Setting up the scene with the robot and the pylone object
    obj_path = pathlib.Path(__file__).parent.parent / "models"
    obj_file = obj_path / "pylone.obj"

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

    motion_gen = create_motion_gen_curobo(pylone_pose=pylone_pose, obj_file=obj_file)
    plan_cfg = create_motion_gen_plan_config()

    robot[:] = q_start

    results = []
    with progress:
        traj_task = progress.add_task(
            "  • Trajectories generated", total=n_trajectories
        )
        for _ in range(n_trajectories):
            traj = plan_with_curobo(motion_gen, q_start, target, plan_cfg)
            resampled_traj = resample_trajectory(traj.position.cpu().numpy(), T=50)
            xs = [resampled_traj[t] for t in range(resampled_traj.shape[0])]

            result = store_results(xs, pin.SE3ToXYZQUAT(target), rmodel)
            results.append(result)
            for x in xs:
                q = x[: rmodel.nq]
                robot[:] = q
                input("Press Enter to continue...")
            progress.update(traj_task, advance=1)

    # Save results to a JSON file

    # with open(output_dir / "single_configuration_single_single_pose.json", "w") as f:
    #     json.dump(results, f, indent=2)

    print(
        f"Saved {len(results)} trajectories to {output_dir / 'trajectories_data_shelf.json'}"
    )
