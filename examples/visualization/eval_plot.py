import torch
import numpy as np
import json
import pathlib
import pinocchio as pin
from robomeshcat import Scene, Object, Robot

from deburring_diffusion.robot.panda_env_loader import load_reduced_panda
from deburring_diffusion.robot.traj_gen_utils import from_trajectory_to_ee_poses, store_results
from deburring_diffusion.diffusion.model import Model 
from deburring_diffusion.robot.traj_gen_utils import from_trajectory_to_ee_poses

if __name__ == "__main__":
    # 1. Setup Environment 
    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    vdata = vmodel.createData()

    q_start = np.array([0.0, -0.4, 0.0, -0.2, 0.0, 1.57, 0.79])
    target_se3 = pin.SE3(np.eye(3), np.array([0.35, 0.4, 0.7]))
    target_xyzquat = pin.SE3ToXYZQUAT(target_se3) # [x, y, z, qx, qy, qz, qw]
    pylone_pose = pin.XYZQUATToSE3([0.45, -0.116, 0.739, 0.0, 0.0, 0.0, 1.0])

    # Setting up the scene with the robot and the pylone object
    obj_path = pathlib.Path(__file__).parent.parent / "models"
    obj_file = obj_path / "pylone.obj"

    # 2. Load the Trained Diffusion Model
    model = Model.load_from_checkpoint("/workspaces/deburring_diffusion/results/diffusion/lightning_logs/version_0/checkpoints/epoch=99-step=700.ckpt")
    model.eval()
    model.cuda()

    # 3. Prepare Conditioning Tensor
    # We concatenate q_start (7) and target (7) = 14 dims
    # Note: Ensure this matches the 'cond' format used during training!
    cond_dict = {
            "q0": torch.from_numpy(q_start).float().unsqueeze(0).cuda(),      # (1, 7)
            "goal": torch.from_numpy(target_xyzquat).float().unsqueeze(0).cuda() # (1, 7)
        }
    # Shape: (bs=1, n_tokens=1, dim=14)

    # 4. Sample from Diffusion
    n_samples = 1 # How many trajectories to generate for this one goal
    results = []

    print(f"Sampling {n_samples} trajectories from Diffusion...")
    with torch.no_grad():
        sampled_trajs = model.sample(
            cond=cond_dict,
            bs=n_samples,
            seq_length=50, # Matches your cuRobo resampling T=50
            configuration_size=rmodel.nq
        )

    # 5. Process and Store Results
    output_dir = pathlib.Path(__file__).parent.parent / "results" / "diffusion_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    goal.pose = target_se3.homogeneous


    for i in range(n_samples):
        # Convert torch to list of numpy arrays
        traj_np = sampled_trajs[i].cpu().numpy()
        xs = [traj_np[t] for t in range(traj_np.shape[0])]
        
        # Use your existing utility to format the JSON entry
        result = store_results(xs, target_xyzquat, rmodel)
        results.append(result)

    # for x in xs:
    #     q = x[: rmodel.nq]
    #     robot[:] = q
    #     input("Press Enter to continue...")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ee_poses = from_trajectory_to_ee_poses(rmodel, xs)
    ee_xyz = np.array([pose.translation for pose in ee_poses])

    ax.plot(
        ee_xyz[:, 0],
        ee_xyz[:, 1],
        ee_xyz[:, 2],
        label=f"traj",
    )

    # Optional, plot target
    target = np.array(results[0]["target"][:3])
    ax.scatter(
        target[0],
        target[1],
        target[2],
        marker="x",
        s=80,
        label="target",
    )

    # ---------------------------
    # Formatting
    # ---------------------------
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("End-effector trajectories")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    plt.savefig(pathlib.Path(__file__).parent.parent / "results" / "traj_generator" / 'trajectory_output.png')
    print("Plot saved as trajectory_output.png")