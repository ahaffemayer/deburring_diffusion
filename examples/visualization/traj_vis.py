import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from deburring_diffusion.traj_gen_utils import (
    from_trajectory_to_ee_poses,
    load_results
)
from deburring_diffusion.panda_env_loader import load_reduced_panda


if __name__ == "__main__":
    import pathlib
    results_path = pathlib.Path(__file__).parent.parent.parent / "results" / "traj_generator" / "single_configuration_single_single_pose.json"
    results = load_results(results_path)

    rmodel, _, _ = load_reduced_panda()
    
        
    # ---------------------------
    # Plot EE translations
    # ---------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for traj_id, result in enumerate(results):
        trajectory = [np.array(q) for q in result["trajectory"]]

        ee_poses = from_trajectory_to_ee_poses(rmodel, trajectory)
        ee_xyz = np.array([pose.translation for pose in ee_poses])

        ax.plot(
            ee_xyz[:, 0],
            ee_xyz[:, 1],
            ee_xyz[:, 2],
            label=f"traj {traj_id}",
        )

    # Optional, plot target
    target = np.array(results[0]["target"])
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

    plt.show()
