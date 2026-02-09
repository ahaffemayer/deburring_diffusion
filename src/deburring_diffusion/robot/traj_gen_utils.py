from typing import Dict, List
import numpy as np
import pinocchio as pin

def store_results(xs: List[np.ndarray], target: np.ndarray, rmodel: pin.Model) -> Dict:
    """Converts trajectory data and metadata into a dictionary format for saving."""
    q0 = xs[0][: rmodel.nq]
    qfinal = xs[-1][: rmodel.nq]

    return {
        "target": target.tolist(),
        "q0": q0.tolist(),
        "qfinal": qfinal.tolist(),
        "trajectory": [X[:7].tolist() for X in xs],
    }

def load_results(file_path: str) -> dict:
    """Loads trajectory results from a JSON file."""
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def q_to_se3(
    rmodel: pin.Model, q: np.ndarray, ee_frame: str = "fer_hand_tcp"
) -> pin.SE3:
    """Converts a configuration vector to an SE3 pose of the end-effector."""
    rdata = rmodel.createData()
    pin.framesForwardKinematics(rmodel, rdata, q)
    ee_frame_id = rmodel.getFrameId(ee_frame)
    ee_pose = rdata.oMf[ee_frame_id]
    return ee_pose


def from_trajectory_to_ee_poses(
    rmodel: pin.Model, trajectory: List[np.ndarray], ee_frame: str = "fer_hand_tcp"
) -> List[pin.SE3]:
    """Extracts end-effector poses from a list of configuration vectors."""
    ee_poses = []
    for q in trajectory:
        ee_pose = q_to_se3(rmodel, q, ee_frame)
        ee_poses.append(ee_pose)
    return ee_poses