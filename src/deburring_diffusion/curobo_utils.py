import torch
from curobo.types.base import TensorDeviceType
import hppfcl
import pinocchio as pin


def get_device_args():
    # choose CUDA if available, otherwise CPU
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return TensorDeviceType(device=dev)




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
            qx, qy, qz, qw = rotation # Pinocchio uses (qx, qy, qz, qw) order
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
