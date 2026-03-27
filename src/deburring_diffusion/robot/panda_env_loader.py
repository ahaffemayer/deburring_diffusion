import pathlib

import example_robot_data as robex
import numpy as np
import pinocchio as pin

robot_links = [
    # "panda_link0_capsule_0",
    "panda_hand_capsule_0",
    "panda_link7_capsule_0",
    "panda_link7_capsule_1",
    "panda_link6_capsule_0",
    "panda_link5_capsule_0",
    "panda_link5_capsule_1",
    "panda_link4_capsule_0",
    "panda_rightfinger_capsule_0",
    "panda_leftfinger_capsule_0",
]


self_collision_pairs = [
    ("panda_link4_capsule_0", "panda_link6_capsule_0"),
    ("panda_link4_capsule_0", "panda_link7_capsule_0"),
    ("panda_link5_capsule_0", "panda_link7_capsule_0"),
    ("panda_link6_capsule_0", "panda_link7_capsule_0"),
    ("panda_link0_capsule_0", "panda_link7_capsule_0"),
]


def load_fer():
    package_dir = pathlib.Path(__file__).parent.parent.parent.parent
    urdf_file = package_dir / "models" / "fer" / "fer.urdf"
    rmodel, cmodel, vmodel = pin.buildModelsFromUrdf(
        urdf_file, package_dirs=[package_dir]
    )
    return rmodel, cmodel, vmodel


def load_panda():
    panda = robex.load("panda_collision")
    rmodel, cmodel, vmodel = panda.model, panda.collision_model, panda.visual_model
    return rmodel, cmodel, vmodel


def load_reduced_panda():
    rmodel, cmodel, vmodel = load_panda()
    geom_models = [vmodel, cmodel]
    rmodel, geometric_models_reduced = pin.buildReducedModel(
        rmodel,
        list_of_geom_models=geom_models,
        list_of_joints_to_lock=[8, 9],
        reference_configuration=np.array(
            [
                -0.6513877410293797,
                1.3677075286603906,
                -0.17736737718858037,
                -0.3973375018143172,
                -0.11554961778792178,
                1.2408486160482337,
                8.644879755868687e-05,
                0.01,
                0.02,
            ]
        ),
    )

    vmodel, cmodel = geometric_models_reduced[0], geometric_models_reduced[1]
    return rmodel, cmodel, vmodel


def add_self_collision(cmodel: pin.GeometryModel) -> pin.GeometryModel:
    for cp in self_collision_pairs:
        if cmodel.existGeometryName(cp[0]) and cmodel.existGeometryName(cp[1]):
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId(cp[0]),
                    cmodel.getGeometryId(cp[1]),
                )
            )
    return cmodel
