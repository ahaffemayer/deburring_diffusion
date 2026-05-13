"""Replay tomato pick-place trajectories in Meshcat.

Run after generating a dataset with:
    python examples/traj_generator_tomato_pick_place.py

Then visualize one trajectory:
    python examples/visualization/visualize_tomato_pick_place_meshcat.py --index 0

Open the printed Meshcat URL in your browser. In the devcontainer this is
usually:
    http://localhost:7000/static/
"""
from __future__ import annotations

import argparse
import atexit
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import numpy as np
import pinocchio as pin
from pinocchio import visualize

from deburring_diffusion.robot.panda_env_loader import load_panda


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "results" / "traj_generator" / "tomato_pick_place.json"

SCENE_ROOT = "tomato_pick_place"
BOX_COLOR = 0x8A8F98
SHELF_COLOR = 0xD8C6A3
A_COLOR = 0xE7C783
B_COLOR = 0xD84B3E
C_COLOR = 0x3AC6D8
D_COLOR = 0xA9A9A9
E_COLOR = 0xF2D35E
TARGET_COLOR = 0x45C96A
PATH_COLOR = 0x2A6FDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize tomato pick-place JSON trajectories in Meshcat."
    )
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--pause", action="store_true", help="Step with Enter instead of time.")
    parser.add_argument("--path-stride", type=int, default=2)
    parser.add_argument(
        "--zmq-url",
        default=None,
        help=(
            "Connect to an existing Meshcat server, e.g. tcp://127.0.0.1:6000. "
            "By default this script starts one."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the Meshcat web server started by this script.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=7000,
        help="HTTP port for the Meshcat web server started by this script.",
    )
    parser.add_argument("--open", action="store_true", help="Ask Meshcat to open a browser tab.")
    return parser.parse_args()


def load_trajectory(dataset_path: pathlib.Path, index: int) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} does not exist. Generate it with "
            "examples/traj_generator_tomato_pick_place.py first."
        )
    with open(dataset_path) as f:
        dataset = json.load(f)

    trajectories = dataset.get("trajectories", [])
    if not trajectories:
        raise ValueError(f"{dataset_path} contains no trajectories")
    if not 0 <= index < len(trajectories):
        raise IndexError(f"--index {index} out of range for {len(trajectories)} trajectories")
    return trajectories[index]


def pose_matrix_xyzquat_xyzw(
    xyz: list[float] | tuple[float, ...],
    quat_xyzw: list[float] | tuple[float, ...],
) -> np.ndarray:
    return pin.XYZQUATToSE3([*xyz, *quat_xyzw]).homogeneous


def cylinder_mesh_transform() -> np.ndarray:
    # Three.js cylinders are y-axis aligned. Scene cylinders are z-axis aligned.
    return mtf.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])


def color_for_object(name: str) -> int:
    if name.startswith("shelf"):
        return SHELF_COLOR
    if name.startswith("A_"):
        return A_COLOR
    if name.startswith("B_"):
        return B_COLOR
    if name.startswith("C_"):
        return C_COLOR
    if name.startswith("D_"):
        return D_COLOR
    if name.startswith("E_"):
        return E_COLOR
    return BOX_COLOR


def add_primitive(viewer: meshcat.Visualizer, path: str, primitive: dict[str, Any]) -> None:
    dims = primitive["dims_m"]
    pose = pose_matrix_xyzquat_xyzw(
        primitive["pose_xyz_m"],
        primitive["pose_quat_xyzw"],
    )
    material = g.MeshLambertMaterial(
        color=color_for_object(path),
        transparent=True,
        opacity=0.75,
    )

    node = viewer[path]
    if primitive["type"] == "box":
        node.set_object(g.Box(dims), material)
        node.set_transform(pose)
    elif primitive["type"] == "cylinder":
        node.set_object(g.Cylinder(dims[1], dims[0]), material)
        node.set_transform(pose @ cylinder_mesh_transform())
    else:
        raise ValueError(f"unsupported primitive type {primitive['type']!r}")


def add_scene_primitives(viewer: meshcat.Visualizer, trajectory: dict[str, Any]) -> None:
    viewer[SCENE_ROOT].delete()
    primitives = trajectory["scene"]["primitives"]

    for name, primitive in primitives.items():
        if name == "B_tomato":
            continue
        add_primitive(viewer, f"{SCENE_ROOT}/static/{name}", primitive)

    b = primitives["B_tomato"]
    add_primitive(viewer, f"{SCENE_ROOT}/moving/B_tomato", b)

    place_xyz = trajectory["target"]["place_tcp_xyzquat_xyzw"][:3]
    viewer[f"{SCENE_ROOT}/target/place"].set_object(
        g.Sphere(0.018),
        g.MeshLambertMaterial(color=TARGET_COLOR, transparent=True, opacity=0.9),
    )
    viewer[f"{SCENE_ROOT}/target/place"].set_transform(
        mtf.translation_matrix(place_xyz)
    )


def display_configuration(q: np.ndarray, nq: int) -> np.ndarray:
    if q.shape[0] >= nq:
        return q[:nq]
    if nq == 9 and q.shape[0] == 7:
        return np.concatenate([q, [0.04, 0.04]])
    raise ValueError(f"cannot display configuration with shape {q.shape}; robot nq={nq}")


def frame_pose(rmodel: pin.Model, rdata: pin.Data, q: np.ndarray, frame_name: str) -> pin.SE3:
    pin.framesForwardKinematics(rmodel, rdata, display_configuration(q, rmodel.nq))
    return rdata.oMf[rmodel.getFrameId(frame_name)]


def add_ee_path(
    viewer: meshcat.Visualizer,
    rmodel: pin.Model,
    rdata: pin.Data,
    trajectory: np.ndarray,
    stride: int,
) -> None:
    stride = max(1, stride)
    for i, q in enumerate(trajectory[::stride]):
        pose = frame_pose(rmodel, rdata, q, "panda_hand_tcp")
        viewer[f"{SCENE_ROOT}/path/{i:04d}"].set_object(
            g.Sphere(0.006),
            g.MeshBasicMaterial(color=PATH_COLOR, transparent=True, opacity=0.7),
        )
        viewer[f"{SCENE_ROOT}/path/{i:04d}"].set_transform(
            mtf.translation_matrix(pose.translation)
        )


def _connectable_zmq_url(zmq_url: str) -> str:
    return re.sub(r"tcp://(?:0\.0\.0\.0|\*)", "tcp://127.0.0.1", zmq_url)


def start_meshcat_server(host: str, web_port: int) -> tuple[subprocess.Popen, str, str]:
    server_code = """
from meshcat.servers.zmqserver import ZMQWebSocketBridge
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--web-port", type=int, default=7000)
args = parser.parse_args()

bridge = ZMQWebSocketBridge(host=args.host, port=args.web_port)
print(f"zmq_url={bridge.zmq_url}", flush=True)
print(f"web_url={bridge.web_url}", flush=True)
bridge.run()
"""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            server_code,
            "--host",
            host,
            "--web-port",
            str(web_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    zmq_url = ""
    web_url = ""
    while not (zmq_url and web_url):
        line = proc.stdout.readline() if proc.stdout is not None else ""
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(
                f"Meshcat server exited with code {proc.returncode}:\n{stderr}"
            )
        line = line.strip()
        if line.startswith("zmq_url="):
            zmq_url = line.split("=", 1)[1]
        elif line.startswith("web_url="):
            web_url = line.split("=", 1)[1]

    def cleanup() -> None:
        proc.kill()
        proc.wait()

    atexit.register(cleanup)
    return proc, _connectable_zmq_url(zmq_url), web_url


def make_visualizer(
    zmq_url: str | None,
    open_browser: bool,
    host: str,
    web_port: int,
):
    rmodel, cmodel, vmodel = load_panda()
    viz = visualize.MeshcatVisualizer(
        model=rmodel,
        collision_model=cmodel,
        visual_model=vmodel,
    )
    web_url = None
    if zmq_url is None:
        _, zmq_url, web_url = start_meshcat_server(host=host, web_port=web_port)

    viewer = meshcat.Visualizer(zmq_url=zmq_url)
    if open_browser:
        viewer.open()
    viz.initViewer(viewer=viewer)
    viz.clean()
    viz.loadViewerModel("pinocchio")
    viz.displayCollisions(False)
    viz.displayVisuals(True)
    viz.viewer["/Grid"].set_property("visible", False)
    return viz, rmodel, rmodel.createData(), web_url


def animate(
    viz,
    rmodel: pin.Model,
    rdata: pin.Data,
    trajectory_entry: dict[str, Any],
    fps: float,
    pause: bool,
    loop: bool,
) -> None:
    q_traj = np.asarray(trajectory_entry["trajectory"], dtype=np.float64)
    primitives = trajectory_entry["scene"]["primitives"]
    b = primitives["B_tomato"]
    b_start = pin.XYZQUATToSE3([*b["pose_xyz_m"], *b["pose_quat_xyzw"]])
    grasp_index = trajectory_entry["phase_boundaries"]["grasp_index"]
    attach_index = trajectory_entry["phase_boundaries"].get("close_end_index", grasp_index)
    tcp_at_grasp = frame_pose(rmodel, rdata, q_traj[grasp_index], "panda_hand_tcp")
    tcp_t_b = tcp_at_grasp.inverse() * b_start
    b_node = viz.viewer[f"{SCENE_ROOT}/moving/B_tomato"]

    delay = 1.0 / max(fps, 1e-6)
    while True:
        for i, q in enumerate(q_traj):
            viz.display(display_configuration(q, rmodel.nq))
            if i < attach_index:
                world_t_b = b_start
            else:
                world_t_b = frame_pose(rmodel, rdata, q, "panda_hand_tcp") * tcp_t_b
            b_node.set_transform(world_t_b.homogeneous @ cylinder_mesh_transform())

            if pause:
                input(f"step {i + 1}/{len(q_traj)} - Enter for next")
            else:
                time.sleep(delay)

        if not loop:
            break


def main() -> int:
    args = parse_args()
    entry = load_trajectory(args.dataset, args.index)
    viz, rmodel, rdata, web_url = make_visualizer(
        args.zmq_url,
        args.open,
        args.host,
        args.web_port,
    )

    add_scene_primitives(viz.viewer, entry)
    q_traj = np.asarray(entry["trajectory"], dtype=np.float64)
    add_ee_path(viz.viewer, rmodel, rdata, q_traj, args.path_stride)

    print("Meshcat viewer:", web_url or viz.viewer.url())
    print("Open the /static/ URL, not bare localhost:7000.")
    print(f"Dataset: {args.dataset}")
    print(f"Trajectory index: {args.index}")
    print("Press Ctrl+C to stop.")

    animate(
        viz=viz,
        rmodel=rmodel,
        rdata=rdata,
        trajectory_entry=entry,
        fps=args.fps,
        pause=args.pause,
        loop=args.loop,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
