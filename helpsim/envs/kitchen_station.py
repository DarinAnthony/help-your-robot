"""The DROID station in the installed SRC kitchen twin, as one MuJoCo model.

The twin (``$SRC_KITCHEN_TWIN_DIR/scene.xml``) gets the stock ``franka_droid`` (FR3 kinematics = Panda, Robotiq 2F-85)
with its arm root (``fr3_link0``) at a measured station pose and the record's cameras:

- ``ext``: the ZED 2i, fixed to the arm root at ``cameras.ext.T_parent_optical``;
- ``d435``: the RealSense D435, fixed to the arm root (records from 2026-11-09);
- ``wrist``: the model's ``gripper/wrist_camera`` at the record's field of view, as xemb's ``droid_record_cameras``.

No support box: the real station stands on its own table beside the island, which the twin lacks (as xemb's
``SweepIntoBinTaskSampler._place_arm_root``). The arm's joint ranges are the real Panda's working range
(``droid_panda``, ``stock_neighborhoods.DROID_WORKING_JOINT_RANGES``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.measured.rig import mujoco_quat, record
from molmo_spaces.robots.scaling.stock_neighborhoods import DROID_WORKING_JOINT_RANGES

PREFIX = "robot_0/"
HOME = (0.0, -0.7853, 0.0, -2.35619, 0.0, 1.57079, 0.0)  # FrankaRobotConfig.init_qpos["arm"]


@dataclass(frozen=True)
class Station:
    """The arm root's world pose in the installed twin and the record whose cameras it carries."""

    x: float
    y: float
    z: float
    yaw_deg: float
    record: str
    source: str


STATIONS = {
    # Refit 2026-11-09 by ICP of the ZED 2i and D435 depth against the twin of feat/src-kitchen-twin-axis-aligned
    # (droid_panda_20261109 geometry.base_xy_yaw_in_kitchen; z is the measured 0.867, 5 cm below the island top).
    "droid_20261109": Station(6.553007603421816, -2.2443711587430326, 0.867, 85.94809750041327,
                              "droid_panda_20261109", "droid_panda_20261109 geometry.base_xy_yaw_in_kitchen"),
    # The cart facing the return wall's uppers in the 2026-10-28/29 door sessions (notes/20261029_src_kitchen_cabinet_
    # door_measured.md, ~/droid/sysid/results/kitchen_pose_corner.json): it opened upper_return::door_1_front from here.
    # Frame not recorded beyond "twin"; treat as approximate until a real frame is overlaid.
    "droid_corner_20261029": Station(8.188, 0.379, 0.848, 40.6, "droid_panda_20261109",
                                     "notes/20261029_src_kitchen_cabinet_door_measured.md (door-session cart)"),
    # Main's station (house_tasks.SWEEP_BASES["droid_real_v1"]), localized from the 2026-10-29 episodes.
    "droid_real_v1": Station(6.7159, -2.4872, 0.862, 93.572, "droid_panda_20261103",
                             "house_tasks.SWEEP_BASES['droid_real_v1'] / droid_panda_20261103"),
}


def twin_dir() -> Path:
    path = Path(os.environ.get("SRC_KITCHEN_TWIN_DIR", ""))
    if not (path / "scene.xml").is_file():
        raise FileNotFoundError(f"no installed twin at SRC_KITCHEN_TWIN_DIR={path!s} (source env.sh)")
    return path


def _vfov_deg(lens: dict) -> float:
    return math.degrees(2 * math.atan(lens["height"] / 2 / lens["K"][1][1]))


def build_spec(station: str = "droid_20261109", gravcomp: bool = True) -> mujoco.MjSpec:
    """The twin with the DROID at ``STATIONS[station]``, uncompiled.

    gravcomp: the arm's bodies are gravity-compensated, as the real Franka controller does (xemb's
    ``--robot_gravcomp``). Without it the stock position servos let the outstretched arm sag about 1 cm at the upper
    cabinet, enough to drag the open fingers on the shelf."""
    st = STATIONS[station]
    cams = record(st.record)["cameras"]
    spec = mujoco.MjSpec.from_file(str(twin_dir() / "scene.xml"))
    robot = mujoco.MjSpec.from_file(str(Path(get_robot_path("franka_droid")) / "model.xml"))
    for joint in robot.joints:
        if joint.name in DROID_WORKING_JOINT_RANGES:
            joint.range = DROID_WORKING_JOINT_RANGES[joint.name]
    for actuator in robot.actuators:
        if actuator.target in DROID_WORKING_JOINT_RANGES:
            actuator.ctrlrange = DROID_WORKING_JOINT_RANGES[actuator.target]
    link0 = robot.body("fr3_link0")
    for key in ("ext", "d435"):
        if key in cams:
            T = np.asarray(cams[key]["T_parent_optical"], float)
            if cams[key]["parent"] != "base":
                raise ValueError(f"{st.record}: camera {key} is not on the base")
            link0.add_camera(name=f"cam_{key}", pos=T[:3, 3], quat=mujoco_quat(T), fovy=_vfov_deg(cams[key]["lens_parameters"]))
    half = math.radians(st.yaw_deg) / 2
    frame = spec.worldbody.add_frame(pos=[st.x, st.y, st.z], quat=[math.cos(half), 0, 0, math.sin(half)])
    frame.attach_body(robot.worldbody.first_body(), PREFIX, "")  # the fr3 class body holding fr3_link0 at its origin
    spec.camera(f"{PREFIX}gripper/wrist_camera").fovy = _vfov_deg(cams["wrist"]["lens_parameters"])
    if gravcomp:
        for body in spec.bodies:
            if body.name.startswith(PREFIX):
                body.gravcomp = 1.0
    return spec


def build(station: str = "droid_20261109") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compiled model and data with the arm at the home posture."""
    model = build_spec(station).compile()
    data = mujoco.MjData(model)
    set_arm(model, data, HOME)
    return model, data


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, q) -> None:
    for i, value in enumerate(q):
        name = f"{PREFIX}fr3_joint{i + 1}"
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        actuator = model.actuator(name).id
        data.ctrl[actuator] = value
    mujoco.mj_forward(model, data)


def robot_scene_contacts(model: mujoco.MjModel, data: mujoco.MjData, margin: float = 0.0) -> list[tuple[str, str, float]]:
    """Penetrating contacts between a robot body and a non-robot body: (robot body, scene body, depth m)."""
    out = []
    for c in data.contact[: data.ncon]:
        if c.dist > -margin:
            continue
        b1, b2 = (model.body(int(model.geom_bodyid[g])).name for g in (c.geom1, c.geom2))
        r1, r2 = b1.startswith(PREFIX), b2.startswith(PREFIX)
        if r1 != r2:
            out.append((b1 if r1 else b2, b2 if r1 else b1, float(c.dist)))
    return out
