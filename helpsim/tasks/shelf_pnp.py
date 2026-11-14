"""Shelf pick and place for the station DROID: one kitchen object on upper_return's middle shelf.

Each episode stages the object at a random reachable cell on the shelf (its doors open) and asks for it at another
cell of the same shelf. A scripted expert approaches level into the cabinet (jaws horizontal), grasps, lifts, carries,
lowers and releases, with Cartesian waypoints solved by IK and tracked by the stock position servos. A failed grasp
or a slip on the shelf is retried from the state it left, with a perturbed grasp, up to ``max_attempts``; each attempt
is logged with its outcome, so post-failure states are real data for the value function.

Outcomes per attempt: ``success``, ``unreachable`` (IK off by > 1 cm or 10 deg on a waypoint), ``grasp_miss`` (the
pads do not both touch the object after closing), ``slip`` (held at closing, lost while carried), ``place_miss``
(released > 5 cm from the target), ``off_shelf`` (the object left the shelf; terminal).

Faults (``FAULTS``) degrade the expert in the ways a real pick and place commonly fails, to collect bad trajectories
whose cause is known: an error in the object's estimated position (missed or knocked grasps), an error in where it
believes the target is (misplacements), a slippery object, and a weak grip (slips under load). They are fixed per
``ShelfPnP`` and hidden from the recorded state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from helpsim.envs.kitchen_station import HOME, PREFIX, build_spec, robot_scene_contacts
from helpsim.ik import ArmIK, grasp_frame

SHELF = "interior/upper_return_0_1_upper_shelf_collision"
DOORS = "cabinet/upper_return__door_"
# Reachable front half of the shelf from the door-session cart (scripts/shelf_reach.py): object centres.
REGION_X = (8.975, 9.03)
REGION_Y = (0.40, 0.74)
CABINET_CAM_EYE = (8.05, 0.15, 1.95)
CABINET_CAM_LOOK = (8.98, 0.57, 1.50)
CONTROL_HZ = 15
STEP_M = 0.012          # Cartesian step per control tick (~18 cm/s)
STEP_RAD = 0.05         # rotation step per control tick (~0.75 rad/s)
CLOSE_S, OPEN_S, SETTLE_S = 1.0, 0.6, 0.8
PLACE_TOL_M = 0.05
FAULTS = {
    "pose_noise_m": 0.0,      # sigma of the expert's object-position estimate (x, y; half in z), redrawn each attempt
    "place_noise_m": 0.0,     # sigma of its target-position estimate (x, y), redrawn each attempt
    "grip_friction": None,    # sliding friction between the pads and the object (a slippery object), the shelf's unchanged;
                              # unused: at 0.08 the kitchen's light objects still hold, and slips come from a weak grip
    "grip_scale": 1.0,        # the gripper actuator's force limit, scaled
}


def objects_dir() -> Path:
    import os
    return Path(os.environ["XEMB"]) / "envs" / "src_kitchen" / "objects"


@dataclass
class Recorder:
    rows: dict = field(default_factory=lambda: {k: [] for k in (
        "t", "arm_qpos", "arm_target", "gripper_ctrl", "gripper_pos", "tcp_pos", "tcp_quat", "obj_pos", "obj_quat",
        "phase", "attempt")})
    frames: dict = field(default_factory=dict)
    frame_tick: list = field(default_factory=list)


class ShelfPnP:
    PHASES = ("settle", "approach", "grasp", "close", "lift", "carry", "lower", "release", "retreat")

    def __init__(self, obj: str, station: str = "droid_corner_20261029", render: bool = True,
                 size: tuple[int, int] = (180, 320), render_every: int = 3, **faults):
        unknown = set(faults) - set(FAULTS)
        if unknown:
            raise ValueError(f"unknown faults {sorted(unknown)}; known: {sorted(FAULTS)}")
        self.faults = {**FAULTS, **faults}
        spec = build_spec(station)
        odir = objects_dir()
        ospec = mujoco.MjSpec.from_file(str(odir / f"{obj}.xml"))
        for asset in list(ospec.meshes) + list(ospec.textures):
            if asset.file and not Path(asset.file).is_absolute():
                asset.file = str(odir / asset.file)
        spec.worldbody.add_frame().attach_body(ospec.body(obj), "obj/", "")
        # A synthetic third-person view of the shelf (no real camera stands here: the record's ZED 2i is calibrated for
        # the island station and from this cart sees the counter).
        eye, look = np.array(CABINET_CAM_EYE), np.array(CABINET_CAM_LOOK)
        back = (eye - look) / np.linalg.norm(eye - look)
        right = np.cross([0.0, 0.0, 1.0], back)
        right /= np.linalg.norm(right)
        spec.worldbody.add_camera(name="cam_cabinet", pos=eye, xyaxes=np.concatenate([right, np.cross(back, right)]),
                                  fovy=55)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.obj = obj
        self.ik = ArmIK(self.model)
        m = self.model
        self.obj_body = m.body(f"obj/{obj}").id
        self.obj_qadr = m.jnt_qposadr[m.joint(f"obj/{obj}_free").id]
        self.obj_geoms = set(np.flatnonzero(m.geom_bodyid == self.obj_body))
        self.pads = {m.body(f"{PREFIX}gripper/{s}_pad").id for s in ("left", "right")}
        self.grip_act = m.actuator(f"{PREFIX}gripper/fingers_actuator").id
        self.arm_act = [m.actuator(f"{PREFIX}fr3_joint{i}").id for i in range(1, 8)]
        self.site = self.ik.site
        g = m.geom(SHELF).id
        mujoco.mj_forward(m, self.data)
        self.shelf_top = float(self.data.geom_xpos[g][2] + m.geom_size[g][2])
        R = self.data.geom_xmat[g].reshape(3, 3)
        depth = int(np.argmin(m.geom_size[g][:2]))
        root = self.data.body(f"{PREFIX}fr3_link0").xpos
        self.into = R[:, depth] * np.sign(R[:, depth] @ (self.data.geom_xpos[g] - root))
        self.substeps = max(1, round(1.0 / CONTROL_HZ / m.opt.timestep))
        if self.faults["grip_friction"] is not None:
            # On the pads, with contact priority over the object (MuJoCo otherwise takes the larger friction of a pair):
            # the object's contact with the shelf keeps its own friction, so it does not slide away when touched.
            for g in np.flatnonzero(np.isin(m.geom_bodyid, list(self.pads))):
                m.geom_friction[g, 0] = self.faults["grip_friction"]
                m.geom_priority[g] = 2
        m.actuator_forcerange[self.grip_act] *= self.faults["grip_scale"]
        self.renderer = mujoco.Renderer(m, *size) if render else None
        self.render_every = render_every  # frames at CONTROL_HZ / render_every; frame_tick maps frames to ticks
        self.cams = [c for c in ("cam_cabinet", f"{PREFIX}gripper/wrist_camera")
                     if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) >= 0]

    # ---------------------------------------------------------------- state helpers
    def obj_pose(self):
        return self.data.xpos[self.obj_body].copy(), self.data.xquat[self.obj_body].copy()

    def obj_com(self):
        return self.data.xipos[self.obj_body].copy()

    def arm_q(self):
        return self.data.qpos[self.ik.qadr].copy()

    def pads_touching(self) -> int:
        touching = set()
        for c in self.data.contact[: self.data.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.obj_geoms and self.model.geom_bodyid[b] in self.pads:
                    touching.add(int(self.model.geom_bodyid[b]))
        return len(touching)

    def on_shelf(self) -> bool:
        p = self.obj_com()
        return p[2] > self.shelf_top - 0.02 and abs(p[2] - self.shelf_top) < 0.25

    # ---------------------------------------------------------------- episode
    def reset(self, rng: np.random.Generator, obj_xy=None, yaw=None, target_xy=None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for j in range(m.njnt):
            name = m.joint(j).name
            if name.startswith(DOORS) and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                lo, hi = m.jnt_range[j]
                d.qpos[m.jnt_qposadr[j]] = hi if abs(hi) > abs(lo) else lo
        d.qpos[self.ik.qadr] = HOME
        d.ctrl[self.arm_act] = HOME
        d.ctrl[self.grip_act] = 0
        xy = np.array(obj_xy) if obj_xy is not None else rng.uniform([REGION_X[0], REGION_Y[0]], [REGION_X[1], REGION_Y[1]])
        yaw = rng.uniform(-np.pi, np.pi) if yaw is None else yaw
        q = d.qpos[self.obj_qadr: self.obj_qadr + 7]
        q[:3] = [xy[0], xy[1], self.shelf_top + 0.05]
        q[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        low = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in self.obj_geoms if m.geom_contype[g] or m.geom_conaffinity[g])
        q[2] -= low - (self.shelf_top + 0.002)
        mujoco.mj_forward(m, d)
        if target_xy is None:
            for _ in range(100):
                target_xy = rng.uniform([REGION_X[0], REGION_Y[0]], [REGION_X[1], REGION_Y[1]])
                if abs(target_xy[1] - xy[1]) > 0.12:
                    break
        self.target = np.array([target_xy[0], target_xy[1], self.shelf_top])
        self.rec = Recorder()
        self.attempt = 0
        self.attempts: list[dict] = []
        self._hold(SETTLE_S, "settle")
        self.start_obj = self.obj_pose()

    def _record(self, phase: str):
        d, r = self.data, self.rec.rows
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[self.site])
        p, qo = self.obj_pose()
        for k, v in (("t", d.time), ("arm_qpos", self.arm_q()), ("arm_target", d.ctrl[self.arm_act].copy()),
                     ("gripper_ctrl", d.ctrl[self.grip_act]), ("gripper_pos", d.actuator_length[self.grip_act]),
                     ("tcp_pos", d.site_xpos[self.site].copy()), ("tcp_quat", quat), ("obj_pos", p), ("obj_quat", qo),
                     ("phase", self.PHASES.index(phase)), ("attempt", self.attempt)):
            r[k].append(v)
        tick = len(r["t"]) - 1
        if self.renderer is not None and tick % self.render_every == 0:
            self.rec.frame_tick.append(tick)
            for cam in self.cams:
                self.renderer.update_scene(d, camera=cam)
                self.rec.frames.setdefault(cam.split("/")[-1], []).append(self.renderer.render())

    def _tick(self, phase: str):
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self._record(phase)

    def _hold(self, seconds: float, phase: str):
        for _ in range(max(1, round(seconds * CONTROL_HZ))):
            self._tick(phase)

    def _gripper(self, close: bool, phase: str):
        start = float(self.data.ctrl[self.grip_act])
        goal = 255.0 if close else 0.0
        n = max(1, round((CLOSE_S if close else OPEN_S) * CONTROL_HZ))
        for i in range(n):
            self.data.ctrl[self.grip_act] = start + (goal - start) * (i + 1) / n
            self._tick(phase)

    def _move(self, pos: np.ndarray, rot: np.ndarray, phase: str) -> bool:
        """Cartesian straight line from the current site pose; False if a waypoint is unreachable."""
        start = self.data.site_xpos[self.site].copy()
        r0 = Rotation.from_matrix(self.data.site_xmat[self.site].reshape(3, 3))
        r1 = Rotation.from_matrix(rot)
        turn = (r0.inv() * r1).magnitude()
        n = max(1, int(np.ceil(max(np.linalg.norm(pos - start) / STEP_M, turn / STEP_RAD))))
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        q = self.data.ctrl[self.arm_act].copy()
        for i in range(1, n + 1):
            q, pe, re = self.ik.solve(self.data.qpos, start + (pos - start) * i / n, slerp(i / n).as_matrix(), q)
            if pe > 0.01 or re > np.radians(10):
                return False
            self.data.ctrl[self.arm_act] = q
            self._tick(phase)
        for _ in range(4):  # let the servos settle on the last waypoint
            self._tick(phase)
        return True

    SEEDS = (HOME, (0.6, -0.3, 0.0, -1.6, 0.0, 1.4, 0.8), (0.3, 0.2, 0.0, -1.2, 0.0, 1.6, 0.8),
             (0.9, -0.1, -0.3, -1.4, 0.2, 1.5, 0.6))

    def _move_joint(self, pos: np.ndarray, rot: np.ndarray, phase: str, max_step_rad: float = 0.06) -> bool:
        """Free-space move: the goal solved from several seeds, then joint-space interpolation (no Cartesian line, so
        no IK continuation through a dead end); False if no seed reaches the goal."""
        cur = self.data.ctrl[self.arm_act].copy()
        best = None
        for seed in (cur, *self.SEEDS):
            q, pe, re = self.ik.solve(self.data.qpos, pos, rot, np.asarray(seed, float), iters=300)
            if pe < 0.005 and re < np.radians(5):
                cost = float(np.abs(q - cur).max())
                if best is None or cost < best[0]:
                    best = (cost, q)
        if best is None:
            return False
        n = max(1, int(np.ceil(best[0] / max_step_rad)))
        for i in range(1, n + 1):
            self.data.ctrl[self.arm_act] = cur + (best[1] - cur) * i / n
            self._tick(phase)
        for _ in range(4):
            self._tick(phase)
        return True

    def run(self, rng: np.random.Generator, max_attempts: int = 3) -> dict:
        outcome = "unreachable"
        for self.attempt in range(max_attempts):
            outcome = self._attempt(rng, perturb=self.attempt > 0)
            self.attempts.append({"attempt": self.attempt, "outcome": outcome, "fail_phase": self.fail_phase,
                                  "t_end": float(self.data.time),
                                  "step_end": len(self.rec.rows["t"])})
            if outcome in ("success", "off_shelf", "place_miss"):
                break
        return {"success": outcome == "success", "final_outcome": outcome, "attempts": self.attempts}

    def _attempt(self, rng, perturb: bool) -> str:
        com = self.obj_com()
        if self.faults["pose_noise_m"]:      # drawn only under the fault, so nominal episodes keep their random stream
            com = com + rng.normal(0, self.faults["pose_noise_m"] * np.array([1.0, 1.0, 0.5]))
        offset = rng.normal(0, [0.008, 0.008, 0.006]) if perturb else np.zeros(3)
        grasp = np.array([com[0], com[1], max(com[2], self.shelf_top + 0.026)]) + offset
        grasp[2] = max(grasp[2], self.shelf_top + 0.018)
        # A retry also swings the approach 15-30 deg to one side, which turns the jaws across a long object.
        swing = rng.choice([-1, 1]) * rng.uniform(np.radians(15), np.radians(30)) if perturb else 0.0
        into = Rotation.from_euler("z", swing).apply(self.into)
        rot = grasp_frame(into, roll_rad=rng.normal(0, 0.1) if perturb else 0.0)
        pre = grasp - 0.14 * into
        self.fail_phase = None
        if not self._move_joint(pre + [0, 0, 0.03], rot, "approach") or not self._move(pre, rot, "approach"):
            self.fail_phase = "approach"
            return "unreachable"
        if not self._move(grasp, rot, "grasp"):
            self.fail_phase = "grasp"
            return "unreachable"
        self._gripper(True, "close")
        if self.pads_touching() < 2:
            self._gripper(False, "release")
            self._move(pre, rot, "retreat")
            return "off_shelf" if not self.on_shelf() else "grasp_miss"
        lift = grasp + [0, 0, 0.04]
        place = np.array([self.target[0], self.target[1], grasp[2]])
        if self.faults["place_noise_m"]:
            place[:2] += rng.normal(0, self.faults["place_noise_m"], 2)
        ok = self._move(lift, rot, "lift") and self._move(lift - 0.06 * into, rot, "carry") and \
            self._move(place + [0, 0, 0.04] - 0.06 * into, rot, "carry") and \
            self._move(place + [0, 0, 0.04], rot, "carry")
        if not ok:
            self.fail_phase = "carry"
            self._gripper(False, "release")
            return "unreachable"
        if self.pads_touching() < 2:
            self._gripper(False, "release")
            self._move(self.data.site_xpos[self.site] - 0.14 * into, rot, "retreat")
            return "off_shelf" if not self.on_shelf() else "slip"
        self._move(place + [0, 0, 0.006], rot, "lower")
        self._gripper(False, "release")
        self._move(self.data.site_xpos[self.site] - 0.16 * into, rot, "retreat")
        self._hold(0.5, "retreat")
        if not self.on_shelf():
            return "off_shelf"
        err = float(np.linalg.norm(self.obj_com()[:2] - self.target[:2]))
        return "success" if err < PLACE_TOL_M else "place_miss"

    def arrays(self) -> dict:
        return {**{k: np.asarray(v) for k, v in self.rec.rows.items()}, "frame_tick": np.asarray(self.rec.frame_tick)}

    def robot_scene_contacts(self):
        return robot_scene_contacts(self.model, self.data)
