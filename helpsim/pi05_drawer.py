"""pi0.5 on the TidyBot drawer task in sim: a policy for xemb's datagen pipeline in place of the kino planner.

``scripts/run_drawer_pi05.sh`` runs xemb's ``run_pipeline.py`` exactly as the drawer collection does
(``scripts/collect_drawer_open.sh``: the same scene, drawer, robot, start pose, cameras, physics, 250 Hz control and
``judge_success``), but each episode's policy is ``Pi05TidybotPolicy``. The source of its action chunks is
``$HELP_PI05_SOURCE``:

- ``ws://host:port`` or ``wss://<name>.trycloudflare.com``: the Colab policy server (``scripts/openpi_run.py serve
  drawer``);
- ``replay:<trajectories h5>:<traj index>``: a recorded episode's own commands, to test the loop without a model
  (the same seed gives the same start, so a recorded success should open the drawer again).

Observations and actions match ``scripts/to_lerobot.py``'s drawer dataset: ``exo_camera_1`` (rendered from the mast
camera by ``helpsim/xemb_hooks.py``) and ``wrist_camera`` resized to 180x320; state = 7 arm joints, gripper, base
(x, y, yaw) in the episode's first base frame; action = 7 arm joint targets, gripper command / 255, base target in the
same frame. The observation at step k gets the command for step k+1, and each chunk's first ``$HELP_PI05_EXEC``
(default 8) commands are executed, one per policy step (~15 Hz). A base target within 2 mm (0.005 rad) of the last one sent is sent as
the last one, unchanged (BASE_DEADBAND_M): the measured arm stands still while the base is commanded. A close holds until
pi0.5 asks for open 8 steps in a row (GRIP_RELEASE_STEPS): its chunks flicker back to open before the jaws have moved.

**Attempts** (TASK_SPEC.md §4), from sim state before every policy step (``DrawerMonitor``):

- ``success``: the task's ``judge_success`` (drawer open);
- ``grasp_miss``: the gripper commanded closed for 2.5 s without both finger pads on the drawer (the expert takes
  1.2 s from the close command to contact);
- ``hold_lost``: after a hold (both pads on the drawer, gripper closed), the pads off it for 0.3 s, slipped or let go;
- ``stall``: drawer travel under 5 mm over the last 2 s of a hold;
- ``timeout``: none of these within 25 s (``truncated``: the task's step limit ended the episode first).

**Decisions** (§5) come at the start and after each failed attempt, at most ``$HELP_PI05_ATTEMPTS`` (default 3)
attempts: ``$HELP_PI05_DECIDE`` = ``reset`` (default: the jaws open, the base backs off the handle, the hand returns to
its start pose and the base to its start, then pi0.5 again; RESET_* below) or ``continue`` (pi0.5 goes on from where
it is). The episode ends (the policy returns None) at a success or after the last attempt. Each attempt's
record (outcome, the drawer's and the base's travel, its policy steps) and each decision point (state, and pi0.5's
spread over ``$HELP_PI05_SAMPLES`` sampled chunks if set) go into the trajectory's ``obs_scene`` as ``help_attempts``
and ``help_decisions``; ``scripts/drawer_attempts.py`` writes them to the run's ``attempts.jsonl`` for
``helpsim/help/trees.py``.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

H, W = 180, 320
DRAWER = "drawer/glass_counter__stack_b_3_front"
PADS = ("robot_0/arm/left_finger_pad_1", "robot_0/arm/left_finger_pad_2", "robot_0/arm/right_finger_pad_1",
        "robot_0/arm/right_finger_pad_2")
ATTEMPT_S = 25.0
GRASP_S = 2.5                  # closed this long without a hold: a grasp miss (nominal: 1.2 s from close to contact)
LOST_S = 0.3                   # pads off the drawer this long after a hold: the hold is lost
STALL_S, STALL_M = 2.0, 0.005
CLOSING = 0.5                  # gripper command (fraction of 255) that counts as closing
# The gripper latch. pi0.5's state has the jaws' position, not its last command: after a close command the jaws have
# barely moved when the next chunk is planned (in the expert data, frames with the jaws part-closed are 2-3 per
# episode, against many open ones before the close), and the chunks fall back to open. Executed as given, the jaws
# never close (commands 255 255 0 0 0 0 0 255 ...; the jaws stall at 0.24, the expert's reach 0.79). So a close holds
# until pi0.5 asks for open GRIP_RELEASE_STEPS steps in a row ($HELP_PI05_GRIP_RELEASE; 0 turns the latch off): the
# flicker's open runs were at most 5 steps, the expert's release is a long one.
GRIP_RELEASE_STEPS = int(os.environ.get("HELP_PI05_GRIP_RELEASE", "8"))
# Reset, through xemb's measured command path (``molmo_spaces/controllers/measured.py``): the arm tracks a TCP pose (a
# joint target goes through FK, the hub's 40 mm lead and the mode-7 tracker's IK), so the hand returns to its start
# pose but the redundant 7-joint arm need not return to its start joints (they drift 0.3-0.5 rad in its null space,
# even holding still; the planner's re-pull returns the same way); the arm is stopped while the base drives; the base
# is the teleop's open-loop velocity (a moving target is a velocity, covering ~90% of each move; a held target stops
# it). So: the jaws open with
# arm and base held; the base backs 15 cm away from the handle; the hand goes to its start pose, then the arm to its
# start joints; the base drives back;
# then, at most 5 s, the base is steered onto its start (RESET_GAIN times the error, within the teleop's caps) until it
# is within 5 mm and 0.01 rad of it and the TCP within 1 cm of its start position in the base frame.
# Joint homing: through the TCP path the joints came back 0.3-2.8 rad from their start (the hand within 8 mm), and pi0.5,
# which reads the joints and was trained from one start configuration, never succeeded after a reset. So once the hand
# is home (TCP path, as before: a joint-space path straight from where an attempt ended swept the fingers through a
# cabinet front and dragged it open), the arm goes to its start joints in joint space, as the xArm does in its joint
# mode, mostly a self-motion with the hand in place and the base backed off: its servos follow the start joints directly
# (``install_joint_homing``: MeasuredArmController's own held-joints path, the one it uses while the base drives), at
# most JOINT_STEP_RAD a step (under xemb's 0.066 rad command cap), and hold them through the drive back and the settle;
# when pi0.5 takes over, the controller re-arms mode 7 at the measured pose, as it does after a base move. The jaws are
# put back to the episode's start opening on the settle's last step (``DrawerSim.restore_jaws``): every episode starts
# with them wider than the G2's commanded open (84 mm at most, the bridge's 75 mm), and retries began with the arm at
# its start and the jaws at 75 mm (state 0.16-0.23 against 0.0), a pair pi0.5 never saw in training.
RESET_OPEN_S, RESET_BACK_M, RESET_HOME_S, RESET_BASE_SPEED = 1.2, 0.15, 2.5, 0.05
JOINT_STEP_RAD = 0.05
JOINT_HOME = "_joint_home"     # the reset queue's marker for the joint-space phase (built from the joints then)
RESET_TOL_M, RESET_TOL_YAW, RESET_TOL_TCP_M, RESET_SETTLE_MAX_S = 0.005, 0.01, 0.01, 5.0
RESET_GAIN, BASE_CAPS = 2.0, np.array([0.15, 0.15, 0.5])
# The base's deadband for pi0.5's base targets. The measured arm is stopped whenever the base is commanded, and any
# change of the base target, however small, commands it (MeasuredBaseController.set_target: the change over the time
# since the last one becomes a velocity). The expert sends the very same base target while the arm works (71% of its
# steps); when it drives the base, each step moves it 2 mm or more (median 5.8 mm). pi0.5's base output wanders by
# ~0.4 mm a step, so taken as it is, the base "drives" all the time and the arm never moves. A pi0.5 base target is
# therefore sent only when it is at least BASE_DEADBAND_M (or BASE_DEADBAND_YAW) from the last base target sent;
# otherwise the last one is sent again, unchanged.
BASE_DEADBAND_M, BASE_DEADBAND_YAW = 0.002, 0.005
TCP_SITE = "robot_0/arm/link_tcp"
DEBUG = bool(os.environ.get("HELP_PI05_DEBUG"))   # a line per policy step and per reset step
STUCK_M = 0.05                 # scripts/drawer_attempts.py: a failed pull under 5 cm of travel is "stuck"


def to_first_frame(pose: np.ndarray, base0: np.ndarray) -> np.ndarray:
    """[..., (x, y, yaw)] world poses in base0's frame (as ``scripts/to_lerobot.in_first_frame``)."""
    pose = np.atleast_2d(np.asarray(pose, float))
    c, s = np.cos(base0[2]), np.sin(base0[2])
    d = pose[:, :2] - base0[:2]
    yaw = (pose[:, 2] - base0[2] + np.pi) % (2 * np.pi) - np.pi
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], yaw], 1)


def to_world(pose: np.ndarray, base0: np.ndarray) -> np.ndarray:
    """The inverse of ``to_first_frame`` for one pose."""
    c, s = np.cos(base0[2]), np.sin(base0[2])
    x, y, yaw = pose
    return np.array([base0[0] + c * x - s * y, base0[1] + s * x + c * y,
                     (base0[2] + yaw + np.pi) % (2 * np.pi) - np.pi])


def resize(img: np.ndarray) -> np.ndarray:
    if img.shape[:2] == (H, W):
        return img
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(img, np.uint8)).resize((W, H), Image.BILINEAR))


class DrawerSim:
    """The quantities the monitor reads from xemb's running sim."""

    def __init__(self, task):
        import mujoco

        self.task = task
        m = task.env.current_model
        self.model = m
        j = m.joint(f"{DRAWER}/slide")
        self.slide = int(m.jnt_qposadr[j.id])
        self.slide_range = float(abs(m.jnt_range[j.id][1] - m.jnt_range[j.id][0]))
        self.threshold = float(task.config.task_config.task_success_threshold)
        root = int(m.jnt_bodyid[j.id])
        bodies = {root}
        for b in range(m.nbody):      # the drawer body and everything below it (front, handle)
            p = b
            while p > 0 and p not in bodies:
                p = int(m.body_parentid[p])
            if p in bodies:
                bodies.add(b)
        self.drawer_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies}
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in PADS]
        if min(ids) < 0:
            raise RuntimeError(f"finger pad geoms not found: {dict(zip(PADS, ids))}")
        self.left, self.right = set(ids[:2]), set(ids[2:])
        self.base = [int(m.jnt_qposadr[m.joint(f"robot_0/joint_{a}").id]) for a in ("x", "y")]
        self.yaw = int(m.jnt_qposadr[m.joint("robot_0/joint_th").id])
        self.tcp = m.site(TCP_SITE).id
        # The gripper's joints (drivers and their linkage) where the episode starts (DrawerSim is built on its first step).
        jaws = [j for j in range(m.njnt) if m.joint(j).name.startswith("robot_0/arm/")
                and any(k in m.joint(j).name for k in ("driver_joint", "finger_joint", "knuckle_joint"))]
        self.jaw_q = np.array([int(m.jnt_qposadr[j]) for j in jaws])
        self.jaw_v = np.array([int(m.jnt_dofadr[j]) for j in jaws])
        self.jaw0 = self.data.qpos[self.jaw_q].copy()

    @property
    def data(self):
        return self.task.env.current_data

    def restore_jaws(self, gripper_controller) -> None:
        """The jaws back to the episode's start opening (free of contact, at the end of a reset); the measured jaws'
        emulator restarts from there."""
        import mujoco

        d = self.data
        d.qpos[self.jaw_q] = self.jaw0
        d.qvel[self.jaw_v] = 0.0
        mujoco.mj_forward(self.model, d)
        if gripper_controller is not None:
            gripper_controller.reset()

    def drawer_m(self) -> float:
        return float(self.data.qpos[self.slide])

    def base_xy(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.base], float)

    def tcp_in_base(self) -> np.ndarray:
        """The TCP's position in the base's frame (x forward, y left, z up from the base's origin)."""
        d = self.data
        p = d.site_xpos[self.tcp] - np.r_[self.base_xy(), 0.0]
        c, s = np.cos(d.qpos[self.yaw]), np.sin(d.qpos[self.yaw])
        return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2]])

    def pads_on_drawer(self) -> bool:
        d, left, right = self.data, False, False
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            for pad, other in ((g1, g2), (g2, g1)):
                if other in self.drawer_geoms:
                    left |= pad in self.left
                    right |= pad in self.right
        return left and right

    def success(self) -> bool:
        """The target drawer open past the task's threshold (the task's own rule, on this drawer only). The task's
        judge_success takes the most-open joint of every articulated object in the scene (xemb's "BUG5", fixed there
        only under SANITY_LOWVAR with a joint index): a cabinet front dragged open counted as the drawer."""
        return abs(self.drawer_m()) / self.slide_range >= self.threshold


class DrawerMonitor:
    """Watches one attempt; ``update(grip_cmd)`` before every policy step returns the outcome once it has ended."""

    def __init__(self, sim: DrawerSim, dt: float, step0: int):
        self.sim, self.dt, self.step0 = sim, dt, step0
        self.n = lambda s: max(1, round(s / dt))
        self.steps = 0
        self.close_steps = 0
        self.held = sim.pads_on_drawer()      # a continue may start holding
        self.hold = []                        # drawer travel over the current hold
        self.off = 0
        self.d0 = self.dmax = self.dmin = sim.drawer_m()
        self.b0 = sim.base_xy()
        self.bmin = self.bmax = self.b0.copy()

    def update(self, cmd: float) -> str | None:
        sim = self.sim
        self.steps += 1
        d = sim.drawer_m()
        self.dmax, self.dmin = max(self.dmax, d), min(self.dmin, d)
        b = sim.base_xy()
        self.bmin, self.bmax = np.minimum(self.bmin, b), np.maximum(self.bmax, b)
        if sim.success():
            return "success"
        closing, touching = cmd >= CLOSING, sim.pads_on_drawer()
        if DEBUG:
            print(f"[pi05_drawer] step {self.step0 + self.steps} cmd {cmd:.2f} pads {int(touching)} drawer {d:.4f}",
                  flush=True)
        self.close_steps = self.close_steps + 1 if closing else 0
        if closing and touching:
            self.held, self.off = True, 0
            self.hold.append(d)
            w = self.hold[-self.n(STALL_S):]
            if len(self.hold) >= self.n(STALL_S) and max(w) - min(w) < STALL_M:
                return "stall"
        else:
            self.hold = []
            if self.held:
                self.off += 1
                if self.off >= self.n(LOST_S):
                    return "hold_lost"
            elif self.close_steps >= self.n(GRASP_S):
                return "grasp_miss"
        if self.steps >= self.n(ATTEMPT_S):
            return "timeout"
        return None

    def record(self, k: int, outcome: str, step: int) -> dict:
        travel = self.dmax
        cat = "success" if outcome == "success" else ("stuck" if travel < STUCK_M else "short")
        return {"attempt": k, "outcome": cat, "event": outcome, "drawer_max_m": round(travel, 4),
                "drawer_end_m": round(self.sim.drawer_m(), 4), "drawer_start_m": round(self.d0, 4),
                "base_travel_m": round(float(np.linalg.norm(self.bmax - self.bmin)), 4), "held": self.held,
                "first_step": self.step0, "last_step": step, "steps": self.steps, "start": getattr(self, "start", None)}


class ReplaySource:
    """A recorded trajectory's commands as model-format chunks, read at the policy's step."""

    needs_images = False

    def __init__(self, h5: str, traj: int, horizon: int = 32):
        import h5py

        dec = lambda row: json.loads(bytes(row).rstrip(b"\x00").decode())
        with h5py.File(h5) as h:
            t = h[f"traj_{traj}"]
            q = [dec(r) for r in t["obs/agent/qpos"][:]]
            a = [dec(r) for r in t["actions/joint_pos"][:]]
        base = np.array([x["base"] for x in q], float)
        cmd = np.array([x.get("base", q[k]["base"]) for k, x in enumerate(a)], float)
        self.base0 = base[0]
        self.a = np.concatenate([np.array([x["arm"] for x in a]), np.array([[x["gripper"][0] / 255.0] for x in a]),
                                 to_first_frame(cmd, self.base0)], 1).astype(np.float32)
        self.q0 = q[0]
        self.horizon = horizon

    def infer(self, obs: dict, step: int) -> dict:
        idx = np.clip(np.arange(step + 1, step + 1 + self.horizon), 0, len(self.a) - 1)
        return {"actions": self.a[idx]}


class ServerSource:
    needs_images = True

    def __init__(self, url: str):
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(url)

    def infer(self, obs: dict, step: int) -> dict:
        return self.client.infer(obs)


_SOURCE = None


def source():
    """One source per process (the server connection is kept across episodes)."""
    global _SOURCE
    spec = os.environ["HELP_PI05_SOURCE"]
    if spec.startswith("replay:"):
        h5, traj = spec[len("replay:"):].rsplit(":", 1)
        return ReplaySource(h5, int(traj))      # fresh per episode: replay is for single-episode tests
    if _SOURCE is None:
        _SOURCE = ServerSource(spec)
    return _SOURCE


def make_policy_class():
    from molmo_spaces.policy.base_policy import BasePolicy

    class Pi05TidybotPolicy(BasePolicy):
        def __init__(self, config, task=None):
            super().__init__(config, task)
            self.exec_steps = int(os.environ.get("HELP_PI05_EXEC", "8"))
            self.max_attempts = int(os.environ.get("HELP_PI05_ATTEMPTS", "3"))
            self.decide_mode = os.environ.get("HELP_PI05_DECIDE", "reset")
            self.n_samples = int(os.environ.get("HELP_PI05_SAMPLES", "0"))
            if self.decide_mode not in ("reset", "continue"):
                raise SystemExit(f"HELP_PI05_DECIDE={self.decide_mode}: reset or continue")
            self.dt = float(getattr(config, "policy_dt_ms", 66.0)) / 1000.0
            self.prompt = "open the drawer"
            self.reset()

        def reset(self):
            self.src = source()
            self.base0, self.chunk, self.i, self.step, self.src_step = None, None, 0, 0, 0
            self.infer_s, self.calls = 0.0, 0
            self.sim = self.monitor = None
            self.attempts, self.decisions, self.queue = [], [], []
            self.last = None          # the last command sent: {"arm", "gripper", "base"}
            self.grip_closed, self.open_run = False, 0     # the gripper latch (GRIP_RELEASE_STEPS)
            self.jaws_restored = True                     # set False by each reset (settle)
            self.settle_left = 0      # policy steps left to wait for the reset to arrive
            self.settle_steps = self.reset_steps = 0
            self.done = False

        # ------------------------------------------------------------ pi0.5 I/O
        def model_input(self, o: dict) -> dict:
            q = o["qpos"]
            base = np.asarray(q["base"], float)
            state = np.concatenate([np.asarray(q["arm"], float)[:7], [float(q["gripper"][0])],
                                    to_first_frame(base, self.base0)[0]]).astype(np.float32)
            inp = {"observation/state": state, "prompt": self.prompt}
            if self.src.needs_images:
                inp["observation/exterior"] = resize(o["exo_camera_1"])
                inp["observation/wrist"] = resize(o["wrist_camera"])
            return inp

        def to_command(self, a) -> dict:
            return {"arm": np.asarray(a[:7], float), "gripper": np.array([self.gripper(float(a[7]))]),
                    "base": self.base_target(np.asarray(a[8:11], float))}

        def gripper(self, g: float) -> float:
            """pi0.5's gripper output (fraction of 255) -> the command, through the latch (GRIP_RELEASE_STEPS above)."""
            if GRIP_RELEASE_STEPS <= 0:
                return float(np.clip(g * 255.0, 0.0, 255.0))
            if not self.grip_closed:
                self.grip_closed, self.open_run = g >= CLOSING, 0
            else:
                self.open_run = self.open_run + 1 if g < CLOSING else 0
                if self.open_run >= GRIP_RELEASE_STEPS:
                    self.grip_closed = False
            return 255.0 if self.grip_closed else 0.0

        def base_target(self, b: np.ndarray) -> np.ndarray:
            """pi0.5's base target (first frame) -> world, through the deadband (BASE_DEADBAND_M above)."""
            last = np.asarray(self.last["base"] if self.last else self.base0, float)
            held = to_first_frame(last, self.base0)[0]
            d = b - held
            d[2] = (d[2] + np.pi) % (2 * np.pi) - np.pi
            if np.hypot(d[0], d[1]) < BASE_DEADBAND_M and abs(d[2]) < BASE_DEADBAND_YAW:
                return last.copy()
            return to_world(b, self.base0)

        def pi05(self, o: dict) -> dict:
            if self.chunk is None or self.i >= self.exec_steps:
                t = time.perf_counter()
                self.chunk = np.asarray(self.src.infer(self.model_input(o), self.src_step)["actions"])
                self.infer_s += time.perf_counter() - t
                self.calls += 1
                self.i = 0
            a = self.chunk[self.i]
            self.i += 1
            self.src_step += 1
            return self.to_command(a)

        def spread(self, o: dict) -> dict:
            """pi0.5's uncertainty here: the spread of n sampled chunks from the same observation."""
            inp = self.model_input(o)
            c = np.stack([np.asarray(self.src.infer(inp, self.src_step)["actions"]) for _ in range(self.n_samples)])
            e = self.exec_steps
            return {"n": self.n_samples, "arm_std_exec": float(c[:, :e, :7].std(0).mean()),
                    "arm_std_chunk": float(c[:, :, :7].std(0).mean()), "grip_std_exec": float(c[:, :e, 7].std(0).mean()),
                    "base_std_exec": float(c[:, :e, 8:11].std(0).mean())}

        # ------------------------------------------------------------ attempts and decisions
        def decision_point(self, k: int, o: dict) -> str:
            q = o["qpos"]
            choice = "continue" if k == 0 else self.decide_mode     # the first attempt starts at the start pose
            dp = {"attempt": k, "choice": choice, "step": self.step,
                  "state": {"arm": np.round(np.asarray(q["arm"], float)[:7], 5).tolist(),
                            "gripper": float(q["gripper"][0]),
                            "base": np.round(to_first_frame(np.asarray(q["base"], float), self.base0)[0], 5).tolist(),
                            "drawer_m": round(self.sim.drawer_m(), 5)}}
            if self.n_samples and self.src.needs_images:
                dp["spread"] = self.spread(o)
            self.decisions.append(dp)
            return choice

        def start_attempt(self, k: int, o: dict) -> None:
            if self.decision_point(k, o) == "reset":
                self.queue = self.reset_commands(o)
                if isinstance(self.src, ReplaySource):
                    self.src_step = 0          # the recording starts again from the start pose
            self.chunk = None                   # pi0.5 re-plans from the attempt's first observation
            self.grip_closed, self.open_run = False, 0     # the reset opened the jaws
            self.monitor = None                 # started when the reset (if any) has finished

        def reset_commands(self, o: dict) -> list[dict]:
            """Reset-and-retry (RESET_* above): open, base back, arm to its start pose, base to its start pose. The hold
            until the robot has arrived follows in ``get_action``."""
            arm = np.asarray(self.last["arm"] if self.last else o["qpos"]["arm"], float)[:7]
            here = to_first_frame(np.asarray(o["qpos"]["base"], float), self.base0)[0]
            back = np.array([min(here[0], -RESET_BACK_M), here[1], here[2]])
            opened = np.array([0.0])
            world = lambda p: to_world(p, self.base0)

            def drive(arm_q, a, b):
                m = max(1, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / (RESET_BASE_SPEED * self.dt))))
                return [{"arm": arm_q, "gripper": opened, "base": world(a + (b - a) * i / m)} for i in range(1, m + 1)]

            out = [{"arm": arm, "gripper": opened, "base": world(here)}] * self.n(RESET_OPEN_S)
            out += drive(arm, here, back)
            n = self.n(RESET_HOME_S)
            out += [{"arm": arm + (self.arm0 - arm) * i / n, "gripper": opened, "base": world(back)}
                    for i in range(1, n + 1)]
            out.append({JOINT_HOME: True, "gripper": opened, "base": world(back)})    # expanded in get_action
            out += [dict(c, joint=self.arm0) for c in drive(self.arm0, back, np.zeros(3))]
            self.settle_left, self.settle_steps, self.reset_steps = self.n(RESET_SETTLE_MAX_S), 0, len(out)
            self.jaws_restored = False
            self.vtarget = out[-1]["base"].copy()
            return out

        def joint_home(self, o: dict, marker: dict) -> list[dict]:
            """The joint-space path from the measured joints (the hand home by now) to the start joints."""
            q = np.asarray(o["qpos"]["arm"], float)[:7]
            n = max(1, int(np.ceil(np.abs(self.arm0 - q).max() / JOINT_STEP_RAD)))
            return [{"arm": q + (self.arm0 - q) * i / n, "gripper": marker["gripper"], "base": marker["base"],
                     "joint": q + (self.arm0 - q) * i / n} for i in range(1, n + 1)]

        def base_error(self, o: dict) -> np.ndarray:
            """The start pose minus the base's pose, world frame, yaw wrapped."""
            e = self.base0 - np.asarray(o["qpos"]["base"], float)
            e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
            return e

        def settle(self, o: dict) -> dict | None:
            """One step of the reset's final hold: steer the base onto its start; None once the robot has arrived."""
            e = self.base_error(o)
            base_ok = np.linalg.norm(e[:2]) < RESET_TOL_M and abs(e[2]) < RESET_TOL_YAW
            arrived = base_ok and np.linalg.norm(self.sim.tcp_in_base() - self.tcp0) < RESET_TOL_TCP_M
            if (arrived or self.settle_left <= 1) and not self.jaws_restored:
                self.sim.restore_jaws(self.controllers().get("gripper"))    # one more hold step: the next
                self.jaws_restored = True                                   # observation shows them restored
                return {"arm": self.arm0, "gripper": np.array([0.0]), "base": self.vtarget.copy(), "joint": self.arm0}
            if arrived:
                return None
            if not base_ok:      # the teleop reads the target's motion as the velocity
                self.vtarget = self.vtarget + np.clip(RESET_GAIN * e, -BASE_CAPS, BASE_CAPS) * self.dt
            return {"arm": self.arm0, "gripper": np.array([0.0]), "base": self.vtarget.copy(), "joint": self.arm0}

        def n(self, s: float) -> int:
            return max(1, round(s / self.dt))

        def get_action(self, observation):
            o = observation[0] if isinstance(observation, (list, tuple)) else observation
            if self.done:
                return None
            if self.base0 is None:
                # xemb keeps the robot across episodes, and its arm-command cap (action_constraints.ArmCommandCap)
                # limits each command to 0.066 rad from the robot's last command: without this, the first commands
                # slew from where the previous episode ended, as the planner policies avoid with the same call.
                self.task.env.current_robot.reset_command_history()
                self.joint_target(None)
                self.base0 = np.asarray(o["qpos"]["base"], float).copy()
                self.arm0 = np.asarray(o["qpos"]["arm"], float)[:7].copy()
                self.sim = DrawerSim(self.task)
                self.tcp0 = self.sim.tcp_in_base()
                if isinstance(self.src, ReplaySource):     # the replayed episode must start where this one does
                    d = max(np.abs(np.asarray(o["qpos"]["arm"]) - self.src.q0["arm"]).max(),
                            np.abs(self.base0 - self.src.base0).max())
                    print(f"[pi05_drawer] replay start differs from the recording by {d:.2e}", flush=True)
                self.start_attempt(0, o)
            if self.monitor is not None:
                out = self.monitor.update(float(self.last["gripper"][0]) / 255.0 if self.last else 0.0)
                if out:
                    k = len(self.attempts)
                    self.attempts.append(self.monitor.record(k, out, self.step))
                    print(f"[pi05_drawer] attempt {k + 1}: {out} (drawer max {self.monitor.dmax * 100:.1f} cm, "
                          f"{self.monitor.steps} steps)", flush=True)
                    if out == "success" or k + 1 >= self.max_attempts:
                        self.done = True
                        return None
                    self.start_attempt(k + 1, o)
            if self.queue:
                if JOINT_HOME in self.queue[0]:
                    self.queue[:1] = self.joint_home(o, self.queue[0])
                cmd = self.queue.pop(0)
                if DEBUG:
                    q = o["qpos"]
                    print(f"[pi05_drawer] reset {len(self.queue)} left: arm err "
                          f"{np.abs(np.asarray(q['arm'], float)[:7] - np.asarray(cmd['arm'])).max():.3f} to home "
                          f"{np.abs(np.asarray(q['arm'], float)[:7] - self.arm0).max():.3f} base "
                          f"{np.round(to_first_frame(np.asarray(q['base'], float), self.base0)[0], 4).tolist()} cmd "
                          f"{np.round(to_first_frame(cmd['base'], self.base0)[0], 4).tolist()} drawer {self.sim.drawer_m():.4f}",
                          flush=True)
            elif self.settle_left > 0 and (cmd := self.settle(o)) is not None:
                self.settle_left -= 1
                self.settle_steps += 1
                if DEBUG and self.settle_left % 15 == 0:
                    print(f"[pi05_drawer] settling: base error {np.round(self.base_error(o), 4).tolist()} tcp error "
                          f"{np.linalg.norm(self.sim.tcp_in_base() - self.tcp0):.4f} m", flush=True)
            else:
                self.settle_left = 0
                if self.monitor is None:
                    self.monitor = DrawerMonitor(self.sim, self.dt, self.step)
                    self.monitor.start = {     # where the attempt starts, relative to the episode's start
                        "base": np.round(to_first_frame(np.asarray(o["qpos"]["base"], float), self.base0)[0], 4).tolist(),
                        "arm_max_diff": round(float(np.abs(np.asarray(o["qpos"]["arm"], float)[:7] - self.arm0).max()), 4),
                        "tcp_err_m": round(float(np.linalg.norm(self.sim.tcp_in_base() - self.tcp0)), 4),
                        "reset_steps": self.reset_steps, "settle_steps": self.settle_steps}
                cmd = self.pi05(o)
            self.joint_target(cmd.get("joint"))      # pi0.5's commands have none: mode 7 again
            self.last = cmd
            self.step += 1
            return {k: np.asarray(v, float).copy() for k, v in cmd.items() if k != "joint"}

        def controllers(self) -> dict:
            return getattr(self.task.env.current_robot, "_controllers", {})

        def joint_target(self, q) -> None:
            """Joint homing (JOINT_STEP_RAD above): the arm's servos follow ``q``; None hands the arm back to mode 7."""
            c = self.controllers().get("arm")
            if hasattr(c, "_held_q"):           # the measured command path's arm (install_joint_homing)
                c.help_joint_target = None if q is None else np.asarray(q, float).copy()

        def get_info(self) -> dict:
            attempts = list(self.attempts)
            if self.monitor is not None and not self.done:      # the task's step limit cut the attempt
                attempts.append(self.monitor.record(len(attempts), "truncated", self.step))
            info = super().get_info()
            info.update({"policy_name": "pi05_help", "policy_source": os.environ.get("HELP_PI05_SOURCE", ""),
                         "policy_exec_steps": self.exec_steps, "policy_infer_calls": self.calls,
                         "policy_infer_s": round(self.infer_s, 2), "help_decide": self.decide_mode,
                         "help_max_attempts": self.max_attempts, "help_attempts": attempts,
                         "help_decisions": self.decisions})
            return info

    return Pi05TidybotPolicy


def install_joint_homing() -> None:
    """MeasuredArmController with ``help_joint_target`` set: the servos follow that joint target, the arm held against
    gravity, as the controller holds its joints while the base drives (and marked held, so that once the target is
    cleared the next step re-arms mode 7 at the measured pose). Unset, the controller is unchanged."""
    from molmo_spaces.controllers.measured import MeasuredArmController

    original = MeasuredArmController.compute_ctrl_inputs
    if getattr(original, "_help_joint_homing", False):
        return

    def compute_ctrl_inputs(self):
        q = getattr(self, "help_joint_target", None)
        if q is None:
            return original(self)
        self._t = float(self.d.time)
        self._held_q = q.copy()
        self.d.qfrc_applied[self.arm.arm_v] = self.d.qfrc_bias[self.arm.arm_v]
        self._record()
        return q.copy()

    compute_ctrl_inputs._help_joint_homing = True
    MeasuredArmController.compute_ctrl_inputs = compute_ctrl_inputs


def install() -> None:
    """Every episode of the datagen pipeline gets a Pi05TidybotPolicy instead of its configured planner."""
    import molmo_spaces.data_generation.pipeline as pipeline

    install_joint_homing()

    cls = make_policy_class()

    def setup_policy(exp_config, task, preloaded_policy=None, datagen_profiler=None, episode_seed=None):
        policy = cls(task.config, task)
        task.register_policy(policy)
        return policy

    pipeline.setup_policy = setup_policy
