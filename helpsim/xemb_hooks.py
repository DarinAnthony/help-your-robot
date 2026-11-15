"""Hooks on xemb's datagen pipeline, installed by ``scripts/run_pipeline_hooked.py``.

Drawer variants. xemb's arena ``Drawer`` (``molmo_spaces/env/arena/drawer.py``), built for every part the scene
metadata files as a Drawer, overwrites each episode what the scene file says: the slide's frictionloss U(1, 2) N, damping U(0.01, 0.1),
armature U(0, 0.1) and every geom's friction U(0.1, 0.3). So the twin's own drawer values (frictionloss 0.4 N,
damping 4) never reach the simulation, and neither would a variant's. With ``$SRC_KITCHEN_TWIN_DIR/variant.json``
(``scripts/make_drawer_variants.py``) present, the hook applies the variant's values to its drawer after that
randomization; the variant's tendons (the latch) are not touched by it. Without one it does nothing.

Faults, for bad trajectories with a known cause: ``$HELP_FAULT``, a JSON object, set for a whole run.

- ``"hidden_drawer": true``: the planner does not know the variant. Its grasp screen and its rehearsal of the pull run on
  xemb's nominal drawer draw with the variant's latch released; the variant (friction, damping, latch) takes effect
  when the planned pull starts, and is hidden again when the next episode resets the policy (xemb keeps the model
  across episodes). The robot commits to a pull that a stuck or locked drawer then resists, as a real
  robot that cannot see the drawer's state would. Without it, the planner sees the variant and mostly refuses to pull.
- ``"grip_scale": s``: the gripper's force limit is scaled by s from the pull's start to the episode's end (a weak or
  worn grip); the grasp itself closes at full force.
- ``"stop_fraction": f``: the pull is cut to its first f of the planned steps (the base stops early); with
  ``"stop_pulls": k`` only the episode's first k pulls are cut (a one-off stop that a re-pull recovers from).

Re-pulls: ``$HELP_REPULLS=n`` (default 0, xemb unchanged). xemb retries a pull only when its own rehearsal predicts a
slip, so a pull that fails on the real drawer (a hidden variant, a weak grip, an early stop) ends the episode after one
attempt. With n > 0, when a planned pull has finished and the task does not judge the drawer open, the robot lets go
and returns (``_release_return``): the jaws open with the arm held (1.2 s; xemb's 0.25 s gripper action is shorter
than the identified jaws take, and its retreat then re-commands them closed, dragging the drawer), the hand backs off
8 cm, the arm goes home, and the base drives back to where the pull started (a base pull backs the base away even when
the drawer does not follow, out of the arm's reach of the handle). Then the policy's retry (``max_retries``) re-plans
grasp and pull from there, with the drawer where the failed pull left it, at most n times per episode. Each pull is one attempt for the help policy; the
episode's ``obs/extra/policy_num_retries`` numbers them. A hidden drawer is hidden again for every re-plan (the grasp
screen and the rehearsal) and takes effect again when the next pull starts; a weak grip stays weak for the episode.

TidyBot cameras. ``run_pipeline.py`` gives the mobile unified arm the UR5e's camera system: ``exo_camera_1`` at a
fixed offset from the arm root, tuned for a UR5e on a table, which on the TidyBot looks across the kitchen and sees
neither the robot nor the drawer, and the wrist camera at the UR5e's 56.74 deg. For a robot built with
``mast_camera`` (the identified TidyBot's calibrated mast camera), the hook renders ``exo_camera_1`` from it and the
wrist camera at its MJCF field of view (the record's lens, 81.7 deg). The stream keeps its name, which the π0.5
converter reads as the primary image (``pi05/scripts/data/mlspaces_to_rlds.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def install() -> None:
    variant = install_drawer_variant()
    install_faults(variant)
    install_tidybot_cameras()


def fault() -> dict:
    return json.loads(os.environ.get("HELP_FAULT") or "{}")


def install_drawer_variant() -> dict | None:
    path = Path(os.environ.get("SRC_KITCHEN_TWIN_DIR", "")) / "variant.json"
    if not path.is_file():
        return None
    variant = json.loads(path.read_text())
    hidden = bool(fault().get("hidden_drawer"))
    from molmo_spaces.env.arena import drawer as arena_drawer

    original = arena_drawer.Drawer.__init__

    def __init__(self, object_name, data, body_name2id) -> None:
        original(self, object_name, data, body_name2id)
        if object_name == variant["drawer"]:
            if hidden:
                set_latch(self.mj_model, variant, False)
            else:
                apply(self, variant)

    arena_drawer.Drawer.__init__ = __init__
    print(f"[help!] drawer variant {variant['name']} ({path.parent}) {'hidden from the planner on' if hidden else 'pinned on'} "
          f"{variant['drawer']}", flush=True)
    return variant


def set_latch(model, variant: dict, engaged: bool) -> None:
    """The variant's latch (a limited tendon on the slide, make_drawer_variants.py) engaged or released."""
    if variant.get("stop_m") is not None:
        model.tendon_limited[model.tendon(f"{variant['drawer']}/slide/latch").id] = int(engaged)


def reveal(model, variant: dict) -> tuple[float, float]:
    """The variant's slide physics and latch on the running model (the hidden drawer at the pull's start). Returns the
    slide's frictionloss and damping as the planner knew them, for ``conceal``."""
    dof = model.jnt_dofadr[model.joint(f"{variant['drawer']}/slide").id]
    believed = float(model.dof_frictionloss[dof]), float(model.dof_damping[dof])
    if variant.get("frictionloss") is not None:
        model.dof_frictionloss[dof] = variant["frictionloss"]
    if variant.get("damping") is not None:
        model.dof_damping[dof] = variant["damping"]
    set_latch(model, variant, True)
    return believed


def conceal(model, variant: dict, believed: tuple[float, float]) -> None:
    dof = model.jnt_dofadr[model.joint(f"{variant['drawer']}/slide").id]
    model.dof_frictionloss[dof], model.dof_damping[dof] = believed
    set_latch(model, variant, False)


def apply(drawer, variant: dict) -> None:
    """The variant's slide frictionloss and damping and its handle's friction on an arena Drawer."""
    if variant.get("frictionloss") is not None:
        drawer._set_joint_frictionloss(variant["frictionloss"])
    if variant.get("damping") is not None:
        drawer._set_joint_damping(variant["damping"])
    if variant.get("handle_friction") is not None:
        model = drawer.mj_model
        for geom in drawer.geom_ids:
            if model.geom(geom).name.startswith(f"{variant['drawer']}/handle_"):
                model.geom(geom).friction[0] = variant["handle_friction"]


def repulls() -> int:
    return int(os.environ.get("HELP_REPULLS") or 0)


def install_faults(variant: dict | None) -> None:
    """``$HELP_FAULT``'s faults applied from the start of each planned pull, and ``$HELP_REPULLS``'s re-pulls (the
    module docstring)."""
    f, n_repulls = fault(), repulls()
    if not f and not n_repulls:
        return
    if f.get("hidden_drawer") and variant is None:
        raise SystemExit("[help!] HELP_FAULT hidden_drawer needs a drawer variant (scripts/collect_drawer_open.sh ... <variant>)")
    from molmo_spaces.policy.solvers.object_manipulation import kino_open_close_planner_policy as kino

    original = kino._LazyKinoOpening._build
    revealed = {}       # model address -> what the fault changed this episode, undone by the policy's resets

    def _build(self) -> None:
        policy = self.policy
        pull = getattr(policy, "_help_pulls", 0)        # pulls started before this one in the episode
        policy._help_pulls = pull + 1
        base = policy.task.env.current_robot.robot_view.get_qpos_dict().get("base")
        policy._help_pull_base = None if base is None else np.array(base, float)   # where a re-pull returns to
        stop = f.get("stop_fraction") if pull < f.get("stop_pulls", float("inf")) else None
        if stop is not None:
            replay = policy._replay_sequence

            def truncated(targets, **kwargs):
                if len(targets) > 1:
                    targets = targets[: max(1, round(len(targets) * stop))]
                return replay(targets, **kwargs)

            policy._replay_sequence = truncated
        try:
            original(self)
        finally:
            if stop is not None:
                del policy._replay_sequence
        if not f:
            return
        model = policy.task.env.current_model
        undo = revealed.setdefault(model._address, {"model": model, "grip": {}})
        if f.get("hidden_drawer"):
            undo["believed"] = reveal(model, variant)   # every pull: the retry's reset hid it for the re-plan
        if f.get("grip_scale") is not None and not undo["grip"]:
            for a in range(model.nu):
                if model.actuator(a).name.endswith("gripper/fingers_actuator"):
                    undo["grip"][a] = model.actuator_forcerange[a].copy()
                    model.actuator_forcerange[a] *= f["grip_scale"]
        print(f"[help!] pull {pull + 1}: fault {json.dumps(f)}{' (cut)' if stop is not None else ''} from its start",
              flush=True)

    original_execute = kino._LazyKinoOpening.execute

    def execute(self) -> bool:
        done = original_execute(self)
        if not (done and n_repulls) or getattr(self, "_help_judged", False):
            return done
        self._help_judged = True
        policy = self.policy
        used = getattr(policy, "_help_repulls", 0)
        if policy.task.judge_success() or used >= n_repulls:
            return done
        # The pull ran to its end and the drawer is not open: release and back off as xemb's slip re-pull does; the
        # policy's failure check (``_slip_failed`` once the chain ends) then retries from the backed-off hand.
        policy._help_repulls = used + 1
        self._seq = _release_return(kino, policy, getattr(policy, "_help_pull_base", None))
        self._slip_pending = True
        print(f"[help!] pull {getattr(policy, '_help_pulls', 0)} left the drawer closed: release, back off, "
              f"re-pull ({used + 1} of {n_repulls})", flush=True)
        return False

    original_opening_reset = kino._LazyKinoOpening.reset

    def opening_reset(self):
        self._help_judged = False
        return original_opening_reset(self)

    original_reset = kino.KinoOpenClosePlannerPolicy.reset

    def reset(self, reset_retries: bool = True):
        for undo in revealed.values():
            if "believed" in undo:
                conceal(undo["model"], variant, undo.pop("believed"))
        if reset_retries:
            for undo in revealed.values():
                for a, limits in undo["grip"].items():
                    undo["model"].actuator_forcerange[a] = limits
            revealed.clear()
            self._help_pulls = self._help_repulls = 0
        return original_reset(self, reset_retries)

    kino._LazyKinoOpening._build = _build
    kino._LazyKinoOpening.execute = execute
    kino._LazyKinoOpening.reset = opening_reset
    kino.KinoOpenClosePlannerPolicy.reset = reset
    print(f"[help!] HELP_FAULT {json.dumps(f)} HELP_REPULLS {n_repulls}", flush=True)


REPULL_OPEN_S = 1.2          # the jaws open with the arm held
REPULL_BACKOFF_M = 0.08      # then the hand backs off along its approach axis
REPULL_HOME_S = 2.0          # then the arm goes home
REPULL_BASE_SPEED = 0.15     # m/s, then the base drives back to the pull's start


def _release_return(kino, policy, base_goal):
    """A primitive that lets go of the handle and returns for a re-pull, as direct replay setpoints with the gripper
    commanded open on every step (``_KinoReplay(gripper_open=True)``; a replay that names the held base drives it):
    the jaws open with arm and base held, the hand backs off along its approach axis (IK per waypoint), the arm moves
    home in joint space, and the base drives to ``base_goal`` ([x, y, theta], world; None: it stays)."""

    class ReleaseReturn(kino.ActionPrimitive):
        def __init__(self):
            super().__init__(policy.task.env.current_robot.robot_view, duration=1e6)
            self._replay = None

        def _build(self):
            rv = self.robot_view
            dt = policy.config.policy_dt_ms / 1000.0
            arm = {mg: np.asarray(q, float) for mg, q in policy._current_arm_targets().items()}
            qpos = rv.get_qpos_dict()
            base = np.array(qpos["base"], float) if "base" in qpos else None
            held = {} if base is None else {"base": base}
            targets = [{**arm, **held}] * max(1, round(REPULL_OPEN_S / dt))
            gmg = rv.get_gripper_movegroup_ids()[0]
            tcp = rv.get_move_group(gmg).leaf_frame_to_world.copy()
            for i in range(1, 11):
                pose = tcp.copy()
                pose[:3, 3] -= REPULL_BACKOFF_M * i / 10 * tcp[:3, 2]   # against the approach axis
                sol = policy._tcp_to_jp_fn(gmg, pose)
                arm = {mg: np.asarray(sol.get(mg, arm[mg]), float) for mg in arm}
                targets.append({**arm, **held})
            init = policy.config.robot_config.init_qpos or {}
            home = {mg: np.asarray(init[mg], float) for mg in arm if mg in init and len(init[mg]) == len(arm[mg])}
            if len(home) == len(arm):
                n = max(1, round(REPULL_HOME_S / dt))
                targets += [{**{mg: arm[mg] + (home[mg] - arm[mg]) * i / n for mg in arm}, **held}
                            for i in range(1, n + 1)]
                arm = home
            if base is not None and base_goal is not None:
                n = max(1, int(np.ceil(np.linalg.norm(base_goal[:2] - base[:2]) / (REPULL_BASE_SPEED * dt))))
                targets += [{**arm, "base": base + (base_goal - base) * min(1.0, i / n)} for i in range(1, n + 8)]
            self._replay = kino._KinoReplay(rv, targets, dt, gripper_open=True)
            self.duration = self._replay.duration

        def execute(self) -> bool:
            if self.start_time is None:
                self.start_time = self.robot_view.mj_data.time
            if self._replay is None:
                self._build()
            return self._replay.execute()

        def get_current_action(self):
            if self._replay is None:
                self._build()
            return self._replay.get_current_action()

        def get_current_phase(self) -> str:
            return "retreat"

        def check_failure(self) -> bool:
            return False

        def reset(self):
            self._replay = None

    return ReleaseReturn()


def install_tidybot_cameras() -> None:
    from molmo_spaces.configs.camera_configs import MjcfCameraConfig
    from molmo_spaces.tasks import src_kitchen_twin_task_sampler as samplers

    original = samplers.SrcKitchenTwinMapOpenTaskSampler.__init__

    def __init__(self, config) -> None:
        if _has_mast(config.robot_config):
            cameras = config.camera_config.cameras
            wrist = next(c for c in cameras if c.name == "wrist_camera")
            config.camera_config.cameras = [
                wrist.model_copy(update={"fov": None}),
                MjcfCameraConfig(name="exo_camera_1", mjcf_name="mast_camera", robot_namespace="robot_0/", fov=None),
            ]
            print("[help!] exo_camera_1 = the robot's mast_camera; wrist_camera at its MJCF fovy", flush=True)
        original(self, config)

    samplers.SrcKitchenTwinMapOpenTaskSampler.__init__ = __init__


def _has_mast(robot_config) -> bool:
    variant = getattr(robot_config, "variant_dir", None)
    if variant is None or getattr(robot_config, "model_name_prefix", None) != "arm/":
        return False
    model = Path(variant) / "model.xml"
    return model.is_file() and 'name="mast_camera"' in model.read_text()
