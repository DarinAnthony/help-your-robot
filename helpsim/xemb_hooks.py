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
- ``"stop_fraction": f``: the pull is cut to its first f of the planned steps (the base stops early).

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


def install_faults(variant: dict | None) -> None:
    """``$HELP_FAULT``'s faults applied from the start of the planned pull (the module docstring)."""
    f = fault()
    if not f:
        return
    if f.get("hidden_drawer") and variant is None:
        raise SystemExit("[help!] HELP_FAULT hidden_drawer needs a drawer variant (scripts/collect_drawer_open.sh ... <variant>)")
    from molmo_spaces.policy.solvers.object_manipulation import kino_open_close_planner_policy as kino

    original = kino._LazyKinoOpening._build
    revealed = {}       # model address -> what the fault changed this episode, undone by the policy's next reset

    def _build(self) -> None:
        policy = self.policy
        stop = f.get("stop_fraction")
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
        model = policy.task.env.current_model
        if model._address in revealed:      # a re-pull in the same episode: the fault is already on
            return
        undo = revealed[model._address] = {"model": model, "grip": {}}
        if f.get("hidden_drawer"):
            undo["believed"] = reveal(model, variant)
        if f.get("grip_scale") is not None:
            for a in range(model.nu):
                if model.actuator(a).name.endswith("gripper/fingers_actuator"):
                    undo["grip"][a] = model.actuator_forcerange[a].copy()
                    model.actuator_forcerange[a] *= f["grip_scale"]
        print(f"[help!] fault {json.dumps(f)} from the pull's start", flush=True)

    original_reset = kino.KinoOpenClosePlannerPolicy.reset

    def reset(self, reset_retries: bool = True):
        if reset_retries:
            for undo in revealed.values():
                if "believed" in undo:
                    conceal(undo["model"], variant, undo["believed"])
                for a, limits in undo["grip"].items():
                    undo["model"].actuator_forcerange[a] = limits
            revealed.clear()
        return original_reset(self, reset_retries)

    kino._LazyKinoOpening._build = _build
    kino.KinoOpenClosePlannerPolicy.reset = reset
    print(f"[help!] HELP_FAULT {json.dumps(f)}", flush=True)


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
