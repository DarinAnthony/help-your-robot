"""Hooks on xemb's datagen pipeline, installed by ``scripts/run_pipeline_hooked.py``.

Drawer variants. xemb's arena ``Drawer`` (``molmo_spaces/env/arena/drawer.py``), built for every part the scene
metadata files as a Drawer, overwrites each episode what the scene file says: the slide's frictionloss U(1, 2) N, damping U(0.01, 0.1),
armature U(0, 0.1) and every geom's friction U(0.1, 0.3). So the twin's own drawer values (frictionloss 0.4 N,
damping 4) never reach the simulation, and neither would a variant's. With ``$SRC_KITCHEN_TWIN_DIR/variant.json``
(``scripts/make_drawer_variants.py``) present, the hook applies the variant's values to its drawer after that
randomization; the variant's tendons (the latch) are not touched by it. Without one it does nothing.

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
    install_drawer_variant()
    install_tidybot_cameras()


def install_drawer_variant() -> dict | None:
    path = Path(os.environ.get("SRC_KITCHEN_TWIN_DIR", "")) / "variant.json"
    if not path.is_file():
        return None
    variant = json.loads(path.read_text())
    from molmo_spaces.env.arena import drawer as arena_drawer

    original = arena_drawer.Drawer.__init__

    def __init__(self, object_name, data, body_name2id) -> None:
        original(self, object_name, data, body_name2id)
        if object_name == variant["drawer"]:
            apply(self, variant)

    arena_drawer.Drawer.__init__ = __init__
    print(f"[help!] drawer variant {variant['name']} ({path.parent}) pinned on {variant['drawer']}", flush=True)
    return variant


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
