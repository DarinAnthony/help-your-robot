"""Hard evaluation variants of the TidyBot's demo drawer, as copies of the installed twin.

Each variant is ``$MLSPACES_ARTIFACT_ROOT/src_kitchen_twin_drawer_<name>/``: the installed twin
(``$SRC_KITCHEN_TWIN_DIR``) with every entry symlinked except ``scene.xml``, where only the drawer's slide changes.
Point ``SRC_KITCHEN_TWIN_DIR`` at it and run the same collection command; nothing in xemb changes.

- ``hard``: the slide's Coulomb friction (``frictionloss``) at 6.5 N, where the identified TidyBot opened 3 of 6
  in calibration (README);
- ``locked``: a hidden latch, a limited fixed tendon on the slide that stops it 8 mm out. The joint's range stays
  0-0.5 m because the opening task scores the slide as a fraction of that range (a clamped range would score a
  locked drawer as fully open).

xemb's arena ``Drawer`` overwrites a drawer's slide friction, damping and armature and its geoms' friction every
episode (frictionloss U(1, 2) N), so the scene file's values (nominal: frictionloss 0.4 N, damping 4 N·s/m) reach
the simulation only through ``helpsim/xemb_hooks.py``: run the pipeline with ``scripts/run_pipeline_hooked.py``
(``scripts/collect_drawer_open.sh`` does). Tendons (the latch, the soft-close ``.../slide/closing{,_cap}``) are not
overwritten. Each build pushes the drawer alone along its slide with fixed forces, with the scene file's values,
and records how far it moves in 1 s (``variant.json``).

    source env.sh && $PY scripts/make_drawer_variants.py hard locked
    $PY scripts/make_drawer_variants.py --custom sticky_8 --frictionloss 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from xml.etree import ElementTree

import mujoco

DRAWER = "drawer/glass_counter__stack_b_3_front"
SLIDE = f"{DRAWER}/slide"
LATCH = f"{SLIDE}/latch"
VARIANTS = {
    "hard": {"frictionloss": 6.5, "damping": None, "stop_m": None},  # README: 3 of 6 opened in calibration
    "locked": {"frictionloss": None, "damping": None, "stop_m": 0.008},
}


def build(name: str, frictionloss: float | None, damping: float | None, stop_m: float | None,
          handle_friction: float | None = None) -> Path:
    src = Path(os.environ["SRC_KITCHEN_TWIN_DIR"])
    out = Path(os.environ["MLSPACES_ARTIFACT_ROOT"]) / f"src_kitchen_twin_drawer_{name}"
    out.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        link = out / entry.name
        if entry.name in ("scene.xml", "variant.json"):
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry)
    tree = ElementTree.parse(src / "scene.xml")
    root = tree.getroot()
    joint = next(j for j in root.iter("joint") if j.get("name") == SLIDE)
    nominal = {k: joint.get(k) for k in ("frictionloss", "damping", "range")}
    if frictionloss is not None:
        joint.set("frictionloss", f"{frictionloss:g}")
    if damping is not None:
        joint.set("damping", f"{damping:g}")
    if handle_friction is not None:
        for geom in root.iter("geom"):
            if geom.get("name", "").startswith(f"{DRAWER}/handle_"):
                geom.set("friction", f"{handle_friction:g}")
                geom.set("priority", "1")  # else MuJoCo takes the larger of the pair's frictions (the pads')
    if stop_m is not None:
        tendon = root.find("tendon")
        latch = ElementTree.SubElement(tendon, "fixed", name=LATCH, limited="true", range=f"0 {stop_m:g}")
        ElementTree.SubElement(latch, "joint", joint=SLIDE, coef="1")
    tree.write(out / "scene.xml")
    record = {"name": name, "source_twin": str(src.resolve()), "drawer": DRAWER, "nominal": nominal,
              "frictionloss": frictionloss, "damping": damping, "stop_m": stop_m, "handle_friction": handle_friction,
              "push_test": push_test(out / "scene.xml")}
    (out / "variant.json").write_text(json.dumps(record, indent=1))
    return out


def push_test(scene: Path, forces=(10, 20, 30, 40, 60, 80, 120, 200), hold_s: float = 1.0) -> dict:
    """How far (m) the drawer slides in hold_s under each pull force (N) along its opening direction, from closed."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    jid = model.joint(SLIDE).id
    qadr, dadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    out = {}
    for f in forces:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for _ in range(int(hold_s / model.opt.timestep)):
            data.qfrc_applied[:] = 0
            data.qfrc_applied[dadr] = f  # generalized force on the slide = pull along its axis
            mujoco.mj_step(model, data)
        out[str(f)] = round(float(data.qpos[qadr]), 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", help=f"of {list(VARIANTS)}; none prints the nominal drawer's push test")
    p.add_argument("--custom", help="build one variant under this name from the flags below")
    p.add_argument("--frictionloss", type=float)
    p.add_argument("--damping", type=float)
    p.add_argument("--stop-m", type=float)
    p.add_argument("--handle-friction", type=float, help="sliding friction of the handle's collision geoms (nominal 1.2)")
    a = p.parse_args()
    jobs = [(n, VARIANTS[n]) for n in a.names]
    if a.custom:
        jobs.append((a.custom, {"frictionloss": a.frictionloss, "damping": a.damping, "stop_m": a.stop_m,
                              "handle_friction": a.handle_friction}))
    for name, v in jobs:
        out = build(name, **v)
        rec = json.loads((out / "variant.json").read_text())
        print(f"{name:12s} {out}\n  slide travel in 1 s by pull force (N: m): {rec['push_test']}")
    if not jobs:
        nominal = Path(os.environ["SRC_KITCHEN_TWIN_DIR"]) / "scene.xml"
        print(f"nominal: {push_test(nominal)}")


if __name__ == "__main__":
    main()
