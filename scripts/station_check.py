"""M0 check: the DROID at its station in the installed twin.

For each station: robot-vs-kitchen penetration at the home posture, the gripper's height over the island top, and
renders from the station's cameras (ext ZED 2i, D435, wrist) plus an overview, written to --out.

    source env.sh && $PY scripts/station_check.py --out data/station_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from helpsim.envs.kitchen_station import PREFIX, STATIONS, build, robot_scene_contacts


def overview(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.body(f"{PREFIX}fr3_link0").xpos + [0.3, 0.0, 0.1]
    cam.distance, cam.azimuth, cam.elevation = 2.6, 200.0, -30.0
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/station_check"))
    parser.add_argument("--stations", nargs="+", default=list(STATIONS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.stations:
        model, data = build(name)
        contacts = robot_scene_contacts(model, data)
        pad = min(data.xpos[b][2] for b in range(model.nbody) if model.body(b).name.startswith(f"{PREFIX}gripper/"))
        report[name] = {
            "station": vars(STATIONS[name]),
            "home_penetrations": [{"robot": r, "scene": s, "depth_m": round(d, 4)} for r, s, d in contacts],
            "lowest_gripper_body_z_m": round(float(pad), 4),
        }
        renderer = mujoco.Renderer(model, 480, 848)
        for cam in (f"{PREFIX}cam_ext", f"{PREFIX}cam_d435", f"{PREFIX}gripper/wrist_camera"):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                continue
            renderer.update_scene(data, camera=cam)
            Image.fromarray(renderer.render()).save(args.out / f"{name}__{cam.split('/')[-1]}.png")
        Image.fromarray(overview(model, data, renderer)).save(args.out / f"{name}__overview.png")
        renderer.close()
        print(name, json.dumps(report[name]["home_penetrations"]), "lowest gripper z", report[name]["lowest_gripper_body_z_m"])
    (args.out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
