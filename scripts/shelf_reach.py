"""Which shelf cells of a cabinet the DROID reaches from a station, by IK with a horizontal approach.

Opens the cabinet's doors, then for a grid of points above each shelf solves damped least squares for the grasp site at
the point with the gripper's approach axis level and pointing into the cabinet, from several seeds; a cell counts as
reached when the solution is within 1 cm and 10 deg, inside the joint ranges and free of robot-scene penetration.

    source env.sh && $PY scripts/shelf_reach.py --station droid_corner_20261029 --cabinet upper_return_0_1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from helpsim.envs.kitchen_station import HOME, PREFIX, build, robot_scene_contacts, set_arm

ARM = [f"{PREFIX}fr3_joint{i}" for i in range(1, 8)]


def open_doors(model, data, prefix: str) -> list[str]:
    opened = []
    for j in range(model.njnt):
        name = model.joint(j).name
        if name.startswith(prefix) and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lo, hi = model.jnt_range[j]
            data.qpos[model.jnt_qposadr[j]] = hi if abs(hi) > abs(lo) else lo
            opened.append(name)
    mujoco.mj_forward(model, data)
    return opened


def solve(model, data, site: int, target: np.ndarray, approach: np.ndarray, q0, iters: int = 300):
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in ARM]
    dadr = [model.jnt_dofadr[model.joint(n).id] for n in ARM]
    rng = np.array([model.jnt_range[model.joint(n).id] for n in ARM])
    set_arm(model, data, q0)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        z = data.site_xmat[site].reshape(3, 3)[:, 2]
        e = np.concatenate([target - data.site_xpos[site], np.cross(z, approach)])
        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2:
            break
        mujoco.mj_jacSite(model, data, jp, jr, site)
        J = np.vstack([jp[:, dadr], jr[:, dadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
        q = np.clip(data.qpos[qadr] + 0.5 * dq, rng[:, 0], rng[:, 1])
        data.qpos[qadr] = q
    q = data.qpos[qadr].copy()
    set_arm(model, data, q)
    z = data.site_xmat[site].reshape(3, 3)[:, 2]
    pos_err = float(np.linalg.norm(target - data.site_xpos[site]))
    ang_err = float(np.degrees(np.arccos(np.clip(z @ approach, -1, 1))))
    return q, pos_err, ang_err, robot_scene_contacts(model, data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--station", default="droid_corner_20261029")
    p.add_argument("--cabinet", default="upper_return_0_1", help="interior/<cabinet>_<shelf>_shelf_collision geoms")
    p.add_argument("--doors", default="cabinet/upper_return__door_", help="hinge joint prefix to open")
    p.add_argument("--clearance", type=float, default=0.04, help="grasp point height above the shelf top, m")
    p.add_argument("--out", type=Path, default=Path("data/shelf_reach"))
    a = p.parse_args()
    model, data = build(a.station)
    print("opened", open_doors(model, data, a.doors))
    site = model.site(f"{PREFIX}gripper/grasp_site").id
    root = data.body(f"{PREFIX}fr3_link0").xpos.copy()
    seeds = [HOME, (0.6, -0.3, 0.0, -1.6, 0.0, 1.4, 0.8), (0.3, 0.2, 0.0, -1.2, 0.0, 1.6, 0.8)]
    report = {}
    for shelf in ("lower", "upper", "top"):
        g = model.geom(f"interior/{a.cabinet}_{shelf}_shelf_collision").id
        c, R, s = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        depth_axis = int(np.argmin(s[:2]))           # the shelf's short horizontal axis runs into the cabinet
        into = R[:, depth_axis] * np.sign(R[:, depth_axis] @ (c - root))
        along = R[:, 1 - depth_axis]
        cells = []
        for u in np.linspace(-0.8, 0.8, 5):           # along the shelf
            for v in (-0.6, -0.2, 0.2):               # front (-) to back (+) of the shelf depth
                pt = c + u * s[1 - depth_axis] * along + v * s[depth_axis] * into + R[:, 2] * (s[2] + a.clearance)
                best = min((solve(model, data, site, pt, into, q0) for q0 in seeds), key=lambda r: (len(r[3]) > 0, r[1]))
                q, pe, ae, contacts = best
                ok = pe < 0.01 and ae < 10 and not contacts
                cells.append({"u": round(u, 2), "v": round(v, 2), "point": pt.round(3).tolist(), "reached": ok,
                              "pos_err_m": round(pe, 4), "ang_err_deg": round(ae, 1),
                              "contacts": sorted({f"{r}~{s_}" for r, s_, _ in contacts})[:4], "q": q.round(3).tolist()})
        n = sum(c_["reached"] for c_ in cells)
        report[shelf] = {"top_z_m": round(float(c[2] + s[2]), 4), "reached": n, "cells": cells}
        grid = "\n".join("  " + " ".join("#" if next(x for x in cells if x["u"] == round(u, 2) and x["v"] == v)["reached"]
                                         else "." for u in np.linspace(-0.8, 0.8, 5)) for v in (-0.6, -0.2, 0.2))
        print(f"{shelf}: top z {report[shelf]['top_z_m']} m, reached {n}/{len(cells)} (rows front->back, # reached)\n{grid}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.station}__{a.cabinet}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
