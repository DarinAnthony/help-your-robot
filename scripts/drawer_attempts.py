"""Per-pull attempt records for TidyBot drawer runs (scripts/collect_drawer_open.sh), the drawer's ``episodes.jsonl``.

Each policy step's ``obs/extra/measured_streams`` blob holds the 250 Hz rows recorded since the previous step (every
scalar joint of the scene, then the controller's report), and ``obs/extra/policy_num_retries`` numbers the step's
attempt (a re-pull is a retry, helpsim/xemb_hooks.py). So each pull is a run of steps with the same retry count, and
for it this records the drawer's travel (max and at the pull's end), the base's travel, whether the task judged the
episode a success during it, and its steps. A pull's ``outcome`` is ``success`` or ``stuck`` (the drawer travelled less
than 5 cm) or ``short`` (it moved but did not open).

pi0.5 runs (``scripts/run_drawer_pi05.sh``) have no retry count: their attempts are the policy's own monitor's
(``helpsim/pi05_drawer.py``), saved in each trajectory's ``obs_scene`` as ``help_attempts`` (the same fields, plus the
monitor's ``event``: success, grasp_miss, hold_lost, stall, timeout or truncated) and ``help_decisions``; those are
used as recorded.

    source env.sh && $PY scripts/drawer_attempts.py tidybot_open_hidden_hard_20261114 [more runs ...]

writes ``<run>/attempts.jsonl`` (one line per episode) and prints a per-run summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np

DRAWER = "drawer/glass_counter__stack_b_3_front/slide"
BASE = ("robot_0/joint_x", "robot_0/joint_y")
STUCK_M = 0.05


def pi05_episode(traj) -> dict | None:
    if "obs_scene" not in traj:
        return None
    scene = json.loads(bytes(traj["obs_scene"][()]).decode())
    if "help_attempts" not in scene:
        return None
    attempts = scene["help_attempts"]
    return {"success": any(a["outcome"] == "success" for a in attempts), "n_steps": int(len(traj["success"])),
            "attempts": attempts, "decisions": scene.get("help_decisions", []), "decide": scene.get("help_decide")}


def episode(traj) -> dict:
    pi05 = pi05_episode(traj)
    if pi05 is not None:
        return pi05
    from molmo_spaces.robots.measured.stream_recorder import decode

    retries = traj["obs/extra/policy_num_retries"][:]
    success = traj["success"][:]
    names, per_step = None, []
    for blob in traj["obs/extra/measured_streams"][:]:
        n, _t, rows, _e, _m = decode(blob)
        names = n or names
        per_step.append(rows if len(rows) else None)
    di = names.index(DRAWER)
    bi = [names.index(b) for b in BASE]
    pulls = []
    for k in np.unique(retries):
        steps = np.flatnonzero(retries == k)
        rows = [per_step[s] for s in steps if per_step[s] is not None]
        if not rows:
            continue
        rows = np.concatenate(rows)
        travel = float(rows[:, di].max())
        base = rows[:, bi]
        ok = bool(success[steps].any())
        pulls.append({"attempt": int(k), "outcome": "success" if ok else ("stuck" if travel < STUCK_M else "short"),
                      "drawer_max_m": round(travel, 4), "drawer_end_m": round(float(rows[-1, di]), 4),
                      "base_travel_m": round(float(np.linalg.norm(base.max(0) - base.min(0))), 4),
                      "first_step": int(steps[0]), "last_step": int(steps[-1])})
    return {"success": bool(success.any()), "n_steps": int(len(success)), "attempts": pulls}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    root = Path(os.environ["MLSPACES_ARTIFACT_ROOT"]) / "collect/datagen/open_planner_v1"
    for run in a.runs:
        rdir = root / run
        info = json.loads((rdir / "help_run.json").read_text()) if (rdir / "help_run.json").is_file() else {}
        lines = []
        for f in sorted(glob.glob(str(rdir / "house_0" / "*.h5"))):
            with h5py.File(f) as h:
                for t in sorted(h, key=lambda name: int(name.split("_")[-1])):
                    rec = {"run": run, "file": Path(f).name, "traj": t, "variant": info.get("variant"),
                           "fault": info.get("fault"), "repulls": info.get("repulls", 0), **episode(h[t])}
                    lines.append(rec)
        (rdir / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
        n_ok = sum(r["success"] for r in lines)
        pulls = [len(r["attempts"]) for r in lines]
        first = [r["attempts"][0]["drawer_max_m"] for r in lines if r["attempts"]]
        print(f"{run}: {n_ok}/{len(lines)} opened, pulls per episode {np.bincount(pulls).tolist() if pulls else []}, "
              f"first-pull travel median {np.median(first) if first else float('nan'):.3f} m")


if __name__ == "__main__":
    main()
