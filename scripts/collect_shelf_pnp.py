"""Collect shelf pick-and-place episodes (helpsim.tasks.shelf_pnp) with the station DROID, one process per object.

Output, per object: ``<out>/<split>/<object>/ep_<seed>.npz`` (per-tick state and actions: see ShelfPnP._record),
``ep_<seed>_<camera>.mp4`` (exterior ZED 2i and wrist at 5 Hz; ``frame_tick`` in the npz maps frames to ticks), and ``episodes.jsonl`` (one line per episode: seed,
start and target, success, final outcome, every attempt's outcome, failure phase and end tick). ``<out>/run.json``
records the split, objects, code versions and station.

    source env.sh && $PY scripts/collect_shelf_pnp.py --out data/shelf_pnp/20261113 --episodes 30

``--mode`` runs the expert with one of ``MODES``, the common failure causes (``shelf_pnp.FAULTS``), on every episode;
``run.json`` and every ``episodes.jsonl`` line record the mode and its faults.

``--tree`` collects counterfactual action trees instead (``ShelfPnP.run_tree``, EXPERIMENT_PLAN.md §4): from the same
start states as the episodes with the same ``--seed0``/``--mode``, every attempt / adjust sequence of up to 3 attempts.
Per object: ``trees.jsonl`` (one line per start: seed, start, target, and every node's path, outcome, failure phase and
decision-point state) and ``tree_<seed>.npz`` (each node's decision-point frames, keyed ``<path>/<camera>``, the root
``root/<camera>``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

SPLITS = {
    "train": ["tomato", "egg", "drumstick", "wooden_ball", "banana", "waffle", "sponge", "duster", "tape"],
    "heldout_in_distribution": ["orange"],
    "heldout_hard": ["metal_spoon"],
}
MODES = {
    "nominal": {},
    "pose_error": {"pose_noise_m": 0.015},     # missed, edge and knocked grasps
    "place_error": {"place_noise_m": 0.04},    # misplacements
    "weak_grip": {"grip_scale": 0.05},         # missed grasps and slips under load
}


def collect_tree(job: tuple) -> dict:
    split, obj, seeds, out, render, mode, _ = job
    import numpy as np

    from helpsim.tasks.shelf_pnp import ShelfPnP

    env = ShelfPnP(obj, render=render, render_every=10 ** 9, **MODES[mode])   # frames only at decision points
    odir = Path(out) / split / obj
    odir.mkdir(parents=True, exist_ok=True)
    n_ok = n_attempts = 0
    with open(odir / "trees.jsonl", "a") as log:
        for seed in seeds:
            env.reset(np.random.default_rng(seed))
            start = env.start_obj[0].tolist()
            nodes, frames = env.run_tree(seed)
            np.savez_compressed(odir / f"tree_{seed:05d}.npz", **{f"{path or 'root'}/{cam}": img
                                                                 for path, cams in frames.items() for cam, img in cams.items()})
            n_ok += any(n["outcome"] == "success" for n in nodes)
            n_attempts += len(nodes) - 1
            log.write(json.dumps({"seed": seed, "object": obj, "split": split, "mode": mode, "faults": env.faults,
                                  "start_obj_pos": start, "target": env.target.tolist(), "nodes": nodes}) + "\n")
            log.flush()
    return {"split": split, "object": obj, "episodes": len(seeds), "successes": n_ok, "attempts": n_attempts}


def collect_object(job: tuple) -> dict:
    split, obj, seeds, out, render, mode, tree = job
    if tree:
        return collect_tree(job)
    import imageio.v2 as imageio
    import numpy as np

    from helpsim.tasks.shelf_pnp import ShelfPnP

    env = ShelfPnP(obj, render=render, **MODES[mode])
    odir = Path(out) / split / obj
    odir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(odir / "episodes.jsonl", "a") as log:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            env.reset(rng)
            start = env.start_obj[0].tolist()
            res = env.run(rng)
            np.savez_compressed(odir / f"ep_{seed:05d}.npz", **env.arrays())
            for cam, frames in env.rec.frames.items():
                imageio.mimwrite(odir / f"ep_{seed:05d}_{cam}.mp4", frames, fps=15 / env.render_every, macro_block_size=1)
            env.rec.frames.clear()
            env.rec.frame_tick.clear()
            n_ok += res["success"]
            log.write(json.dumps({"seed": seed, "object": obj, "split": split, "mode": mode, "faults": env.faults,
                                  "start_obj_pos": start,
                                  "target": env.target.tolist(), "n_ticks": len(env.rec.rows["t"]), **res}) + "\n")
            log.flush()
    return {"split": split, "object": obj, "episodes": len(seeds), "successes": n_ok}


def git_sha(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30, help="per object")
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--mode", default="nominal", choices=list(MODES))
    p.add_argument("--objects", nargs="+", default=None, help="only these objects of the chosen splits")
    p.add_argument("--tree", action="store_true", help="counterfactual action trees from the same starts")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs, k = [], 0
    for split in a.splits:
        for obj in SPLITS[split]:
            if a.objects is None or obj in a.objects:
                jobs.append((split, obj, list(range(a.seed0 + 1000 * k, a.seed0 + 1000 * k + a.episodes)), str(a.out),
                             not a.no_render, a.mode, a.tree))
            k += 1
    (a.out / "run.json").write_text(json.dumps({
        "task": "shelf_pnp_tree" if a.tree else "shelf_pnp", "station": "droid_corner_20261029", "shelf": "upper_return_0_1 upper (z 1.5065)",
        "splits": {s: SPLITS[s] for s in a.splits}, "objects": [j[1] for j in jobs], "mode": a.mode,
        "faults": MODES[a.mode], "episodes_per_object": a.episodes, "seed0": a.seed0,
        "help_commit": git_sha(str(Path(__file__).resolve().parents[1])), "xemb_commit": git_sha(os.environ["XEMB"]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1))
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(collect_object, jobs):
            extra = f"  {r['attempts']} attempts" if "attempts" in r else ""
            print(f"{r['split']:24s} {r['object']:12s} {r['successes']}/{r['episodes']}{extra}  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
