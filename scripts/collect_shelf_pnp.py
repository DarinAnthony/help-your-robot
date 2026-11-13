"""Collect shelf pick-and-place episodes (helpsim.tasks.shelf_pnp) with the station DROID, one process per object.

Output, per object: ``<out>/<split>/<object>/ep_<seed>.npz`` (per-tick state and actions: see ShelfPnP._record),
``ep_<seed>_<camera>.mp4`` (exterior ZED 2i and wrist at 5 Hz; ``frame_tick`` in the npz maps frames to ticks), and ``episodes.jsonl`` (one line per episode: seed,
start and target, success, final outcome, every attempt's outcome, failure phase and end tick). ``<out>/run.json``
records the split, objects, code versions and station.

    source env.sh && $PY scripts/collect_shelf_pnp.py --out data/shelf_pnp/20261113 --episodes 30
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


def collect_object(job: tuple) -> dict:
    split, obj, seeds, out, render = job
    import imageio.v2 as imageio
    import numpy as np

    from helpsim.tasks.shelf_pnp import ShelfPnP

    env = ShelfPnP(obj, render=render)
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
            log.write(json.dumps({"seed": seed, "object": obj, "split": split, "start_obj_pos": start,
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
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs, k = [], 0
    for split in a.splits:
        for obj in SPLITS[split]:
            jobs.append((split, obj, list(range(a.seed0 + 1000 * k, a.seed0 + 1000 * k + a.episodes)), str(a.out),
                         not a.no_render))
            k += 1
    (a.out / "run.json").write_text(json.dumps({
        "task": "shelf_pnp", "station": "droid_corner_20261029", "shelf": "upper_return_0_1 upper (z 1.5065)",
        "splits": {s: SPLITS[s] for s in a.splits}, "episodes_per_object": a.episodes, "seed0": a.seed0,
        "help_commit": git_sha(str(Path(__file__).resolve().parents[1])), "xemb_commit": git_sha(os.environ["XEMB"]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1))
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(collect_object, jobs):
            print(f"{r['split']:24s} {r['object']:12s} {r['successes']}/{r['episodes']}  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
