"""pi0.5 shelf rollouts in sim (helpsim/pi05_shelf.py): episodes, evaluation trees, and a replay test of the loop.

    source env.sh
    # the attempt monitor against the expert's labelled attempts (no model needed)
    $PY scripts/pi05_rollouts.py replay-test --src data/shelf_pnp/20261114 --per-object 3
    # pi0.5 behind the notebook's tunnel
    $PY scripts/pi05_rollouts.py episodes --policy wss://<name>.trycloudflare.com --out data/shelf_pnp/pi05_eval/run1 \\
        --objects tomato egg orange --episodes 20 --decide reset --samples 5
    $PY scripts/pi05_rollouts.py trees --policy wss://... --out data/shelf_pnp/pi05_trees/run1 --objects tomato --episodes 10

``episodes`` writes ``<out>/<object>/episodes.jsonl`` (one line per episode: seed, start, target, every attempt's outcome
and timing, every decision with pi0.5's spread). ``trees`` writes ``<out>/<object>/trees.jsonl`` (one line per start,
every node) and ``tree_<seed>.npz`` (each node's decision frames, ``<path or root>/<camera>``). Both skip seeds already
written, so a run can be restarted. Seeds are ``--seed0 + 1000 * object index + i`` (the object's index in
``collect_shelf_pnp.SPLITS``, across splits), so a start is the same across runs and decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_shelf_pnp import MODES, SPLITS  # noqa: E402

OBJECTS = [o for objs in SPLITS.values() for o in objs]


def _json_default(x):
    """numpy scalars and arrays in a record (a tree node's numpy bool once ended a run)."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


def seeds_for(obj: str, seed0: int, n: int) -> list[int]:
    k = OBJECTS.index(obj)
    return list(range(seed0 + 1000 * k, seed0 + 1000 * k + n))


def done_seeds(log: Path) -> set[int]:
    return {json.loads(line)["seed"] for line in log.read_text().splitlines() if line} if log.is_file() else set()


def replay_test(a) -> None:
    """Replays expert episodes with continue decisions and compares the monitor's attempt outcomes with the expert's."""
    from helpsim.pi05_shelf import ReplayPolicy, ShelfRollout

    pairs, first, rows = Counter(), Counter(), []
    for mode in a.modes:
        for odir in sorted((a.src / mode / "train").iterdir()):
            log = odir / "episodes.jsonl"
            if not log.is_file():
                continue
            eps = [json.loads(line) for line in log.read_text().splitlines() if line][: a.per_object]
            if not eps:
                continue
            ro = ShelfRollout(odir.name, render=False, **MODES[mode])
            for r in eps:
                pol = ReplayPolicy(odir / f"ep_{r['seed']:05d}.npz", ro.env)
                t = time.perf_counter()
                res = ro.run_episode(pol, np.random.default_rng(r["seed"]), max_attempts=len(r["attempts"]))
                start_ok = np.allclose(ro.env.start_obj[0], r["start_obj_pos"], atol=1e-6)
                exp = [x["outcome"] for x in r["attempts"]]
                got = [x["outcome"] for x in res["attempts"]]
                for e, g in zip(exp, got):
                    pairs[(e, g)] += 1
                first[(exp[0], got[0])] += 1
                rows.append({"mode": mode, "object": odir.name, "seed": r["seed"], "start_ok": bool(start_ok),
                             "expert": exp, "monitor": got, "ticks": [x["ticks"] for x in res["attempts"]],
                             "wall_s": round(time.perf_counter() - t, 1)})
                print(json.dumps(rows[-1]), flush=True)
    # First attempts start from the same state in both; later ones drift apart where the monitor ends an attempt at a
    # different tick than the expert did (and the expert's "unreachable" has no pi0.5 counterpart).
    for title, c in (("first attempts", first), ("all attempts, in order", pairs)):
        print(f"\nexpert outcome -> monitor outcome ({title}):")
        for (e, g), n in sorted(c.items()):
            print(f"  {e:12s} -> {g:12s} {n}")
    if a.json:
        Path(a.json).write_text(json.dumps({"first": {f"{e}->{g}": n for (e, g), n in first.items()},
                                          "pairs": {f"{e}->{g}": n for (e, g), n in pairs.items()}, "episodes": rows}, indent=1))


def collect(a, trees: bool) -> None:
    from helpsim.pi05_shelf import ShelfRollout, connect

    policy = connect(a.policy)
    meta = policy.get_server_metadata()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "server": meta,
                                                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    decide = {"continue": None,
              "reset": lambda k, prev, dp: "reset"}[a.decide]
    for obj in a.objects:
        ro = ShelfRollout(obj, exec_steps=a.exec, **MODES[a.mode])
        odir = a.out / obj
        odir.mkdir(exist_ok=True)
        log = odir / ("trees.jsonl" if trees else "episodes.jsonl")
        done = done_seeds(log)
        n_ok = 0
        for seed in seeds_for(obj, a.seed0, a.episodes):
            if seed in done:
                continue
            t = time.perf_counter()
            if trees:
                nodes, frames = ro.run_tree(policy, seed, a.max_attempts, a.samples)
                np.savez_compressed(odir / f"tree_{seed:05d}.npz", **{f"{p or 'root'}/{cam}": img
                                                                     for p, cams in frames.items() for cam, img in cams.items()})
                rec = {"nodes": nodes, "start_obj_pos": ro.env.start_obj[0].tolist(), "target": ro.env.target.tolist()}
                ok = any(n["outcome"] == "success" for n in nodes)
            else:
                rec = ro.run_episode(policy, np.random.default_rng(seed), decide, a.max_attempts, a.samples)
                ok = rec["success"]
            n_ok += ok
            with open(log, "a") as f:
                f.write(json.dumps({"seed": seed, "object": obj, "mode": a.mode, "faults": ro.env.faults,
                                    "server": meta, "wall_s": round(time.perf_counter() - t, 1), **rec},
                                   default=_json_default) + "\n")
            summary = ([n["outcome"] for n in rec["nodes"][1:]] if trees else
                       [x["outcome"] for x in rec["attempts"]])
            print(f"{obj:12s} {seed}  {summary}  ({time.perf_counter() - t:.0f}s)", flush=True)
        print(f"== {obj}: {n_ok} with a success", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode_", choices=["replay-test", "episodes", "trees"], metavar="{replay-test,episodes,trees}")
    p.add_argument("--src", type=Path, default=Path("data/shelf_pnp/20261114"), help="replay-test: expert run")
    p.add_argument("--modes", nargs="+", default=["nominal", "weak_grip", "pose_error"], help="replay-test")
    p.add_argument("--per-object", type=int, default=3, help="replay-test: episodes per object and mode")
    p.add_argument("--json", default=None, help="replay-test: write the results here")
    p.add_argument("--policy", default=None, help="policy server, e.g. wss://<name>.trycloudflare.com")
    p.add_argument("--out", type=Path)
    p.add_argument("--objects", nargs="+", default=["tomato"], choices=OBJECTS)
    p.add_argument("--episodes", type=int, default=10, help="starts per object")
    p.add_argument("--seed0", type=int, default=90000)
    p.add_argument("--mode", default="nominal", choices=list(MODES), help="injected fault (weak_grip for the shelf)")
    p.add_argument("--decide", default="reset", choices=["continue", "reset"], help="episodes: after a failure")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--samples", type=int, default=0, help="sampled chunks per decision point for pi0.5's spread")
    p.add_argument("--exec", type=int, default=8, help="steps of each chunk executed")
    a = p.parse_args()
    if a.mode_ == "replay-test":
        replay_test(a)
    else:
        if not a.policy or not a.out:
            p.error("--policy and --out are needed")
        collect(a, trees=a.mode_ == "trees")


if __name__ == "__main__":
    main()
