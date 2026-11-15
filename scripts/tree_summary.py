"""Summarize the shelf counterfactual action trees (``collect_shelf_pnp.py --tree``) for EXPERIMENT_PLAN.md.

For every tree (one start) this labels the hidden difficulty from outcomes (easy: the first attempt succeeds;
recoverable: some later attempt within the tree's depth succeeds; infeasible: none does), and checks that the root
attempt reproduces the recorded episode of the same seed (``data/shelf_pnp/<date>/<mode>/.../episodes.jsonl``).

    python scripts/tree_summary.py data/shelf_pnp/20261114_tree [--episodes data/shelf_pnp/20261114]

writes ``<tree root>/difficulty.jsonl`` (one line per start: mode, split, object, seed, difficulty, n_attempts,
first_outcome, best path) and prints the difficulty mix per mode and per object.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def label(nodes: list[dict]) -> tuple[str, str | None]:
    """Difficulty and the shortest successful path (attempt-count first, then fewer adjusts)."""
    wins = sorted((n["path"] for n in nodes if n["outcome"] == "success"), key=lambda p: (len(p), p.count("D")))
    if not wins:
        return "infeasible", None
    return ("easy" if wins[0] == "A" else "recoverable"), wins[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--episodes", type=Path, default=None, help="recorded episodes root to check root attempts against")
    a = p.parse_args()
    episodes = {}
    if a.episodes:
        for f in a.episodes.glob("*/*/*/episodes.jsonl"):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                episodes[(r["mode"], r["object"], r["seed"])] = r
    out, mix, per_obj = [], collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    match = mismatch = 0
    for f in sorted(a.root.glob("*/*/*/trees.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            nodes = r["nodes"]
            diff, best = label(nodes)
            first = next(n["outcome"] for n in nodes if n["path"] == "A")
            ep = episodes.get((r["mode"], r["object"], r["seed"]))
            if ep is not None:
                same = ep["attempts"][0]["outcome"] == first
                match += same
                mismatch += not same
            out.append({"mode": r["mode"], "split": r["split"], "object": r["object"], "seed": r["seed"],
                        "difficulty": diff, "n_attempts": sum(1 for n in nodes if n["depth"] > 0),
                        "first_outcome": first, "best_path": best})
            mix[r["mode"]][diff] += 1
            per_obj[(r["mode"], r["split"], r["object"])][diff] += 1
    (a.root / "difficulty.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    keys = ("easy", "recoverable", "infeasible")
    print(f"{len(out)} trees, {sum(o['n_attempts'] for o in out)} attempts")
    if a.episodes:
        print(f"root attempt matches the recorded episode: {match}/{match + mismatch}")
    print("\nmode         " + "  ".join(f"{k:>11}" for k in keys))
    for mode, c in sorted(mix.items()):
        print(f"{mode:12} " + "  ".join(f"{c[k]:>11}" for k in keys))
    print("\nmode         split                    object       easy/recov/infeas")
    for (mode, split, obj), c in sorted(per_obj.items()):
        print(f"{mode:12} {split:24} {obj:12} {c['easy']:>3} {c['recoverable']:>3} {c['infeasible']:>3}")


if __name__ == "__main__":
    main()
