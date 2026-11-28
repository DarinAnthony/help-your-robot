"""Example-episode figure for the report: one test start, played by Ours (belief POMDP) and by B6 (V_k threshold).

    source env.sh
    $PY scripts/example_episode.py --trees data/shelf_pnp/20261114_tree --list          # candidate starts
    $PY scripts/example_episode.py --trees data/shelf_pnp/20261114_tree --out reports/help_eval/expert \
        --key expert/weak_grip/sponge/26003 --key expert/nominal/metal_spoon/30001

Models are fit exactly as in ``scripts/help_eval.py`` (same folds, train fit, cal tuning, default costs). For each
decision point on Ours's path the figure shows the scene camera, Ours's belief over {easy, recoverable, infeasible},
its Q values and choice, and what B6 would do at the same point (V_k against its tuned threshold).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpsim.help import policies as P  # noqa: E402
from helpsim.help.trees import DIFFICULTIES, FAMILIES, Costs, assign_folds, best_value, load_trees, rollout, view  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from help_eval import DEFAULT, INK, INK2, GRID, SERIES, SURFACE, TAUS, tune  # noqa: E402

ACTION_NAME = {"A": "attempt", "D": "adjust", "ask": "ask"}


def fit(trees, c: Costs):
    train = [t for t in trees if t.fold == "train"]
    cal = [t for t in trees if t.fold == "cal"]
    V = P.ValueModel().fit(train, cal)
    head = P.DifficultyHead().fit(train)
    obs = P.ObservationModel(FAMILIES["expert"]).fit(train)
    tau6, _ = tune(cal, lambda g: P.v_threshold(V, g), TAUS, c)
    return V, P.BeliefPlanner(obs, head, c), P.v_threshold(V, tau6), float(tau6)


def trace(tree, ours, V, tau, c):
    """Ours's episode, with its belief, Q values and B6's view at every decision point on that path."""
    steps, path = [], ""
    res = rollout(tree, ours, c)
    for i in range(len(res["path"]) + 1):
        path = res["path"][:i]
        v = view(tree, path)
        if i == len(res["path"]) and not res["voluntary"]:
            steps.append({"path": path, "terminal": tree.nodes[path]["outcome"] if path else None})
            break
        b = ours.belief(v)
        q = {a: x for a, x in ours.q(b, v["k"]).items() if a == "ask" or a in v["actions"]}
        vk = V(v)
        steps.append({"path": path, "belief": b, "q": q, "choice": max(q, key=q.get), "vk": vk,
                      "b6": "ask" if vk < tau else P.default_action(v),
                      "outcome": tree.nodes[path]["outcome"] if path else None})
    return res, steps


def frame(tree, path):
    d = np.load(tree.frames)
    return d[f"{path or 'root'}/cam_cabinet"]


def draw(examples, tau, c, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = max(len(s) for _, _, _, s in examples)
    fig = plt.figure(figsize=(3.3 * ncol, 5.2 * len(examples)), facecolor=SURFACE)
    subs = np.atleast_1d(fig.subfigures(len(examples), 1, hspace=0.06))
    for r, (tree, res, b6res, steps) in enumerate(examples):
        sf = subs[r]
        sf.set_facecolor(SURFACE)
        sf.suptitle(f"{chr(97 + r)}) {tree.object.replace('_', ' ')}, {tree.condition.replace('_', ' ')} start "
                    f"(hidden difficulty: {tree.difficulty})", fontsize=11, color=INK, weight="bold", x=0.01,
                    ha="left")
        gs = sf.add_gridspec(2, ncol, height_ratios=[1.2, 1], hspace=0.75, wspace=0.3, top=0.86, bottom=0.12)
        for j, s in enumerate(steps):
            ax = sf.add_subplot(gs[0, j])
            ax.imshow(frame(tree, s["path"]))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if s["path"]:
                last = s["path"][-1]
                hist = f"after {ACTION_NAME[last]} #{len(s['path'])}: " + \
                       tree.nodes[s["path"]]["outcome"].replace("_", " ")
            else:
                hist = "start"
            ax.set_title(f"decision {len(s['path']) + 1}" if "terminal" not in s else "end", fontsize=9,
                         color=INK, loc="left", weight="bold", pad=14)
            ax.text(0, 1.02, hist, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
            if "terminal" in s:
                ax.set_xlabel(f"episode ends: {s['terminal'].replace('_', ' ')}\nOurs reward {res['reward']:+.1f}  "
                              f"(B6 {b6res['reward']:+.1f}, best possible {best_value(tree, c):+.1f})",
                              fontsize=8, color=INK)
                continue
            bx = sf.add_subplot(gs[1, j])
            bx.barh(range(3), s["belief"], color=SERIES[0], height=0.6)
            bx.set_yticks(range(3), DIFFICULTIES, fontsize=8, color=INK2)
            bx.invert_yaxis()
            bx.set_xlim(0, 1)
            bx.tick_params(axis="x", labelsize=7, colors=INK2)
            for sp in ("top", "right"):
                bx.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                bx.spines[sp].set_color(GRID)
            qs = "  ".join(f"{ACTION_NAME[a]} {x:+.2f}" for a, x in s["q"].items())
            bx.set_title(f"Ours belief → {ACTION_NAME[s['choice']].upper()}\nQ: {qs}", fontsize=8, color=INK, loc="left")
            bx.set_xlabel(f"B6: V_k = {s['vk']:.2f} vs τ = {tau:.2f} → {ACTION_NAME[s['b6']]}", fontsize=8,
                          color=SERIES[1] if s["b6"] != s["choice"] else INK2)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trees", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--key", action="append", default=[], help="tree key(s) to draw, one row each")
    p.add_argument("--list", action="store_true", help="list test starts where Ours and B6 differ")
    a = p.parse_args()
    trees = load_trees(a.trees, "expert")
    assign_folds(trees, seed=0)
    c = Costs(*DEFAULT)
    V, ours, b6, tau = fit(trees, c)
    test = [t for t in trees if t.fold == "test"]
    if a.list:
        for t in test:
            ro, rb = rollout(t, ours, c), rollout(t, b6, c)
            if ro["reward"] != rb["reward"]:
                print(f"{t.key:45s} {t.difficulty:12s} Ours {ro['path'] or '-':4s}{'+ask' if ro['voluntary'] else '':5s}"
                      f"{ro['reward']:+6.1f}  B6 {rb['path'] or '-':4s}{'+ask' if rb['voluntary'] else '':5s}{rb['reward']:+6.1f}")
        return
    by_key = {t.key: t for t in test}
    examples = []
    for k in a.key:
        t = by_key[k]
        res, steps = trace(t, ours, V, tau, c)
        examples.append((t, res, rollout(t, b6, c), steps))
    a.out.mkdir(parents=True, exist_ok=True)
    draw(examples, tau, c, a.out / "example_episode.png")


if __name__ == "__main__":
    main()
