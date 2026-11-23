"""Exact offline evaluation of the help policies on decision trees (EXPERIMENT_PLAN.md §4-6; helpsim/help).

    source env.sh
    $PY scripts/help_eval.py --trees data/shelf_pnp/20261114_tree --family expert --out reports/help_eval/expert
    $PY scripts/help_eval.py --trees data/shelf_pnp/pi05_trees/run1 --family pi05 --out reports/help_eval/pi05

Folds: training objects' starts 60/15/25 train/cal/test by (object, seed); held-out objects test only. Models are fit on
train; Platt scaling and every threshold or k are tuned on cal, separately for each cost setting; everything is scored
on test, paired by start. Writes ``results.json`` (every method x cost: mean reward with a bootstrap CI, success, help
rate, failed attempts, off-shelf rate, voluntary-ask precision and recall against "infeasible", regret to the clairvoyant; paired
differences Ours - each method overall and per condition; the predictors' metrics), ``summary.md`` and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpsim.help import policies as P  # noqa: E402
from helpsim.help.models import calibration_metrics  # noqa: E402
from helpsim.help.trees import (DIFFICULTIES, FAMILIES, Costs, assign_folds, best_value, decision_points,  # noqa: E402
                                default_success, load_trees, rollout)

C_HELP = (1, 2, 3, 5, 8)
C_FAIL = (0.5, 1, 2)
DEFAULT = (3, 0.5)
TAUS = np.linspace(0, 1, 51)
B = 10_000


def group(t) -> str:
    return t.object if t.split != "train" else t.condition


def score(trees, policy, c, reps: int = 1) -> list[dict]:
    out = []
    for t in trees:
        rs = [rollout(t, policy, c) for _ in range(reps)]
        r = {k: float(np.mean([x[k] for x in rs])) for k in ("reward", "success", "asked", "voluntary", "fails", "off_shelf")}
        r["key"], r["group"], r["difficulty"] = t.key, group(t), t.difficulty
        out.append(r)
    return out


def mean_reward(trees, policy, c) -> float:
    return float(np.mean([rollout(t, policy, c)["reward"] for t in trees]))


def tune(cal, make, grid, c):
    vals = [mean_reward(cal, make(g), c) for g in grid]
    i = int(np.argmax(vals))
    return grid[i], vals[i]


def boot_ci(x: np.ndarray, rng) -> list[float]:
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(rows: list[dict], o2: dict, rng) -> dict:
    r = np.array([x["reward"] for x in rows])
    asked = np.array([x["asked"] for x in rows])
    vol = np.array([x["voluntary"] for x in rows])          # asks the policy chose (not forced by failures)
    inf = np.array([x["difficulty"] == "infeasible" for x in rows])
    return {"n": len(rows), "reward": float(r.mean()), "reward_ci": boot_ci(r, rng),
            "success": float(np.mean([x["success"] for x in rows])), "help": float(asked.mean()),
            "voluntary_ask": float(np.mean([x["voluntary"] for x in rows])),
            "fails": float(np.mean([x["fails"] for x in rows])), "off_shelf": float(np.mean([x["off_shelf"] for x in rows])),
            "ask_precision": float((vol * inf).sum() / max(vol.sum(), 1e-9)),
            "ask_recall": float((vol * inf).sum() / max(inf.sum(), 1e-9)),
            "regret": float(np.mean([o2[x["key"]] - x["reward"] for x in rows]))}


def paired(a: list[dict], b: list[dict], rng) -> dict:
    d = np.array([x["reward"] - y["reward"] for x, y in zip(a, b)])
    out = {"all": {"diff": float(d.mean()), "ci": boot_ci(d, rng), "n": len(d)}}
    for g in sorted({x["group"] for x in a}):
        sel = np.array([x["group"] == g for x in a])
        out[g] = {"diff": float(d[sel].mean()), "ci": boot_ci(d[sel], rng), "n": int(sel.sum())}
    return out


def predictor_metrics(V, head, trees) -> dict:
    out = {}
    for name, sel in (("in_distribution", lambda t: t.split == "train"), ("orange", lambda t: t.object == "orange"),
                      ("metal_spoon", lambda t: t.object == "metal_spoon")):
        ts = [t for t in trees if sel(t)]
        for k in range(3):
            y, p = [], []
            for t, path, v in decision_points(ts):
                if v["k"] == k:
                    y.append(default_success(t, path))
                    p.append(V(v))
            if y:
                out[f"V/{name}/k{k}"] = calibration_metrics(np.array(y), np.array(p))
        if ts:
            pred = np.array([head(t) for t in ts])
            true = np.array([DIFFICULTIES.index(t.difficulty) for t in ts])
            out[f"difficulty/{name}"] = {"n": len(ts), "accuracy": float((pred.argmax(1) == true).mean()),
                                         "nll": float(-np.log(pred[np.arange(len(ts)), true] + 1e-9).mean()),
                                         "mean_prior": pred.mean(0).round(3).tolist(),
                                         "true_mix": (np.bincount(true, minlength=3) / len(true)).round(3).tolist()}
    return out


def evaluate(trees, family, rng, costs_grid) -> dict:
    train = [t for t in trees if t.fold == "train"]
    cal = [t for t in trees if t.fold == "cal"]
    test = [t for t in trees if t.fold == "test"]
    spec = FAMILIES[family]
    V = P.ValueModel().fit(train, cal)
    head = P.DifficultyHead().fit(train)
    obs = P.ObservationModel(spec).fit(train)
    obs_coarse = P.ObservationModel(spec, coarse=True).fit(train)
    onehot = lambda t: np.eye(3)[DIFFICULTIES.index(t.difficulty)]
    res = {"folds": {f: len([t for t in trees if t.fold == f]) for f in ("train", "cal", "test")},
           "test_groups": {g: sum(group(t) == g for t in test) for g in sorted({group(t) for t in test})},
           "test_difficulty": {d: sum(t.difficulty == d for t in test) for d in DIFFICULTIES},
           "predictors": predictor_metrics(V, head, test), "costs": {}}
    for c_help in costs_grid[0]:
        for c_fail in costs_grid[1]:
            t0 = time.time()
            c = Costs(c_help, c_fail)
            o2 = {t.key: best_value(t, c) for t in test}
            k_best, _ = tune(cal, P.ask_after, [1, 2], c)
            tau6, _ = tune(cal, lambda g: P.v_threshold(V, g), TAUS, c)
            tau5, _ = tune(cal, lambda g: P.v_threshold(V, g, first_only=True), TAUS, c)
            methods = {
                "B1 never ask": P.never_ask,
                "B2 always ask": P.always_ask,
                f"B3 ask after k={k_best}": P.ask_after(k_best),
                "B5 V0 threshold": P.v_threshold(V, tau5, first_only=True),
                "B6 Vk threshold": P.v_threshold(V, tau6),
                "B7 myopic EU": P.myopic(V, c),
                "B8 QMDP": P.BeliefPlanner(obs, head, c, mode="qmdp"),
                "B9 fitted Q": P.FittedQ(c).fit(train),
                "Ours belief POMDP": P.BeliefPlanner(obs, head, c),
                "A1 Ours, uniform prior": P.BeliefPlanner(obs, lambda t: np.ones(3) / 3, c),
                "A2 Ours, frequency prior": P.BeliefPlanner(obs, lambda t: head.freq, c),
                "A3 Ours, success/failure only": P.BeliefPlanner(obs_coarse, head, c),
                "O1 difficulty oracle": P.BeliefPlanner(obs, onehot, c),
            }
            if family == "expert":
                methods["A4 Ours, attempt only"] = P.BeliefPlanner(obs, head, c, actions=("A",))
            rows = {name: score(test, pol, c) for name, pol in methods.items()}
            # B4: random asks at Ours's voluntary-ask rate per decision (expected over repeats).
            ours_rate = np.mean([x["voluntary"] for x in rows["Ours belief POMDP"]])
            n_dec = np.mean([1 + x["fails"] for x in rows["Ours belief POMDP"]])
            q = float(min(1.0, ours_rate / max(n_dec, 1e-9)))
            rows["B4 random at Ours's help rate"] = score(test, P.random_ask(q, rng), c, reps=20)
            rows["O2 clairvoyant"] = [{"key": t.key, "group": group(t), "difficulty": t.difficulty, "reward": o2[t.key],
                                       "success": np.nan, "asked": np.nan, "voluntary": np.nan, "fails": np.nan,
                                       "off_shelf": np.nan} for t in test]
            summ = {name: summarize(r, o2, rng) for name, r in rows.items() if not name.startswith("O2")}
            summ["O2 clairvoyant"] = {"reward": float(np.mean(list(o2.values()))),
                                      "reward_ci": boot_ci(np.array(list(o2.values())), rng)}
            diffs = {name: paired(rows["Ours belief POMDP"], r, rng) for name, r in rows.items()
                     if name != "Ours belief POMDP"}
            res["costs"][f"{c_help},{c_fail}"] = {"c_help": c_help, "c_fail": c_fail, "tuned": {"tau_B6": float(tau6),
                                                  "tau_B5": float(tau5), "k_B3": k_best, "q_B4": q},
                                                  "methods": summ, "ours_minus": diffs}
            print(f"c_help {c_help} c_fail {c_fail}: Ours {summ['Ours belief POMDP']['reward']:.2f}  "
                  f"B6 {summ['B6 Vk threshold']['reward']:.2f}  B1 {summ['B1 never ask']['reward']:.2f}  "
                  f"O2 {summ['O2 clairvoyant']['reward']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return res


# ---------------------------------------------------------------- report
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # dataviz reference palette, slots 1-4 (validated order)
INK, INK2, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(res: dict, out: Path, family: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    made = []
    costs = res["costs"]
    # 1. success vs help as c_help sweeps (c_fail fixed at the default)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    _style(ax)
    def find(h, prefix):     # B3's name carries its tuned k
        m = costs[f"{h},{DEFAULT[1]}"]["methods"]
        return m[next(n for n in m if n.startswith(prefix))]

    names = ["Ours belief POMDP", "B6 Vk threshold", "B9 fitted Q", "B3 ask after k"]
    for i, (color, name, mk, ms) in enumerate(zip(SERIES, names, "osD^", (7, 11, 8, 7))):
        pts = [(find(h, name)["help"], find(h, name)["success"], h) for h in C_HELP if f"{h},{DEFAULT[1]}" in costs]
        xs, ys, hs = zip(*pts)
        # Baselines often share operating points: bigger, hollow markers underneath keep each one visible.
        ax.plot(xs, ys, color=color, linewidth=1.5, marker=mk, markersize=ms, zorder=10 - i,
                markerfacecolor=color if i == 0 else "none", markeredgewidth=1.8,
                label="Ours (belief POMDP)" if name.startswith("Ours") else name.split(" ", 1)[1] + (" (tuned k)" if name.startswith("B3") else ""))
        if name.startswith("Ours"):
            for x, y, h in pts:
                ax.annotate(f"c_help={h:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK2)
    ax.set_xlabel("help rate (episodes ending in an ask)", color=INK, fontsize=10)
    ax.set_ylabel("autonomous success rate", color=INK, fontsize=10)
    ax.set_title(f"Success vs. help as asking gets cheaper ({family} trees, c_fail = {DEFAULT[1]:g})", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "success_vs_help.png", dpi=160)
    plt.close(fig)
    made.append("success_vs_help.png")

    # 2. Ours - B6 over the cost grid (diverging: red where Ours loses, blue where it wins, gray at 0)
    H = np.array([[costs[f"{h},{f}"]["ours_minus"]["B6 Vk threshold"]["all"]["diff"] for h in C_HELP] for f in C_FAIL])
    lo = np.array([[costs[f"{h},{f}"]["ours_minus"]["B6 Vk threshold"]["all"]["ci"][0] for h in C_HELP] for f in C_FAIL])
    hi = np.array([[costs[f"{h},{f}"]["ours_minus"]["B6 Vk threshold"]["all"]["ci"][1] for h in C_HELP] for f in C_FAIL])
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    m = max(abs(H).max(), 1e-3)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    ax.imshow(H, cmap=cmap, norm=TwoSlopeNorm(0, -m, m), aspect="auto")
    for i in range(len(C_FAIL)):
        for j in range(len(C_HELP)):
            sig = lo[i, j] > 0 or hi[i, j] < 0
            ax.text(j, i, f"{H[i, j]:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(C_HELP)), [f"{h:g}" for h in C_HELP])
    ax.set_yticks(range(len(C_FAIL)), [f"{f:g}" for f in C_FAIL])
    ax.set_xlabel("c_help", color=INK, fontsize=10)
    ax.set_ylabel("c_fail", color=INK, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Ours − B6 (V_k threshold), mean reward per episode; * = 95% CI excludes 0", color=INK, fontsize=10,
                 loc="left")
    fig.tight_layout()
    fig.savefig(out / "ours_minus_b6_heatmap.png", dpi=160)
    plt.close(fig)
    made.append("ours_minus_b6_heatmap.png")

    # 3. per-condition gap at the default costs, Ours - B6, with CIs
    d = costs[f"{DEFAULT[0]},{DEFAULT[1]}"]["ours_minus"]["B6 Vk threshold"]
    groups = [g for g in d if g != "all"] + ["all"]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(groups) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(groups))[::-1]
    for y, g in zip(ys, groups):
        ax.plot(d[g]["ci"], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot(d[g]["diff"], y, "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(ys, [f"{g} (n={d[g]['n']})" for g in groups])
    ax.set_xlabel("Ours − B6, mean reward per episode (95% CI)", color=INK, fontsize=10)
    ax.set_title(f"Per condition at c_help = {DEFAULT[0]}, c_fail = {DEFAULT[1]}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "per_condition_gap.png", dpi=160)
    plt.close(fig)
    made.append("per_condition_gap.png")
    return made


def summary_md(res: dict, family: str, figs: list[str]) -> str:
    c = res["costs"][f"{DEFAULT[0]},{DEFAULT[1]}"]
    L = [f"# Help-policy evaluation: {family} trees", "",
         f"Folds (starts x conditions): {res['folds']}. Test groups: {res['test_groups']}. Test difficulty: "
         f"{res['test_difficulty']}.", "",
         f"## Default costs (c_help = {DEFAULT[0]}, c_fail = {DEFAULT[1]})", "",
         f"Tuned on cal: {c['tuned']}", "",
         "| method | reward (95% CI) | success | help | voluntary ask | fails/ep | off shelf | vol. ask prec. | vol. ask recall | regret | Ours − method (95% CI) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sorted(c["methods"].items(), key=lambda kv: -kv[1]["reward"]):
        d = c["ours_minus"].get(name, {}).get("all")
        ds = f"{d['diff']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]" if d else "—"
        if name.startswith("O2"):
            L.append(f"| {name} | {s['reward']:.2f} [{s['reward_ci'][0]:.2f}, {s['reward_ci'][1]:.2f}] | | | | | | | | 0 | {ds} |")
            continue
        L.append(f"| {name} | {s['reward']:.2f} [{s['reward_ci'][0]:.2f}, {s['reward_ci'][1]:.2f}] | {s['success']:.2f} | "
                 f"{s['help']:.2f} | {s['voluntary_ask']:.2f} | {s['fails']:.2f} | {s['off_shelf']:.3f} | "
                 f"{s['ask_precision']:.2f} | {s['ask_recall']:.2f} | {s['regret']:.2f} | {ds} |")
    L += ["", "## Ours − B6 (V_k threshold) and Ours − best other non-oracle method, over the cost grid", "",
          "| c_help | c_fail | Ours | B6 | Ours − B6 (95% CI) | best other | Ours − best other (95% CI) |", "|---|---|---|---|---|---|---|"]
    for key, cc in res["costs"].items():
        m = cc["methods"]
        others = {n: s for n, s in m.items() if not n.startswith(("O", "A", "Ours"))}
        best = max(others, key=lambda n: others[n]["reward"])
        d6, db = cc["ours_minus"]["B6 Vk threshold"]["all"], cc["ours_minus"][best]["all"]
        L.append(f"| {cc['c_help']} | {cc['c_fail']} | {m['Ours belief POMDP']['reward']:.2f} | {m['B6 Vk threshold']['reward']:.2f} | "
                 f"{d6['diff']:+.2f} [{d6['ci'][0]:+.2f}, {d6['ci'][1]:+.2f}] | {best} ({others[best]['reward']:.2f}) | "
                 f"{db['diff']:+.2f} [{db['ci'][0]:+.2f}, {db['ci'][1]:+.2f}] |")
    L += ["", "## Predictors on test (V_k: P(success | the default continuation))", "",
          "| predictor | n | base rate | AUROC | Brier | NLL | ECE |", "|---|---|---|---|---|---|---|"]
    for k, s in res["predictors"].items():
        if k.startswith("V/"):
            L.append(f"| {k} | {s['n']} | {s['base_rate']:.2f} | {s['auroc']:.3f} | {s['brier']:.3f} | {s['nll']:.3f} | {s['ece']:.3f} |")
    L += ["", "| difficulty head | n | accuracy | NLL | mean prior (easy, recoverable, infeasible) | true mix |", "|---|---|---|---|---|---|"]
    for k, s in res["predictors"].items():
        if k.startswith("difficulty/"):
            L.append(f"| {k} | {s['n']} | {s['accuracy']:.2f} | {s['nll']:.3f} | {s['mean_prior']} | {s['true_mix']} |")
    L += [""] + [f"![{f}]({f})" for f in figs] + [""]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trees", type=Path, required=True)
    p.add_argument("--family", choices=list(FAMILIES), default="expert")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quick", action="store_true", help="only the default costs")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    trees = load_trees(a.trees, a.family)
    assign_folds(trees, seed=a.seed)
    print(f"{len(trees)} trees; folds", {f: sum(t.fold == f for t in trees) for f in ("train", "cal", "test")}, flush=True)
    grid = ((DEFAULT[0],), (DEFAULT[1],)) if a.quick else (C_HELP, C_FAIL)
    res = evaluate(trees, a.family, np.random.default_rng(a.seed), grid)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1))
    figs = [] if a.quick else figures(res, a.out, a.family)
    (a.out / "summary.md").write_text(summary_md(res, a.family, figs))
    print(f"wrote {a.out}/summary.md")


if __name__ == "__main__":
    main()
