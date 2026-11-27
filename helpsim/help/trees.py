"""Decision trees for the help study (EXPERIMENT_PLAN.md §4, TASK_SPEC.md §6): loading, exact scoring and features.

A tree is one start state with every action sequence played out to ``MAX_ATTEMPTS`` attempts. Its nodes are keyed by
``path``, the string of actions that reached them (``""`` is the start). Families:

- ``expert``: the scripted expert's shelf trees (``collect_shelf_pnp.py --tree``), actions A (attempt: the nominal
  grasp again) and D (adjust: the swung, offset re-approach), both from every node including the start; terminal
  outcomes success, off_shelf and place_miss.
- ``pi05``: pi0.5's shelf trees (``scripts/pi05_rollouts.py trees``), actions C (continue) and R (reset and retry), only
  C from the start; terminal outcomes success and off_shelf.
- ``drawer``: the TidyBot's logged re-pull runs (``tidybot_open_rp_*``, ``scripts/drawer_attempts.py``): one path of
  up to 3 pulls (action P) per episode, outcomes success / stuck / short. An episode the planner ended before its third
  pull (it refused to re-plan, e.g. after a weak grip's slip) has no further child: the robot must ask there.

A help policy sees only the decision view (``view``): the start, the path so far, each past attempt's outcome and
failure phase, and the state at the current node. ``rollout`` scores it exactly on a tree, since every child it can
choose was played out. Reward (EXPERIMENT_PLAN.md §2): +10 for autonomous success, -c_help for asking (the human then
finishes), -c_fail per failed attempt, -5 more for knocking the object off the shelf; a terminal failure, or the third
failure, forces an ask.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_ATTEMPTS = 3
SUCCESS_R, OFF_SHELF_R = 10.0, 5.0
FAMILIES = {
    "expert": {"actions": ("A", "D"), "root_actions": ("A", "D"), "terminal_fail": ("off_shelf", "place_miss"),
               "outcomes": ("success", "unreachable", "grasp_miss", "slip", "place_miss", "off_shelf"),
               "phases": ("approach", "grasp", "carry")},
    "pi05": {"actions": ("C", "R"), "root_actions": ("C",), "terminal_fail": ("off_shelf",),
             "outcomes": ("success", "grasp_miss", "slip", "place_miss", "off_shelf", "timeout"), "phases": ()},
    "drawer": {"actions": ("P",), "root_actions": ("P",), "terminal_fail": (),
               "outcomes": ("success", "stuck", "short"), "phases": ()},
}
DIFFICULTIES = ("easy", "recoverable", "infeasible")
OBJECT_FEATURES = json.loads((Path(__file__).with_name("object_features.json")).read_text())
SHELF_REGION = ((8.975, 9.03), (0.40, 0.74))     # helpsim.tasks.shelf_pnp REGION_X, REGION_Y


@dataclass
class Tree:
    family: str
    condition: str          # expert mode or pi0.5 fault (hidden from the policies)
    object: str
    split: str              # the object's split in collect_shelf_pnp.SPLITS
    seed: int
    start_obj_pos: np.ndarray
    target: np.ndarray
    nodes: dict = field(repr=False)
    fold: str = ""          # train / cal / test (assign_folds)
    frames: str = ""        # the tree's decision-frame npz (``tree_<seed>.npz``), if any

    @property
    def spec(self) -> dict:
        return FAMILIES[self.family]

    def actions(self, path: str) -> tuple:
        acts = self.spec["root_actions"] if not path else self.spec["actions"]
        return tuple(a for a in acts if path + a in self.nodes)

    def terminal(self, outcome: str | None) -> bool:
        return outcome == "success" or outcome in self.spec["terminal_fail"]

    @property
    def difficulty(self) -> str:
        """easy: the default first attempt succeeds; recoverable: some later node succeeds; infeasible: none does."""
        first = self.spec["root_actions"][0]
        if self.nodes[first]["outcome"] == "success":
            return "easy"
        return "recoverable" if any(n["outcome"] == "success" for n in self.nodes.values()) else "infeasible"

    @property
    def key(self) -> str:
        return f"{self.family}/{self.condition}/{self.object}/{self.seed}"


def load_trees(root: Path, family: str) -> list[Tree]:
    """Every ``trees.jsonl`` under root: ``<mode>/<split>/<object>/`` for the expert, ``<object>/`` for pi0.5."""
    out = []
    for f in sorted(Path(root).rglob("trees.jsonl")):
        for line in f.read_text().splitlines():
            if not line:
                continue
            r = json.loads(line)
            nodes = {n["path"]: {"fail_phase": None, **n} for n in r["nodes"]}   # pi0.5 trees record no phase
            split = r.get("split") or ("heldout_in_distribution" if r["object"] == "orange" else
                                       "heldout_hard" if r["object"] == "metal_spoon" else "train")
            start = r.get("start_obj_pos") or nodes[""]["state"]["obj_pos"]
            target = r.get("target")
            if target is None:
                raise ValueError(f"{f}: no target in tree records")
            out.append(Tree(family, r.get("mode", "nominal"), r["object"], split, r["seed"], np.array(start),
                            np.array(target), nodes, frames=str(f.parent / f"tree_{r['seed']:05d}.npz")))
    return out


DRAWER_HELD_OUT = ("medium",)


def load_drawer_trees(root: Path, runs: str = "tidybot_open_rp_*") -> list[Tree]:
    """The ``attempts.jsonl`` of the runs matching ``runs`` under ``root`` (xemb's ``open_planner_v1``): the planner's
    re-pull runs, or pi0.5's (``scripts/run_drawer_pi05.sh``, one decision rule per run, so one path per episode). A
    tree's object is its drawer condition (folds are stratified by it); ``medium`` is held out (test only)."""
    out = []
    for f in sorted(Path(root).glob(f"{runs}/attempts.jsonl")):
        for i, line in enumerate(f.read_text().splitlines()):
            if not line:
                continue
            r = json.loads(line)
            cond = r.get("variant") or ("transient_stop" if "transient" in r["run"] else
                                        "weak_grip" if "weak_grip" in r["run"] else "nominal")
            nodes = {"": {"path": "", "outcome": None, "fail_phase": None, "state": {}}}
            for a in r["attempts"][:MAX_ATTEMPTS]:
                path = "P" * (a["attempt"] + 1)
                nodes[path] = {"path": path, "outcome": a["outcome"], "fail_phase": None,
                               "state": {**{k: a[k] for k in ("drawer_max_m", "drawer_end_m", "base_travel_m")},
                                         "event": a.get("event")}}      # pi0.5 runs: the monitor's event
            seed = zlib.crc32(f'{r["run"]}/{r["file"]}/{r["traj"]}'.encode())     # stable across processes
            out.append(Tree("drawer", cond, cond, "heldout" if cond in DRAWER_HELD_OUT else "train", seed,
                            np.zeros(3), np.zeros(3), nodes))
    return out


def assign_folds(trees: list[Tree], fractions=(0.60, 0.15, 0.25), seed: int = 0) -> None:
    """Training objects' starts split 60/15/25 into train/cal/test by (object, seed), so a start shared across
    conditions lands in the same fold; held-out objects are test only."""
    starts = sorted({(t.object, t.seed) for t in trees if t.split == "train"})
    rng = np.random.default_rng(seed)
    by_obj = {}
    for o, s in starts:
        by_obj.setdefault(o, []).append(s)
    fold = {}
    for o, seeds in by_obj.items():
        seeds = list(rng.permutation(seeds))
        n_tr, n_cal = int(round(len(seeds) * fractions[0])), int(round(len(seeds) * fractions[1]))
        for i, s in enumerate(seeds):
            fold[(o, s)] = "train" if i < n_tr else "cal" if i < n_tr + n_cal else "test"
    for t in trees:
        t.fold = fold.get((t.object, t.seed), "test")


# ---------------------------------------------------------------- scoring
@dataclass
class Costs:
    c_help: float = 3.0
    c_fail: float = 0.5


def outcome_reward(outcome: str, c: Costs, terminal_fail: tuple) -> tuple[float, bool]:
    """(reward of the attempt that ended this way, whether the episode ends), the forced ask of a terminal failure
    included."""
    if outcome == "success":
        return SUCCESS_R, True
    r = -c.c_fail - (OFF_SHELF_R if outcome == "off_shelf" else 0.0)
    if outcome in terminal_fail:
        return r - c.c_help, True
    return r, False


def view(tree: Tree, path: str) -> dict:
    """What a help policy may see at the decision point after ``path``."""
    hist = [(tree.nodes[path[:i + 1]]["outcome"], tree.nodes[path[:i + 1]]["fail_phase"]) for i in range(len(path))]
    return {"tree": tree, "path": path, "k": len(path), "history": hist, "state": tree.nodes[path]["state"],
            "actions": tree.actions(path)}


def rollout(tree: Tree, policy, c: Costs) -> dict:
    """The exact episode of ``policy`` (``policy(view) -> action | "ask"``) on the tree."""
    path, reward, fails = "", 0.0, 0
    tf = tree.spec["terminal_fail"]
    while True:
        if len(path) >= MAX_ATTEMPTS:
            return {"reward": reward - c.c_help, "success": False, "asked": True, "voluntary": False, "fails": fails,
                    "off_shelf": False, "path": path}
        if not tree.actions(path):      # nothing was played out from here (a logged episode that stopped): ask
            return {"reward": reward - c.c_help, "success": False, "asked": True, "voluntary": False, "fails": fails,
                    "off_shelf": False, "path": path}
        a = policy(view(tree, path))
        if a == "ask":
            return {"reward": reward - c.c_help, "success": False, "asked": True, "voluntary": True, "fails": fails,
                    "off_shelf": False, "path": path}
        if a not in tree.actions(path):
            raise ValueError(f"{tree.key}: action {a} not played out at '{path}'")
        path += a
        o = tree.nodes[path]["outcome"]
        r, done = outcome_reward(o, c, tf)
        reward += r
        fails += o != "success"
        if done:
            return {"reward": reward, "success": o == "success", "asked": o != "success", "voluntary": False,
                    "fails": fails, "off_shelf": o == "off_shelf", "path": path}


def best_value(tree: Tree, c: Costs, path: str = "") -> float:
    """Clairvoyant (O2): the best any policy can do on this tree, knowing every outcome."""
    if len(path) >= MAX_ATTEMPTS:
        return -c.c_help
    best = -c.c_help
    for a in tree.actions(path):
        o = tree.nodes[path + a]["outcome"]
        r, done = outcome_reward(o, c, tree.spec["terminal_fail"])
        best = max(best, r + (0.0 if done else best_value(tree, c, path + a)))
    return best


def default_action(v: dict) -> str:
    """The family's default continuation: the expert attempts, then adjusts; pi0.5 continues, then resets."""
    acts = v["actions"]
    if not acts:
        return "ask"
    return acts[0] if v["k"] == 0 or len(acts) == 1 else acts[1]


def default_success(tree: Tree, path: str) -> bool:
    """Whether the default continuation from ``path`` (never asking) ends in success: the value function's label."""
    while len(path) < MAX_ATTEMPTS:
        a = default_action(view(tree, path))
        if a == "ask":
            return False
        path += a
        o = tree.nodes[path]["outcome"]
        if o == "success":
            return True
        if tree.terminal(o):
            return False
    return False


# ---------------------------------------------------------------- features
def _quat_up(q) -> float:
    """z component of the object's body z axis (1 upright, 0 on its side)."""
    w, x, y, z = q
    return 1.0 - 2.0 * (x * x + y * y)


def static_features(tree: Tree) -> np.ndarray:
    """The start: the object's physical features (never its name, so held-out objects get features too) and layout."""
    o = OBJECT_FEATURES[tree.object]
    s, tg = tree.start_obj_pos, tree.target
    (x0, x1), (y0, y1) = SHELF_REGION
    return np.array([o["long_m"] * 10, o["short_m"] * 10, o["height_m"] * 10, np.log(o["mass_kg"]), o["friction"],
                     (s[0] - x0) / (x1 - x0), (s[1] - y0) / (y1 - y0), (tg[0] - x0) / (x1 - x0), (tg[1] - y0) / (y1 - y0),
                     np.linalg.norm(s[:2] - tg[:2]) * 10])


STATIC_NAMES = ["long", "short", "height", "log_mass", "friction", "start_x", "start_y", "target_x", "target_y",
                "start_target_dist"]


def history_features(v: dict) -> np.ndarray:
    """Decision index, previous outcomes and failure phases, and the object's state now."""
    spec = v["tree"].spec
    fails = [o for o in spec["outcomes"] if o != "success"]
    k = np.eye(MAX_ATTEMPTS)[v["k"]]
    last = np.zeros(len(fails))
    counts = np.zeros(len(fails))
    phase = np.zeros(max(1, len(spec["phases"])))
    for o, ph in v["history"]:
        if o in fails:
            counts[fails.index(o)] += 1
    if v["history"]:
        o, ph = v["history"][-1]
        if o in fails:
            last[fails.index(o)] = 1
        if ph in spec["phases"]:
            phase[spec["phases"].index(ph)] = 1
    st, t = v["state"], v["tree"]
    com, start = np.array(st["obj_com"]), np.array(t.nodes[""]["state"]["obj_com"])
    dyn = np.array([np.linalg.norm(com[:2] - start[:2]) * 10, (com[2] - start[2]) * 10, _quat_up(st["obj_quat"]),
                    np.linalg.norm(com[:2] - t.target[:2]) * 10, float(st.get("on_shelf", True))])
    return np.concatenate([k, last, counts, phase, dyn])


def drawer_features(v: dict) -> np.ndarray:
    """Drawer: the decision index, previous outcomes and the last pull's drawer and base travel (every start is the
    same drawer from the same pose, so there is nothing static to see)."""
    spec = v["tree"].spec
    fails = [o for o in spec["outcomes"] if o != "success"]
    k = np.eye(MAX_ATTEMPTS)[v["k"]]
    last, counts = np.zeros(len(fails)), np.zeros(len(fails))
    for o, _ in v["history"]:
        if o in fails:
            counts[fails.index(o)] += 1
    if v["history"] and v["history"][-1][0] in fails:
        last[fails.index(v["history"][-1][0])] = 1
    st = v["state"]
    travel = np.array([st.get("drawer_max_m", 0.0) * 10, st.get("drawer_end_m", 0.0) * 10, st.get("base_travel_m", 0.0) * 10])
    return np.concatenate([k, last, counts, travel])


def features(v: dict) -> np.ndarray:
    if v["tree"].family == "drawer":
        return drawer_features(v)
    return np.concatenate([static_features(v["tree"]), history_features(v)])


def decision_points(trees: list[Tree]):
    """Every non-terminal node above the depth limit, with its view."""
    for t in trees:
        for path, n in t.nodes.items():
            if len(path) < MAX_ATTEMPTS and not t.terminal(n["outcome"]):
                yield t, path, view(t, path)
