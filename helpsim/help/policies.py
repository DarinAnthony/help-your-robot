"""Help policies (EXPERIMENT_PLAN.md §5): baselines, value-function thresholds, fitted Q-iteration and the belief POMDP.

Every policy is ``policy(view) -> action | "ask"`` over a decision view (``trees.view``). Models are fit on the train
fold, calibrated (Platt) and tuned (thresholds, k) on the cal fold, and scored on test.

- ``ValueModel``: V_k = P(success | the decision point, the default continuation); hand features (``trees.features``),
  logistic regression, Platt-scaled.
- ``ImageValueModel`` (V1): the same target from frozen DINOv2 embeddings of the decision frames (``scripts/
  embed_trees.py``; PCA fit on train) plus proprioception and attempt history, a 5-member bootstrap ensemble of small
  MLPs, Platt-scaled. It never sees the object's identity, physical properties or pose.
- ``ImageDifficultyHead``: P(d | the start's frames), for the image-prior ablation (H3).
- ``DifficultyHead``: P(d | start) for d in easy / recoverable / infeasible, multinomial logistic regression on the
  start's features: the POMDP's prior.
- ``ObservationModel``: P(outcome | d, action, attempt index), counted on the train trees (every played-out child).
- ``BeliefPlanner``: Bayes updates of the belief over d from each attempt's outcome, then exact finite-horizon value
  iteration over beliefs (horizon: the attempts left). ``mode="qmdp"`` plans as if d became known after this step
  (QMDP); a one-hot prior on the true d is the difficulty oracle.
- ``FittedQ``: Q_k(features, action) by backward ridge regression over the train trees' real transitions.
"""

from __future__ import annotations


import numpy as np

from helpsim.help.models import MLP, Logistic, Platt
from helpsim.help.trees import (DIFFICULTIES, MAX_ATTEMPTS, Costs, Tree, decision_points, default_action,
                                default_success, features, history_features, outcome_reward, static_features, view)


# ---------------------------------------------------------------- predictors
class ValueModel:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.cache = {}

    def fit(self, train: list[Tree], cal: list[Tree]) -> "ValueModel":
        X, y = self._data(train)
        self.lr = Logistic(self.l2).fit(X, y, 2)
        Xc, yc = self._data(cal)
        self.platt = Platt().fit(self.lr.predict_proba(Xc)[:, 1], yc)
        return self

    @staticmethod
    def _data(trees):
        X, y = [], []
        for t, path, v in decision_points(trees):
            X.append(features(v))
            y.append(default_success(t, path))
        return np.array(X), np.array(y, int)

    def __call__(self, v: dict) -> float:
        key = (v["tree"].key, v["path"])
        if key not in self.cache:
            self.cache[key] = float(self.platt(self.lr.predict_proba(features(v)[None])[:, 1])[0])
        return self.cache[key]


class Embeddings:
    """Per-node frame embeddings (``scripts/embed_trees.py``): ``get(tree, path)`` -> (cameras * 768,)."""

    def __init__(self, npz):
        z = np.load(npz)
        self.index = {k: i for i, k in enumerate(z["keys"].tolist())}
        self.emb = z["emb"].reshape(len(self.index), -1).astype(np.float32)

    def has(self, tree: Tree) -> bool:
        return f"{tree.key}|" in self.index

    def get(self, tree: Tree, path: str) -> np.ndarray:
        return self.emb[self.index[f"{tree.key}|{path}"]]


class PCA:
    def __init__(self, n: int):
        self.n = n

    def fit(self, X: np.ndarray) -> "PCA":
        self.mu = X.mean(0)
        _, s, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.vt, self.scale = vt[:self.n], s[:self.n] / np.sqrt(len(X))
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - self.mu) @ self.vt.T / self.scale


def proprio_history(v: dict) -> np.ndarray:
    """The robot's own state (arm joints, gripper) and the attempt history (index, outcomes, failure phases), without
    the object's simulated pose (the last 5 entries of ``history_features``)."""
    st = v["state"]
    return np.concatenate([np.asarray(st["arm_qpos"], float), [st["gripper_pos"]], history_features(v)[:-5]])


class ImageValueModel:
    """V1: P(success | the default continuation) from the decision frames, proprioception and history."""

    def __init__(self, emb: Embeddings, pc_grid: tuple = (16, 32, 64, 128), members: int = 5, hidden: int = 32,
                 l2_grid: tuple = (3.0, 10.0, 30.0, 100.0, 300.0)):
        self.emb, self.pc_grid, self.members, self.hidden, self.l2_grid = emb, pc_grid, members, hidden, l2_grid
        self.cache = {}

    def _raw(self, trees):
        E, X, y = [], [], []
        for t, path, v in decision_points(trees):
            E.append(self.emb.get(t, path))
            X.append(proprio_history(v))
            y.append(default_success(t, path))
        return np.array(E), np.array(X), np.array(y, int)

    @staticmethod
    def _ens(models, Xf):
        return np.mean([m.predict_proba(Xf) for m in models], 0)

    def fit(self, train: list[Tree], cal: list[Tree]) -> "ImageValueModel":
        E, X, y = self._raw(train)
        Ec, Xc, yc = self._raw(cal)
        pca_full = PCA(max(self.pc_grid)).fit(E)
        rng = np.random.default_rng(0)
        boots = [rng.integers(0, len(y), len(y)) for _ in range(self.members)]
        best, self.nll = None, {}
        for n_pc in self.pc_grid:      # PCA size and weight penalty: the best calibration-fold likelihood
            Xf, Xcf = np.c_[pca_full(E)[:, :n_pc], X], np.c_[pca_full(Ec)[:, :n_pc], Xc]
            for l2 in self.l2_grid:
                models = [MLP(self.hidden, l2, seed=i).fit(Xf[b], y[b]) for i, b in enumerate(boots)]
                p = np.clip(self._ens(models, Xcf), 1e-6, 1 - 1e-6)
                nll = float(-(yc * np.log(p) + (1 - yc) * np.log(1 - p)).mean())
                self.nll[(n_pc, l2)] = nll
                if best is None or nll < best[0]:
                    best = (nll, n_pc, l2, models, p)
        _, self.n_pc, self.l2, self.models, p = best
        self.pca = pca_full
        self.platt = Platt().fit(p, yc)
        return self

    def __call__(self, v: dict) -> float:
        key = (v["tree"].key, v["path"])
        if key not in self.cache:
            xf = np.c_[self.pca(self.emb.get(v["tree"], v["path"]))[:, :self.n_pc], proprio_history(v)[None]]
            self.cache[key] = float(self.platt(self._ens(self.models, xf))[0])
        return self.cache[key]


class ImageDifficultyHead:
    """P(d | the start's frames): multinomial logistic regression on the PCA of the root embeddings."""

    def __init__(self, emb: Embeddings, n_pc: int = 32, l2: float = 3.0):
        self.emb, self.n_pc, self.l2 = emb, n_pc, l2

    def fit(self, train: list[Tree]) -> "ImageDifficultyHead":
        E = np.array([self.emb.get(t, "") for t in train])
        self.pca = PCA(self.n_pc).fit(E)
        y = np.array([DIFFICULTIES.index(t.difficulty) for t in train])
        self.lr = Logistic(self.l2).fit(self.pca(E), y, len(DIFFICULTIES))
        self.freq = np.bincount(y, minlength=len(DIFFICULTIES)) / len(y)
        return self

    def __call__(self, tree: Tree) -> np.ndarray:
        return self.lr.predict_proba(self.pca(self.emb.get(tree, "")))[0]


class DifficultyHead:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, train: list[Tree]) -> "DifficultyHead":
        X = np.array([static_features(t) if t.family != "drawer" else np.zeros(1) for t in train])
        y = np.array([DIFFICULTIES.index(t.difficulty) for t in train])
        self.lr = Logistic(self.l2).fit(X, y, len(DIFFICULTIES))
        self.freq = np.bincount(y, minlength=len(DIFFICULTIES)) / len(y)
        return self

    def __call__(self, tree: Tree) -> np.ndarray:
        if tree.family == "drawer":         # identical starts: the prior is the training mix
            return self.freq
        return self.lr.predict_proba(static_features(tree)[None])[0]


# The drawer's failed pulls, by how far the drawer moved (drawer_max_m): locked ~8 mm, stuck 19-20 mm, a failed hard
# pull ~34 mm, "short" >= 5 cm (EXPERIMENT_PLAN.md §3). The outcome categories alone lump the first three together.
# Under 4 mm: no pull to speak of (pi0.5's grasp misses and early slips; the expert's failed pulls all held), so the
# locked drawer's 8 mm stop gets a bin of its own: pi0.5's failed attempts in 4-12 mm are 20 of 36 on the locked drawer,
# 1 of 16 on the stuck one, none of 55 on the others (2026-11-26 runs).
DRAWER_TRAVEL_BINS = (0.004, 0.012, 0.026, 0.05)


class ObservationModel:
    """P(o | d, a, k) with additive smoothing over observation symbols. A symbol is the attempt's outcome, refined for
    the drawer by its travel bin (``travel_bins``); ``coarse`` keeps only success / failure / terminal failure types."""

    def __init__(self, family_spec: dict, coarse: bool = False, alpha: float = 0.5, travel_bins: tuple | None = None,
                 spread_key: str | None = None):
        self.spec, self.coarse, self.alpha, self.bins = family_spec, coarse, alpha, travel_bins
        tf = family_spec["terminal_fail"]
        if coarse:
            self.outcomes = ("success", "fail") + tuple(tf)
        elif travel_bins:
            self.outcomes = ("success",) + tuple(f"travel{i}" for i in range(len(travel_bins) + 1))
        else:
            self.outcomes = family_spec["outcomes"]
        self.terminal_fail = tf
        # ``spread_key``: each failure symbol also says whether pi0.5's spread at the next decision (node state
        # ``spread``, e.g. ``arm_std_exec``) is above the training median: low (_lo), high (_hi) or not recorded (_na).
        self.spread_key, self.spread_thr = spread_key, None
        if spread_key:
            self.outcomes = tuple(o if o == "success" or o in tf else f"{o}_{s}" for o in self.outcomes
                                  for s in (("",) if o == "success" or o in tf else ("lo", "hi", "na")))

    def obs(self, outcome: str, state: dict | None = None) -> str:
        if outcome == "success" or outcome in self.terminal_fail:
            return outcome
        if self.coarse:
            sym = "fail"
        elif self.bins:
            sym = f"travel{int(np.searchsorted(self.bins, (state or {}).get('drawer_max_m', 0.0)))}"
        else:
            sym = outcome
        if self.spread_key:
            v = ((state or {}).get("spread") or {}).get(self.spread_key)
            sym += "_na" if v is None or self.spread_thr is None else ("_hi" if v > self.spread_thr else "_lo")
        return sym

    def fit(self, train: list[Tree]) -> "ObservationModel":
        acts = sorted(set(self.spec["actions"]) | set(self.spec["root_actions"]))
        self.acts = acts
        if self.spread_key:      # the median spread after a failed attempt, over the training nodes
            vals = [n["state"]["spread"][self.spread_key] for t in train for p, n in t.nodes.items()
                    if p and n["outcome"] != "success" and (n["state"].get("spread") or {}).get(self.spread_key) is not None]
            self.spread_thr = float(np.median(vals)) if vals else None
        n = np.full((len(DIFFICULTIES), MAX_ATTEMPTS, len(acts), len(self.outcomes)), self.alpha)
        for t in train:
            d = DIFFICULTIES.index(t.difficulty)
            for path, node in t.nodes.items():
                if not path:
                    continue
                parent, a, k = path[:-1], path[-1], len(path) - 1
                if t.terminal(t.nodes[parent]["outcome"]):
                    continue
                n[d, k, acts.index(a), self.outcomes.index(self.obs(node["outcome"], node["state"]))] += 1
        self.p = n / n.sum(-1, keepdims=True)
        return self

    def __call__(self, d: int, k: int, a: str) -> np.ndarray:
        return self.p[d, k, self.acts.index(a)]


# ---------------------------------------------------------------- planners
class BeliefPlanner:
    """The belief POMDP (Ours) and its relatives. ``prior(tree) -> P(d)``."""

    def __init__(self, obs: ObservationModel, prior, costs: Costs, mode: str = "pomdp", actions: tuple | None = None):
        self.obs, self.prior, self.c, self.mode = obs, prior, costs, mode
        self.allowed = actions
        self._vd = {}

    def belief(self, v: dict) -> np.ndarray:
        b = np.asarray(self.prior(v["tree"]), float).copy()
        for i, (o, _ph) in enumerate(v["history"]):
            a = v["path"][i]
            sym = self.obs.obs(o, v["tree"].nodes[v["path"][:i + 1]]["state"])
            b *= np.array([self.obs(d, i, a)[self.obs.outcomes.index(sym)] for d in range(len(b))])
            b /= b.sum()
        return b

    def _acts(self, k: int) -> tuple:
        acts = self.obs.spec["root_actions"] if k == 0 else self.obs.spec["actions"]
        return tuple(a for a in acts if self.allowed is None or a in self.allowed)

    def _step(self, o: str) -> tuple[float, bool]:
        if o.endswith(("_lo", "_hi", "_na")):          # spread-refined symbols (ObservationModel spread_key)
            o = o[:-3]
        if o == "fail" or o.startswith("travel"):     # a non-terminal failure symbol
            return -self.c.c_fail, False
        return outcome_reward(o, self.c, self.obs.terminal_fail)

    def value(self, b: np.ndarray, k: int) -> float:
        return max(self.q(b, k).values())

    def q(self, b: np.ndarray, k: int) -> dict:
        if k >= MAX_ATTEMPTS:
            return {"ask": -self.c.c_help}
        out = {"ask": -self.c.c_help}
        for a in self._acts(k):
            if self.mode == "qmdp":
                out[a] = float(sum(b[d] * self.q_known(d, k)[a] for d in range(len(b))))
                continue
            P = np.array([self.obs(d, k, a) for d in range(len(b))])     # (D, O)
            po = b @ P
            total = 0.0
            for j, o in enumerate(self.obs.outcomes):
                if po[j] < 1e-12:
                    continue
                r, done = self._step(o)
                if not done:
                    r += self.value(b * P[:, j] / po[j], k + 1)
                total += po[j] * r
            out[a] = total
        return out

    def q_known(self, d: int, k: int) -> dict:
        """Q values with the difficulty known (the per-d MDP)."""
        key = (d, k)
        if key not in self._vd:
            if k >= MAX_ATTEMPTS:
                self._vd[key] = {"ask": -self.c.c_help}
            else:
                out = {"ask": -self.c.c_help}
                for a in self._acts(k):
                    total = 0.0
                    for j, o in enumerate(self.obs.outcomes):
                        p = self.obs(d, k, a)[j]
                        r, done = self._step(o)
                        if not done:
                            r += max(self.q_known(d, k + 1).values())
                        total += p * r
                    out[a] = total
                self._vd[key] = out
        return self._vd[key]

    def __call__(self, v: dict) -> str:
        q = self.q(self.belief(v), v["k"])
        q = {a: x for a, x in q.items() if a == "ask" or a in v["actions"]}
        return max(q, key=q.get)


class FittedQ:
    """B9: Q_k(x, a) for k = 2, 1, 0 by ridge regression on the train trees' transitions; ask is worth -c_help."""

    def __init__(self, costs: Costs, l2: float = 1.0):
        self.c, self.l2 = costs, l2

    def fit(self, train: list[Tree]) -> "FittedQ":
        self.W = {}
        for k in reversed(range(MAX_ATTEMPTS)):
            data = {}
            for t, path, v in decision_points(train):
                if len(path) != k:
                    continue
                x = features(v)
                for a in v["actions"]:
                    child = path + a
                    o = t.nodes[child]["outcome"]
                    r, done = outcome_reward(o, self.c, t.spec["terminal_fail"])
                    if not done:
                        r += self.value(view(t, child))
                    data.setdefault(a, ([], []))
                    data[a][0].append(x)
                    data[a][1].append(r)
            for a, (X, y) in data.items():
                X = np.array(X)
                mu, sd = X.mean(0), X.std(0) + 1e-6
                Z = np.c_[(X - mu) / sd, np.ones(len(X))]
                reg = self.l2 * np.eye(Z.shape[1])
                reg[-1, -1] = 0
                w = np.linalg.solve(Z.T @ Z + reg, Z.T @ np.array(y))
                self.W[(k, a)] = (mu, sd, w)
        return self

    def q(self, v: dict) -> dict:
        out = {"ask": -self.c.c_help}
        if v["k"] >= MAX_ATTEMPTS:
            return out
        x = features(v)
        for a in v["actions"]:
            if (v["k"], a) in self.W:
                mu, sd, w = self.W[(v["k"], a)]
                out[a] = float(np.r_[(x - mu) / sd, 1.0] @ w)
        return out

    def value(self, v: dict) -> float:
        return max(self.q(v).values())

    def __call__(self, v: dict) -> str:
        q = self.q(v)
        return max(q, key=q.get)


# ---------------------------------------------------------------- simple policies
def never_ask(v):
    return default_action(v)


def always_ask(v):
    return "ask"


def ask_after(k: int):
    return lambda v: "ask" if sum(o != "success" for o, _ in v["history"]) >= k else default_action(v)


def v_threshold(V, tau: float, first_only: bool = False):
    """B6 (every decision) or B5 (``first_only``: decide once at the start, then never ask)."""
    def policy(v):
        if first_only and v["k"] > 0:
            return default_action(v)
        return "ask" if V(v) < tau else default_action(v)
    return policy


def myopic(V, c: Costs):
    """B7: continue iff 10 V - c_fail (1 - V) > -c_help."""
    return lambda v: default_action(v) if 10 * V(v) - c.c_fail * (1 - V(v)) > -c.c_help else "ask"


def random_ask(q: float, rng: np.random.Generator):
    return lambda v: "ask" if rng.random() < q else default_action(v)
