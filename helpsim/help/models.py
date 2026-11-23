"""Small models for the help study, on numpy and scipy: L2 logistic regression (binary or multinomial), Platt scaling,
and the predictor metrics of EXPERIMENT_PLAN.md §6 (AUROC, Brier, NLL, ECE)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, log_softmax


class Logistic:
    """Multinomial logistic regression with standardized inputs and an L2 penalty on the weights (not the biases)."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "Logistic":
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        C = n_classes or int(y.max()) + 1
        Y = np.eye(C)[y]
        D = Z.shape[1]
        mask = np.ones((D, C))
        mask[-1] = 0

        def loss(w):
            W = w.reshape(D, C)
            lp = log_softmax(Z @ W, axis=1)
            g = Z.T @ (np.exp(lp) - Y) + self.l2 * mask * W
            return -(Y * lp).sum() + 0.5 * self.l2 * ((mask * W) ** 2).sum(), g.ravel()

        res = minimize(loss, np.zeros(D * C), jac=True, method="L-BFGS-B")
        self.W = res.x.reshape(D, C)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = np.c_[(np.asarray(X, float) - self.mu) / self.sd, np.ones(len(X))]
        return np.exp(log_softmax(Z @ self.W, axis=1))


class Platt:
    """p -> sigmoid(a * logit(p) + b), fit by maximum likelihood on a calibration set."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Platt":
        z = logit(np.clip(p, 1e-6, 1 - 1e-6))
        y = np.asarray(y, float)

        def loss(ab):
            q = np.clip(expit(ab[0] * z + ab[1]), 1e-9, 1 - 1e-9)
            return -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum()

        self.a, self.b = minimize(loss, [1.0, 0.0], method="Nelder-Mead").x
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return expit(self.a * logit(np.clip(p, 1e-6, 1 - 1e-6)) + self.b)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, bool), np.asarray(p, float)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):              # average ranks over ties
        tie = p == v
        ranks[tie] = ranks[tie].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, curve = 0.0, []
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
            curve.append([float(p[sel].mean()), float(y[sel].mean()), int(sel.sum())])
    return {"n": int(len(y)), "base_rate": float(y.mean()), "auroc": auroc(y, p), "brier": float(((p - y) ** 2).mean()),
            "nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "ece": float(ece), "reliability": curve}
