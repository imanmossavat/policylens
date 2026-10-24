"""Linear probes: can a quantity the policy never receives be read out of its activations?

Used for "does a vision-only policy know about contact?" (Q-012): the target is a
simulator ground truth (e.g. peg-socket contact), the features are recorded layer
outputs. Episodes, not steps, are split into train and test, so a probe cannot score by
memorising an episode.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank-sum formula: P(score of a positive > score of a negative), ties count half."""
    from scipy.stats import rankdata

    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 200):
    """Standardise, then L2-regularised logistic regression with LBFGS. Returns a scorer."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mu) / sd, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    w_, b_ = w.detach().numpy(), float(b.detach())
    return lambda z: ((z - mu) / sd) @ w_ + b_


def split_episodes(ids: Sequence[int], test_fraction: float = 0.3, seed: int = 0) -> tuple[set, set]:
    ids = sorted(set(ids))
    rng = np.random.default_rng(seed)
    test = set(rng.choice(ids, max(1, int(round(test_fraction * len(ids)))), replace=False).tolist())
    return set(ids) - test, test


def probe(features: np.ndarray, target: np.ndarray, episode: np.ndarray, groups: dict[str, np.ndarray] | None = None,
          n_splits: int = 5) -> dict[str, float]:
    """Mean test AUC over n_splits episode-level splits; also per group (boolean masks over rows)."""
    out: dict[str, list[float]] = {"all": []}
    for g in groups or {}:
        out[g] = []
    for k in range(n_splits):
        train_ids, test_ids = split_episodes(episode.tolist(), seed=k)
        tr, te = np.isin(episode, list(train_ids)), np.isin(episode, list(test_ids))
        if len(set(target[tr])) < 2:
            continue
        score = fit_logistic(features[tr], target[tr])
        s = score(features[te])
        out["all"].append(auc(s, target[te]))
        for g, mask in (groups or {}).items():
            out[g].append(auc(s[mask[te]], target[te][mask[te]]))
    return {g: float(np.nanmean(v)) if v else float("nan") for g, v in out.items()}
