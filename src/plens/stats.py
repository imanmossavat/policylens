"""Uncertainty for success rates and paired comparisons of runs.

Success is one yes/no per episode, so rates from tens of episodes are noisy. Runs on the
same seeds can be compared episode by episode (paired), which needs far fewer episodes
than comparing two rates. Pure Python + numpy; no scipy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n. Stays inside [0, 1], unlike k/n +- 2 SE."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value of k heads in n fair coin flips."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: dict[int, bool], b: dict[int, bool]) -> dict[str, Any]:
    """Compare two runs' successes on the seeds they share.

    only_a / only_b: seeds where just one run succeeded (the discordant pairs; all the
    information about a difference is there). p: exact McNemar test. diff_ci: 95% Wald
    interval for rate_b - rate_a on the shared seeds.
    """
    seeds = sorted(set(a) & set(b))
    n = len(seeds)
    only_a = sum(a[s] and not b[s] for s in seeds)
    only_b = sum(b[s] and not a[s] for s in seeds)
    both = sum(a[s] and b[s] for s in seeds)
    diff = (only_b - only_a) / n if n else 0.0
    se = math.sqrt(max((only_a + only_b) - (only_b - only_a) ** 2 / n, 0.0)) / n if n else 0.0
    return {"n_shared": n, "both": both, "only_a": only_a, "only_b": only_b,
            "neither": n - both - only_a - only_b, "diff": diff,
            "diff_ci": (diff - Z95 * se, diff + Z95 * se),
            "p_mcnemar": binom_two_sided(min(only_a, only_b), only_a + only_b)}


def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval of stat over resampled values (e.g. per-episode step time)."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(lo), float(hi))


def episodes_for_halfwidth(p: float, halfwidth: float, z: float = Z95) -> int:
    """Episodes needed so a success rate near p has a 95% interval of +- halfwidth."""
    return math.ceil(z * z * p * (1 - p) / halfwidth ** 2)


def rate_summary(successes: Sequence[bool]) -> dict[str, Any]:
    k, n = int(sum(successes)), len(successes)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": (round(lo, 4), round(hi, 4))}
