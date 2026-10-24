"""Outcome labels and run summaries.

Labels are deliberately coarse. "near_miss" separates "got the T almost there but
failed final alignment" (difficulty B in docs/tasks.md) from "never got close".
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .stats import paired, rate_summary, wilson

NEAR_MISS_COVERAGE = 0.8  # gym-pusht counts >= 0.95 coverage as success


def outcome_label(success: bool, max_coverage: float,
                  near_miss: float = NEAR_MISS_COVERAGE) -> str:
    if success:
        return "success"
    return "near_miss" if max_coverage >= near_miss else "fail"


def summarize(index: list[dict[str, Any]]) -> dict[str, Any]:
    """Success rate and label counts from a run's episodes.jsonl lines."""
    n = len(index)
    if n == 0:
        return {"n_episodes": 0}
    labels = Counter(line["label"] for line in index)
    k = sum(line["success"] for line in index)
    lo, hi = wilson(k, n)
    return {
        "n_episodes": n,
        "success_rate": round(k / n, 4),
        "success_ci95": [round(lo, 4), round(hi, 4)],
        "labels": dict(sorted(labels.items())),
        "mean_max_coverage": round(sum(line["max_coverage"] for line in index) / n, 4),
        "mean_steps": round(sum(line["n_steps"] for line in index) / n, 1),
    }


def compare(indexes: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One row per run: success with its 95% interval, and a paired test against the first run.

    indexes: {name: episodes.jsonl lines}, first entry = reference. Pairs by seed, so runs
    should share seeds; n_shared says how many did.
    """
    rows, ref = [], None
    for name, index in indexes.items():
        by_seed = {line["seed"]: bool(line["success"]) for line in index}
        row = {"name": name, **rate_summary(list(by_seed.values())),
               "mean_max_coverage": round(sum(line["max_coverage"] for line in index) / len(index), 4)}
        if ref is None:
            ref = by_seed
        else:
            row["vs_ref"] = paired(ref, by_seed)
        rows.append(row)
    return rows
