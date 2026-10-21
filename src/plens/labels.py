"""Outcome labels and run summaries.

Labels are deliberately coarse. "near_miss" separates "got the T almost there but
failed final alignment" (difficulty B in docs/tasks.md) from "never got close".
"""

from __future__ import annotations

from collections import Counter
from typing import Any

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
    return {
        "n_episodes": n,
        "success_rate": round(sum(line["success"] for line in index) / n, 4),
        "labels": dict(sorted(labels.items())),
        "mean_max_coverage": round(sum(line["max_coverage"] for line in index) / n, 4),
        "mean_steps": round(sum(line["n_steps"] for line in index) / n, 1),
    }
