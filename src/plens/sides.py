"""Which way around the T does the agent go, and does it change its mind? (Q-001, diff:A)

Direction is the agent's angle around the block, measured in the block's own frame, so
pushing the block (which turns both together) is not counted as going around it.
Positive is counter-clockwise in env coordinates, negative clockwise.
A "swing" is a re-plan chunk in which the agent turns by at least `min_swing` radians;
a "switch" is a swing whose direction differs from the previous swing's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import store

CHUNK = 8          # steps per re-plan (the policy's n_action_steps)
MIN_SWING = 0.5    # rad (~29 degrees); smaller turns are jitter while pushing


def block_frame_angle(pos_agent: np.ndarray, block_pose: np.ndarray) -> np.ndarray:
    """Unwrapped angle of the agent around the block centre, in the block frame, per step."""
    rel = np.asarray(pos_agent, float) - np.asarray(block_pose, float)[:, :2]
    th = np.asarray(block_pose, float)[:, 2]
    c, s = np.cos(-th), np.sin(-th)
    x = c * rel[:, 0] - s * rel[:, 1]
    y = s * rel[:, 0] + c * rel[:, 1]
    return np.unwrap(np.arctan2(y, x))


def chunk_turns(phi: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """Signed turn (rad) of `phi` during each re-plan chunk."""
    n = len(phi)
    return np.array([phi[min(k + chunk, n) - 1] - phi[max(k - 1, 0)] for k in range(0, n, chunk)])


def analyze_episode(trace: dict[str, np.ndarray], chunk: int = CHUNK,
                    min_swing: float = MIN_SWING) -> dict[str, Any]:
    turns = chunk_turns(block_frame_angle(trace["pos_agent"], trace["block_pose"]), chunk)
    swings = [(int(k), 1 if t > 0 else -1) for k, t in enumerate(turns) if abs(t) >= min_swing]
    signs = [d for _, d in swings]
    return {
        "swings": swings,
        "n_ccw": signs.count(1),
        "n_cw": signs.count(-1),
        "n_switches": sum(a != b for a, b in zip(signs, signs[1:])),
    }


def analyze_run(run_dir: Path | str, chunk: int = CHUNK,
                min_swing: float = MIN_SWING) -> list[dict[str, Any]]:
    """One row per episode: its index line (idx, seed, label, ...) plus analyze_episode."""
    rows = []
    for line in store.read_index(run_dir):
        trace = store.load_episode(run_dir, line["idx"])["trace"]
        rows.append({**line, **analyze_episode(trace, chunk, min_swing)})
    return rows


def by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per outcome label: episodes, mean swings, and how many had at least one switch."""
    out: dict[str, dict[str, float]] = {}
    for label in sorted({r["label"] for r in rows}):
        sub = [r for r in rows if r["label"] == label]
        out[label] = {
            "n": len(sub),
            "mean_swings": round(float(np.mean([r["n_ccw"] + r["n_cw"] for r in sub])), 2),
            "mean_switches": round(float(np.mean([r["n_switches"] for r in sub])), 2),
            "with_switch": sum(r["n_switches"] > 0 for r in sub),
        }
    return out
