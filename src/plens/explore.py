"""Cheap, re-computable views of recorded runs, for interactive exploration (D-010).

Running the robot is slow; everything here only re-reads what a run saved, so every
parameter (chunk, min_swing, layer, episode) can be changed and re-evaluated instantly.
Pure numpy; drawing is in explore_plots.py and sliders in explore_ui.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from . import runmeta, sides, store


@dataclass
class EpisodeView:
    idx: int
    seed: int
    label: str
    trace: dict[str, np.ndarray]
    video: Path | None = None
    run_dir: Path | None = None
    _act: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def n_steps(self) -> int:
        return len(self.trace["pos_agent"])

    @cached_property
    def phi(self) -> np.ndarray:
        """Agent angle around the block in the block frame, unwrapped (D-009)."""
        return sides.block_frame_angle(self.trace["pos_agent"], self.trace["block_pose"])

    def analyze(self, chunk: int = sides.CHUNK, min_swing: float = sides.MIN_SWING) -> dict[str, Any]:
        return sides.analyze_episode(self.trace, chunk, min_swing)

    def switch_steps(self, chunk: int = sides.CHUNK, min_swing: float = sides.MIN_SWING) -> list[int]:
        """Env steps at which a swing reverses the previous swing's direction."""
        swings = self.analyze(chunk, min_swing)["swings"]
        return [k * chunk for (_, a), (k, b) in zip(swings, swings[1:]) if a != b]

    def activations(self) -> dict[str, dict[str, np.ndarray]]:
        """{layer: {"step","call","value"}}; loaded on first use."""
        if self._act is None:
            if self.run_dir is None:
                return {}
            self._act = store.load_episode(self.run_dir, self.idx)["act"]
        return self._act


@dataclass
class RunData:
    run_dir: Path
    episodes: list[EpisodeView]

    def by_label(self) -> dict[str, list[EpisodeView]]:
        out: dict[str, list[EpisodeView]] = {}
        for ep in self.episodes:
            out.setdefault(ep.label, []).append(ep)
        return out

    def get(self, idx: int) -> EpisodeView:
        return next(e for e in self.episodes if e.idx == idx)


def load(run_dir: Path | str | None = None) -> RunData:
    """Load every episode's trace (small). Activations stay on disk until asked for."""
    run_dir = Path(run_dir) if run_dir else store.latest_run()
    if not run_dir.is_dir():
        near = sorted(p.name for p in run_dir.parent.glob("*") if p.is_dir()) if run_dir.parent.is_dir() else []
        raise FileNotFoundError(f"No run folder at {run_dir.resolve()}. Runs found next to it: {near or 'none'}")
    eps = []
    for line in store.read_index(run_dir):
        vid = run_dir / "videos" / f"ep_{line['idx']:05d}.mp4"
        eps.append(EpisodeView(line["idx"], line["seed"], line["label"],
                               store.load_episode(run_dir, line["idx"])["trace"],
                               vid if vid.exists() else None, run_dir))
    return RunData(run_dir, eps)


def provenance(run: RunData) -> dict[str, Any]:
    """Where a run's data came from: the files on disk and the settings that produced them.

    Read straight from meta.json (D-004), so it cannot drift from what actually ran.
    Missing fields come back as None: old or partial runs still display.
    """
    d = Path(run.run_dir).resolve()
    try:
        meta = runmeta.read_meta(d)
    except FileNotFoundError:
        meta = {}
    cfg, env, pol = meta.get("config", {}), meta.get("environment", {}), meta.get("policy", {})
    git = env.get("git") or {}
    return {
        "run_id": meta.get("run_id", d.name),
        "folder": str(d),
        "files": {"meta.json": str(d / runmeta.META_FILE), "episodes.jsonl": str(d / store.INDEX_FILE),
                  "episodes/": str(d / "episodes"), "videos/": str(d / "videos")},
        "status": meta.get("status"),
        "created_utc": meta.get("created_utc"),
        "duration_s": meta.get("duration_s"),
        "command": " ".join(meta["argv"]) if meta.get("argv") else None,
        "policy": {"repo_id": pol.get("repo_id") or cfg.get("policy"), "checkpoint_commit": pol.get("commit"),
                   "type": pol.get("policy_type"), "n_params": pol.get("n_params"), "device": pol.get("device"),
                   "random_weights": cfg.get("random_weights")},
        "settings": {k: cfg.get(k) for k in ("n", "seed", "max_steps", "layers", "no_pool", "all_calls",
                                             "video", "near_miss_coverage")},
        "code": {"commit": git.get("commit"), "dirty": git.get("dirty")},
        "packages": env.get("packages") or {},
        "summary": meta.get("summary"),
        "with_video": [e.idx for e in run.episodes if e.video is not None],
        "raw_meta": meta,
    }


def episode_file(run: RunData, idx: int) -> Path:
    """The .npz holding episode idx's recorded arrays (trace/... and act/...)."""
    return store.episode_path(Path(run.run_dir).resolve(), idx)


def sweep(run: RunData, thresholds: np.ndarray, chunk: int = sides.CHUNK) -> np.ndarray:
    """Switch count per (threshold, episode): shape (len(thresholds), len(run.episodes))."""
    return np.array([[e.analyze(chunk, float(t))["n_switches"] for e in run.episodes] for t in thresholds])


def sweep_by_label(run: RunData, thresholds: np.ndarray, chunk: int = sides.CHUNK) -> dict[str, np.ndarray]:
    """Mean switches per label at each threshold."""
    counts = sweep(run, thresholds, chunk)
    labels = np.array([e.label for e in run.episodes])
    return {lab: counts[:, labels == lab].mean(axis=1) for lab in sorted(set(labels))}


def toy_episode(waypoints: list[float], steps_per_leg: int = 16, radius: float = 100.0,
                wobble: float = 0.0, seed: int = 0, label: str = "toy") -> EpisodeView:
    """A fake dot circling a fixed fake block, passing through `waypoints` (rad, block frame).

    Lets you decide when it switches and see what the measurement counts and misses.
    `wobble` adds Gaussian jitter (rad) to the angle each step.
    """
    legs = [np.linspace(a, b, steps_per_leg, endpoint=False) for a, b in zip(waypoints, waypoints[1:])]
    ang = np.concatenate(legs + [[waypoints[-1]]])
    ang = ang + np.random.default_rng(seed).normal(0, wobble, len(ang))
    block = np.array([256.0, 256.0, 0.0])
    pos = block[:2] + radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    trace = {"pos_agent": pos, "block_pose": np.tile(block, (len(ang), 1)),
             "coverage": np.zeros(len(ang))}
    return EpisodeView(-1, seed, label, trace)


def activation_projection(values: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Project (calls, ...) activations onto their top-k principal axes.

    Returns (coords (calls, k), explained variance fraction (k,)). Within one episode,
    so it shows how the layer moves over time, not a comparison across episodes.
    """
    x = values.reshape(len(values), -1).astype(float)
    x = x - x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    k = min(k, len(s))
    var = s**2 / max((s**2).sum(), 1e-12)
    return u[:, :k] * s[:k], var[:k]


def activation_change(values: np.ndarray) -> np.ndarray:
    """Distance between consecutive re-plans' activations (len calls-1): when does the layer jump?"""
    x = values.reshape(len(values), -1).astype(float)
    return np.linalg.norm(np.diff(x, axis=0), axis=1)
