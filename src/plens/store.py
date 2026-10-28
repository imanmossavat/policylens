"""On-disk layout of a run (grep "LAYOUT" to find this):

LAYOUT runs/<run_id>/
  meta.json              what ran and how it ended (runmeta.py)
  git_diff.patch         only if the code was uncommitted: the exact changes on top of the commit
  episodes.jsonl         one summary line per episode; the index to scan first
  episodes/ep_00000.npz  trace/<key> arrays, plus act/<layer>/{step,call,value}
  videos/ep_00000.mp4    optional, first --video N episodes
  fate.jsonl             fate runs only (scripts/fate.py): one line per snapshot, p(success) from there
  assembled.npz          extraction runs only (scripts/disassembly.py): frozen assembled states
  extraction.json        extraction runs only: stage counts per scripted replay length
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # WHY: rollout imports lerobot, absent in the robosuite envs
    from .rollout import Episode

INDEX_FILE = "episodes.jsonl"


def episode_path(run_dir: Path | str, idx: int) -> Path:
    return Path(run_dir) / "episodes" / f"ep_{idx:05d}.npz"


def save_episode(run_dir: Path | str, idx: int, ep: Episode, label: str,
                 activations: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """Write the episode arrays and append its summary line to the index. Returns that line."""
    path = episode_path(run_dir, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"trace/{k}": v for k, v in ep.trace.items()}
    for layer, parts in (activations or {}).items():
        for part, arr in parts.items():
            arrays[f"act/{layer}/{part}"] = arr
    np.savez_compressed(path, **arrays)

    line = {"idx": idx, **ep.summary(), "label": label,
            "layers": sorted(activations) if activations else []}
    with open(Path(run_dir) / INDEX_FILE, "a") as f:
        f.write(json.dumps(line) + "\n")
    return line


def save_npz_episode(run_dir: Path | str, idx: int, trace: dict[str, np.ndarray],
                     line: dict[str, Any]) -> dict[str, Any]:
    """save_episode for runs without a LeRobot Episode (robosuite Square): trace arrays + index line."""
    path = episode_path(run_dir, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"trace/{k}": v for k, v in trace.items()})
    line = {"idx": idx, **line}
    with open(Path(run_dir) / INDEX_FILE, "a") as f:
        f.write(json.dumps(line) + "\n")
    return line


def save_video(run_dir: Path | str, idx: int, frames: list[np.ndarray], fps: int = 10) -> Path:
    import imageio.v2 as imageio

    path = Path(run_dir) / "videos" / f"ep_{idx:05d}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)
    return path


def read_index(run_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(run_dir) / INDEX_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_episode(run_dir: Path | str, idx: int) -> dict[str, Any]:
    """{"trace": {key: array}, "act": {layer: {"step","call","value"}}}."""
    out: dict[str, Any] = {"trace": {}, "act": {}}
    with np.load(episode_path(run_dir, idx)) as data:
        for key in data.files:
            kind, rest = key.split("/", 1)
            if kind == "trace":
                out["trace"][rest] = data[key]
            else:
                layer, part = rest.rsplit("/", 1)
                out["act"].setdefault(layer, {})[part] = data[key]
    return out


def latest_run(root: Path | str = "runs") -> Path:
    runs = sorted(p for p in Path(root).iterdir() if (p / "meta.json").exists())
    if not runs:
        raise FileNotFoundError(f"No runs in {root}")
    return runs[-1]
