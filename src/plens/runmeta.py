"""Run folders and their metadata.

Every experiment writes to its own folder runs/<run_id>/ with a meta.json that
answers: what ran, with which code (git commit, dirty flag), which package
versions, which device, which config and seeds, and how it ended.
A run folder is never edited after status becomes "done" (see D-004).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

META_FILE = "meta.json"
TRACKED_PACKAGES = ("plens", "lerobot", "gym-pusht", "torch", "numpy", "gymnasium", "robosuite", "mujoco")


def _git_info(cwd: Path) -> dict[str, Any] | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip() != ""
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


PATCH_FILE = "git_diff.patch"
PATCH_DIRS = ("src", "scripts", "tests", "docs")


def _git_patch(cwd: Path) -> str:
    """Uncommitted changes as a patch: tracked edits plus new files under PATCH_DIRS."""
    def git(*args: str) -> str:
        # --no-index diffs exit 1 when files differ, so no check=True
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout

    parts = [git("diff", "HEAD")]
    for path in git("ls-files", "--others", "--exclude-standard", *PATCH_DIRS).split():
        parts.append(git("diff", "--no-index", "--", "/dev/null", path))
    return "".join(parts)


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_info(cwd: Path | None = None) -> dict[str, Any]:
    """Machine, Python, package versions and git state at run time."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
        "git": _git_info(cwd or Path.cwd()),
    }


def new_run(root: Path | str, name: str, config: dict[str, Any]) -> Path:
    """Create runs/<timestamp>_<name>/ and write the initial meta.json."""
    created = datetime.now(timezone.utc)
    run_id = f"{created:%Y%m%d-%H%M%S}_{name}"
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "run_id": run_id,
        "name": name,
        "status": "running",
        "created_utc": created.isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config": config,
        "environment": environment_info(),
    }
    # WHY: a dirty commit hash alone cannot rebuild the code that ran; the patch can.
    if (meta["environment"]["git"] or {}).get("dirty"):
        (run_dir / PATCH_FILE).write_text(_git_patch(Path.cwd()))
        meta["environment"]["git"]["patch"] = PATCH_FILE
    write_meta(run_dir, meta)
    return run_dir


def read_meta(run_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(run_dir) / META_FILE).read_text())


def write_meta(run_dir: Path | str, meta: dict[str, Any]) -> None:
    path = Path(run_dir) / META_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    tmp.replace(path)  # atomic: a crash never leaves a half-written meta.json


def update_meta(run_dir: Path | str, **fields: Any) -> dict[str, Any]:
    """Merge top-level fields into meta.json and return the new meta."""
    meta = read_meta(run_dir)
    meta.update(fields)
    write_meta(run_dir, meta)
    return meta
