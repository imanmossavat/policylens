"""Run episodes and keep everything needed to explain their outcomes afterwards.

run_batch steps N envs in lockstep through one policy call per step (one batched
forward pass serves all N). run_episode is the N=1 case and reproduces runs made
before batching existed exactly (D-012).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from lerobot.envs import preprocess_observation

from .envs import PUSHT, TaskSpec
from .policy import PolicyBundle
from .record import ActivationRecorder

# Per-step env info kept in a PushT trajectory (gym-pusht's info names); other tasks: TaskSpec.
TRACE_KEYS = PUSHT.trace_keys


@dataclass
class Episode:
    seed: int
    success: bool
    n_steps: int
    max_coverage: float    # max task progress in [0, 1]; for PushT the T overlap (TaskSpec.progress)
    final_coverage: float
    trace: dict[str, np.ndarray]  # per step: action, reward, progress, task.trace_keys, task.observe keys
    frames: list[np.ndarray] = field(default_factory=list)  # only if rendered

    def summary(self) -> dict:
        return {"seed": self.seed, "success": self.success, "n_steps": self.n_steps,
                "max_coverage": round(self.max_coverage, 4),
                "final_coverage": round(self.final_coverage, 4)}


def run_batch(bundle: PolicyBundle, venv, seeds: Sequence[int],
              recorder: ActivationRecorder | None = None, render: Sequence[int] = (),
              noise_seed: int | None = None, on_step: Callable | None = None,
              task: TaskSpec = PUSHT) -> list[Episode]:
    """One episode per env of `venv`, env i reset with seeds[i]. Returns them in seed order.

    render: indices of envs whose frames to keep (each 300-step video is ~130 MB in RAM).
    noise_seed: torch seed for the policy's sampling noise; default seeds[0].
    on_step: called as on_step(step, venv, obs) before each policy call (e.g. branch.Recorder).
    """
    n = venv.num_envs
    if len(seeds) != n:
        raise ValueError(f"{len(seeds)} seeds for {n} envs")
    # WHY: seed torch too. Diffusion sampling draws noise, so the env seed alone
    # does not make an episode reproducible.
    # GOTCHA(G-008): the noise is drawn for the whole batch, so an episode's noise depends on
    # which batch it ran in; only batch size 1 reproduces per-seed results exactly.
    torch.manual_seed(seeds[0] if noise_seed is None else noise_seed)
    bundle.policy.reset()
    obs, _ = venv.reset(seed=list(seeds))
    traces: list[dict[str, list]] = [defaultdict(list) for _ in range(n)]
    frames: list[list[np.ndarray]] = [[] for _ in range(n)]
    success = [False] * n
    n_steps = [0] * n
    active = [True] * n
    step = 0

    while any(active):
        if recorder is not None:
            recorder.step = step
        if on_step is not None:
            on_step(step, venv, obs)
        if render:
            images = venv.call("render")
            for i in render:
                if active[i]:
                    frames[i].append(images[i])
        batch = preprocess_observation(obs)
        batch["task"] = [""] * n
        batch = bundle.preprocessor(bundle.env_preprocessor(batch))
        with torch.inference_mode():
            action = bundle.policy.select_action(batch)
        action = bundle.env_postprocessor({"action": bundle.postprocessor(action)})["action"]
        action = action.to("cpu").numpy()
        if task.observe is not None:  # ground truth at the moment of the observation
            for i in range(n):
                if active[i]:
                    for k, v in task.observe(venv.envs[i]).items():
                        traces[i][k].append(v)

        obs, reward, terminated, truncated, info = venv.step(action)
        for i in range(n):
            if not active[i]:
                continue  # WHY: a finished env autoresets into an unseeded episode; ignore it
            done = bool(terminated[i] or truncated[i])
            # GOTCHA(G-004): with same-step autoreset, a finished env's `info` already belongs to
            # its next episode; the real last-step info is under "final_info".
            src = info["final_info"] if done else info
            traces[i]["action"].append(action[i])
            traces[i]["reward"].append(float(reward[i]))
            for k in task.trace_keys:
                traces[i][k].append(np.asarray(src[k][i]))
            traces[i]["progress"].append(task.progress(float(reward[i]), src, i))
            success[i] = success[i] or bool(src["is_success"][i])
            n_steps[i] = step + 1
            active[i] = not done
        step += 1

    episodes = []
    for i, seed in enumerate(seeds):
        arrays = {k: np.asarray(v) for k, v in traces[i].items()}
        cov = arrays["progress"].astype(float)
        episodes.append(Episode(seed=seed, success=success[i], n_steps=n_steps[i],
                                max_coverage=float(cov.max()), final_coverage=float(cov[-1]),
                                trace=arrays, frames=frames[i]))
    return episodes


def run_episode(bundle: PolicyBundle, venv, seed: int,
                recorder: ActivationRecorder | None = None, render: bool = False) -> Episode:
    """A single episode in a one-env venv (make_single_env)."""
    return run_batch(bundle, venv, [seed], recorder, render=(0,) if render else ())[0]
