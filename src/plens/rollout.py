"""Run episodes and keep everything needed to explain their outcomes afterwards.

run_batch steps N envs in lockstep through one policy call per step (one batched
forward pass serves all N). run_episode is the N=1 case and reproduces runs made
before batching existed exactly (D-012).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from lerobot.envs import preprocess_observation

from .policy import PolicyBundle
from .record import ActivationRecorder

# Per-step env info kept in the trajectory. Keys are gym-pusht's info names.
TRACE_KEYS = ("pos_agent", "block_pose", "coverage", "n_contacts")


@dataclass
class Episode:
    seed: int
    success: bool
    n_steps: int
    max_coverage: float
    final_coverage: float
    trace: dict[str, np.ndarray]  # per step: action, reward, and TRACE_KEYS
    frames: list[np.ndarray] = field(default_factory=list)  # only if rendered

    def summary(self) -> dict:
        return {"seed": self.seed, "success": self.success, "n_steps": self.n_steps,
                "max_coverage": round(self.max_coverage, 4),
                "final_coverage": round(self.final_coverage, 4)}


def run_batch(bundle: PolicyBundle, venv, seeds: Sequence[int],
              recorder: ActivationRecorder | None = None, render: Sequence[int] = (),
              noise_seed: int | None = None) -> list[Episode]:
    """One episode per env of `venv`, env i reset with seeds[i]. Returns them in seed order.

    render: indices of envs whose frames to keep (each 300-step video is ~130 MB in RAM).
    noise_seed: torch seed for the policy's sampling noise; default seeds[0].
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
    traces: list[dict[str, list]] = [{k: [] for k in ("action", "reward", *TRACE_KEYS)} for _ in range(n)]
    frames: list[list[np.ndarray]] = [[] for _ in range(n)]
    success = [False] * n
    n_steps = [0] * n
    active = [True] * n
    step = 0

    while any(active):
        if recorder is not None:
            recorder.step = step
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
            for k in TRACE_KEYS:
                traces[i][k].append(np.asarray(src[k][i]))
            success[i] = success[i] or bool(src["is_success"][i])
            n_steps[i] = step + 1
            active[i] = not done
        step += 1

    episodes = []
    for i, seed in enumerate(seeds):
        arrays = {k: np.asarray(v) for k, v in traces[i].items()}
        cov = arrays["coverage"].astype(float)
        episodes.append(Episode(seed=seed, success=success[i], n_steps=n_steps[i],
                                max_coverage=float(cov.max()), final_coverage=float(cov[-1]),
                                trace=arrays, frames=frames[i]))
    return episodes


def run_episode(bundle: PolicyBundle, venv, seed: int,
                recorder: ActivationRecorder | None = None, render: bool = False) -> Episode:
    """A single episode in a one-env venv (make_single_env)."""
    return run_batch(bundle, venv, [seed], recorder, render=(0,) if render else ())[0]
