"""Run one episode and keep everything needed to explain its outcome afterwards."""

from __future__ import annotations

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
    frames: list[np.ndarray] = field(default_factory=list)  # only if render=True

    def summary(self) -> dict:
        return {"seed": self.seed, "success": self.success, "n_steps": self.n_steps,
                "max_coverage": round(self.max_coverage, 4),
                "final_coverage": round(self.final_coverage, 4)}


def run_episode(bundle: PolicyBundle, venv, seed: int,
                recorder: ActivationRecorder | None = None, render: bool = False) -> Episode:
    # WHY: seed torch too. Diffusion sampling draws noise, so the env seed alone
    # does not make an episode reproducible.
    torch.manual_seed(seed)
    bundle.policy.reset()
    obs, _ = venv.reset(seed=[seed])
    trace: dict[str, list] = {k: [] for k in ("action", "reward", *TRACE_KEYS)}
    frames: list[np.ndarray] = []
    success, step, done = False, 0, False

    while not done:
        if recorder is not None:
            recorder.step = step
        if render:
            frames.append(venv.call("render")[0])
        batch = preprocess_observation(obs)
        batch["task"] = [""]
        batch = bundle.preprocessor(bundle.env_preprocessor(batch))
        with torch.inference_mode():
            action = bundle.policy.select_action(batch)
        action = bundle.env_postprocessor({"action": bundle.postprocessor(action)})["action"]
        action = action.to("cpu").numpy()

        obs, reward, terminated, truncated, info = venv.step(action)
        done = bool(terminated[0] or truncated[0])
        if done:
            # GOTCHA(G-004): with same-step autoreset, `info` already belongs to the next
            # episode; the real last-step info is under "final_info".
            info = info["final_info"]
        trace["action"].append(action[0])
        trace["reward"].append(float(reward[0]))
        for k in TRACE_KEYS:
            trace[k].append(np.asarray(info[k][0]))
        success = success or bool(info["is_success"][0])
        step += 1

    arrays = {k: np.asarray(v) for k, v in trace.items()}
    cov = arrays["coverage"].astype(float)
    return Episode(seed=seed, success=success, n_steps=step,
                   max_coverage=float(cov.max()), final_coverage=float(cov[-1]),
                   trace=arrays, frames=frames)
