"""Environment construction. Envs run in lockstep batches; each env still yields its own per-step data."""

from __future__ import annotations

import gymnasium as gym
from lerobot.envs import make_env
from lerobot.envs.configs import PushtEnv

# GOTCHA(G-003): gym-pusht returns 96x96 images whatever LeRobot's default says,
# and the pretrained policy expects 96x96, so pin it here.
PUSHT_IMAGE_SIZE = 96


def pusht_config(max_steps: int = 300) -> PushtEnv:
    return PushtEnv(
        episode_length=max_steps,
        observation_height=PUSHT_IMAGE_SIZE,
        observation_width=PUSHT_IMAGE_SIZE,
    )


def make_envs(env_cfg: PushtEnv, n_envs: int) -> gym.vector.VectorEnv:
    """A synchronous vector env of n_envs copies, stepped together (see rollout.run_batch)."""
    envs = make_env(env_cfg, n_envs=n_envs)
    # make_env returns {suite: {task_id: VectorEnv}}; with one task we want that VectorEnv.
    while isinstance(envs, dict):
        envs = next(iter(envs.values()))
    return envs


def make_single_env(env_cfg: PushtEnv) -> gym.vector.VectorEnv:
    """A vector env holding exactly one env, the shape LeRobot's processors expect."""
    return make_envs(env_cfg, 1)
