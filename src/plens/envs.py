"""Environment construction and per-task specs. Envs run in lockstep batches; each env still yields its own per-step data.

A TaskSpec says, for one task, what to record and what "progress" means, so rollouts,
labels and branching work the same on PushT and on ALOHA (docs/tasks.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from lerobot.envs import make_env
from lerobot.envs.configs import EnvConfig, PushtEnv

# GOTCHA(G-003): gym-pusht returns 96x96 images whatever LeRobot's default says,
# and the pretrained policy expects 96x96, so pin it here.
PUSHT_IMAGE_SIZE = 96


def pusht_config(max_steps: int = 300) -> PushtEnv:
    return PushtEnv(
        episode_length=max_steps,
        observation_height=PUSHT_IMAGE_SIZE,
        observation_width=PUSHT_IMAGE_SIZE,
    )


def aloha_config(task: str, max_steps: int = 400) -> EnvConfig:
    from lerobot.envs.configs import AlohaEnv  # needs gym-aloha (plens-aloha env)

    return AlohaEnv(task=task, episode_length=max_steps)


def make_envs(env_cfg: EnvConfig, n_envs: int) -> gym.vector.VectorEnv:
    """A synchronous vector env of n_envs copies, stepped together (see rollout.run_batch)."""
    envs = make_env(env_cfg, n_envs=n_envs)
    # make_env returns {suite: {task_id: VectorEnv}}; with one task we want that VectorEnv.
    while isinstance(envs, dict):
        envs = next(iter(envs.values()))
    return envs


def make_single_env(env_cfg: EnvConfig) -> gym.vector.VectorEnv:
    """A vector env holding exactly one env, the shape LeRobot's processors expect."""
    return make_envs(env_cfg, 1)


# ---------------------------------------------------------------- ALOHA physics

ALOHA_MAX_REWARD = 4  # both tasks: reward is a stage 0..4, success = 4


def _aloha_physics(env):
    return env.unwrapped._env.physics


def _contact_force(physics, geoms_a: set[str], geoms_b: set[str]) -> tuple[float, int]:
    """Summed normal contact force (N) and contact count between two geom sets."""
    import mujoco

    total, n = 0.0, 0
    f6 = np.zeros(6)
    for i in range(physics.data.ncon):
        c = physics.data.contact[i]
        g1, g2 = physics.model.id2name(c.geom1, "geom"), physics.model.id2name(c.geom2, "geom")
        if (g1 in geoms_a and g2 in geoms_b) or (g1 in geoms_b and g2 in geoms_a):
            mujoco.mj_contactForce(physics.model.ptr, physics.data.ptr, i, f6)
            total += abs(f6[0])
            n += 1
    return total, n


SOCKET_GEOMS = {"socket-1", "socket-2", "socket-3", "socket-4", "pin"}


def observe_insertion(env) -> dict[str, np.ndarray]:
    """Ground truth the policy cannot see directly: object poses and peg-socket contact force."""
    ph = _aloha_physics(env)
    force, n = _contact_force(ph, {"red_peg"}, SOCKET_GEOMS)
    return {"peg_pos": np.array(ph.named.data.xpos["peg"]), "socket_pos": np.array(ph.named.data.xpos["socket"]),
            "peg_socket_force": np.float32(force), "peg_socket_contacts": np.int32(n),
            "agent_pos": np.array(env.unwrapped._env.task.get_qpos(ph))}  # = the policy's state input


def observe_transfer_cube(env) -> dict[str, np.ndarray]:
    ph = _aloha_physics(env)
    return {"box_pos": np.array(ph.named.data.xpos["box"])}


# ---------------------------------------------------------------- task specs

@dataclass(frozen=True)
class TaskSpec:
    """What a rollout records for one task and how its progress is measured.

    trace_keys: per-step info keys kept. progress: per-step task progress in [0, 1] from
    (reward, info, env index); its maximum labels the episode (labels.outcome_label).
    observe: optional per-env ground truth read from the simulator before each step.
    """
    name: str
    make_config: Callable[[int], EnvConfig]
    max_steps: int
    policy: str
    trace_keys: tuple[str, ...] = ()
    layers: tuple[str, ...] = ()  # default layers to record (batch-first outputs only, see G-010)
    progress: Callable[[float, dict, int], float] = field(default=lambda r, info, i: r)
    observe: Callable[[Any], dict[str, np.ndarray]] | None = None
    near_miss: float = 0.8
    note: str = ""


PUSHT = TaskSpec(
    "pusht", pusht_config, 300, "lerobot/diffusion_pusht",
    trace_keys=("pos_agent", "block_pose", "coverage", "n_contacts"),
    layers=("diffusion.rgb_encoder", "diffusion.unet.mid_modules.1"),  # D-005
    progress=lambda r, info, i: float(info["coverage"][i]),
    near_miss=0.8, note="progress = T overlap; success at 0.95")

# GOTCHA(G-010): ACT's transformer outputs are sequence-first (tokens, batch, dim), so the
# recorder's pooling and per-env split would cut the wrong axis; record CNN layers only.
ACT_LAYERS = ("model.backbone.layer4", "model.encoder_img_feat_input_proj")

ALOHA_INSERTION = TaskSpec(
    "aloha_insertion", lambda n: aloha_config("AlohaInsertion-v0", n), 400, "lerobot/act_aloha_sim_insertion_human",
    layers=ACT_LAYERS,
    progress=lambda r, info, i: float(r) / ALOHA_MAX_REWARD, observe=observe_insertion,
    near_miss=0.75, note="progress = reward stage / 4: grasp each part, peg touches socket, inserted")

ALOHA_TRANSFER_CUBE = TaskSpec(
    "aloha_transfer_cube", lambda n: aloha_config("AlohaTransferCube-v0", n), 400,
    "lerobot/act_aloha_sim_transfer_cube_human", layers=ACT_LAYERS,
    progress=lambda r, info, i: float(r) / ALOHA_MAX_REWARD, observe=observe_transfer_cube,
    near_miss=0.75, note="progress = reward stage / 4: touch, lift, both grippers, transferred")

TASKS = {t.name: t for t in (PUSHT, ALOHA_INSERTION, ALOHA_TRANSFER_CUBE)}
