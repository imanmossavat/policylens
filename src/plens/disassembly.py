"""ALOHA extraction: disassembly as the reverse of insertion (docs/tasks.md, diff:F).

No disassembly task or demonstrations exist for ALOHA, so they are made from insertion:
1. collect_assembled: run an insertion policy; at each success, freeze the full MuJoCo state
   (peg in socket, both grippers holding) together with the actions that led there.
2. AssembledStart: an env wrapper whose reset starts from one of those states. Success =
   peg and socket pulled at least EXTRACT_DISTANCE further apart than when assembled, out of
   contact, each still held by its gripper.
3. reverse_replay: a scripted baseline that plays the last insertion actions backwards.

What makes extraction hard: the parts are in contact and can jam; pulling one arm without
the other drags both parts along; and success needs each gripper to keep its grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.envs import preprocess_observation

from .envs import SOCKET_GEOMS, _aloha_physics, _contact_force, aloha_config, make_single_env
from .policy import PolicyBundle

EXTRACT_DISTANCE = 0.05  # m, beyond the assembled peg-socket distance
RIGHT_FINGER, LEFT_FINGER = "vx300s_right/10_right_gripper_finger", "vx300s_left/10_left_gripper_finger"
SOCKET_ONLY = SOCKET_GEOMS - {"pin"}


@dataclass
class Assembled:
    seed: int
    physics_state: np.ndarray  # physics.get_state(): qpos, qvel, act
    ctrl: np.ndarray
    actions: np.ndarray        # the insertion episode's actions, in order
    distance: float            # peg-socket centre distance when assembled


def _distance(ph) -> float:
    return float(np.linalg.norm(ph.named.data.xpos["peg"] - ph.named.data.xpos["socket"]))


def collect_assembled(bundle: PolicyBundle, seeds, max_steps: int = 400) -> list[Assembled]:
    """Run insertion episodes one at a time; keep the state at each success.

    WHY not run_batch: a vector env resets an env in the same step it succeeds (G-004),
    which would destroy exactly the state we want.
    """
    out = []
    env = make_single_env(aloha_config("AlohaInsertion-v0", max_steps)).envs[0]
    for seed in seeds:
        torch.manual_seed(seed)
        bundle.policy.reset()
        obs, _ = env.reset(seed=seed)
        actions = []
        for _ in range(max_steps):
            batch = preprocess_observation({k: (v[None] if not isinstance(v, dict) else {c: im[None] for c, im in v.items()})
                                            for k, v in obs.items()})
            batch["task"] = [""]
            batch = bundle.preprocessor(bundle.env_preprocessor(batch))
            with torch.inference_mode():
                a = bundle.policy.select_action(batch)
            a = bundle.env_postprocessor({"action": bundle.postprocessor(a)})["action"].to("cpu").numpy()[0]
            obs, reward, terminated, truncated, info = env.step(a)
            actions.append(a)
            if terminated or truncated:
                break
        if info.get("is_success"):
            ph = _aloha_physics(env)
            out.append(Assembled(seed, ph.get_state().copy(), ph.data.ctrl.copy(), np.array(actions), _distance(ph)))
    env.close()
    return out


def save_assembled(path: Path | str, items: list[Assembled]) -> None:
    np.savez_compressed(path, seed=[a.seed for a in items], distance=[a.distance for a in items],
                        physics_state=np.stack([a.physics_state for a in items]),
                        ctrl=np.stack([a.ctrl for a in items]),
                        **{f"actions_{i}": a.actions for i, a in enumerate(items)})


def load_assembled(path: Path | str) -> list[Assembled]:
    d = np.load(path)
    return [Assembled(int(d["seed"][i]), d["physics_state"][i], d["ctrl"][i], d[f"actions_{i}"],
                      float(d["distance"][i])) for i in range(len(d["seed"]))]


class AssembledStart(gym.Wrapper):
    """Start each episode from an assembled state (chosen by seed); reward = extraction progress 0..4."""

    max_reward = 4

    def __init__(self, env: gym.Env, assembled: list[Assembled]):
        super().__init__(env)
        self.assembled = assembled
        self.current: Assembled | None = None

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.current = self.assembled[(seed or 0) % len(self.assembled)]
        u = self.env.unwrapped
        ph = u._env.physics
        with ph.reset_context():
            ph.set_state(self.current.physics_state)
            ph.data.ctrl[:] = self.current.ctrl
        obs = u._format_raw_obs(u._env.task.get_observation(ph))
        return obs, {"is_success": False}

    def stage(self) -> int:
        """0 lost a part, 1 both held, 2 out of contact, 3 halfway, 4 extracted (success)."""
        ph = _aloha_physics(self.env)
        held_peg = _contact_force(ph, {"red_peg"}, {RIGHT_FINGER})[1] > 0
        held_socket = _contact_force(ph, SOCKET_ONLY, {LEFT_FINGER})[1] > 0
        if not (held_peg and held_socket):
            return 0
        apart = _contact_force(ph, {"red_peg"}, SOCKET_GEOMS)[1] == 0
        gained = _distance(ph) - self.current.distance
        if not apart:
            return 1
        return 4 if gained >= EXTRACT_DISTANCE else (3 if gained >= EXTRACT_DISTANCE / 2 else 2)

    def step(self, action):
        obs, _, _, truncated, info = self.env.step(action)
        reward = self.stage()
        info = {**info, "is_success": reward == self.max_reward}
        return obs, float(reward), reward == self.max_reward, truncated, info


def make_extraction_envs(assembled: list[Assembled], n_envs: int, max_steps: int = 200) -> gym.vector.VectorEnv:
    """n_envs extraction envs in a synchronous vector env with LeRobot's same-step autoreset."""
    cfg = aloha_config("AlohaInsertion-v0", max_steps)

    def one():
        return AssembledStart(make_single_env(cfg).envs[0], assembled)

    return gym.vector.SyncVectorEnv([one for _ in range(n_envs)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def reverse_replay(item: Assembled, steps: int) -> np.ndarray:
    """Scripted extraction: the last `steps` insertion actions, played backwards."""
    return item.actions[::-1][:steps]


# ---------------------------------------------------------------- demonstrations and evaluation

EXTRACT_TASK = "pull the peg out of the socket"
HOLD_STEPS = 25  # after extracting, keep still: demonstrations should also show stopping


def dataset_features(image_shape=(480, 640, 3)) -> dict:
    joints = [f"joint_{i}" for i in range(14)]
    return {"observation.images.top": {"dtype": "video", "shape": image_shape,
                                       "names": ["height", "width", "channel"]},
            "observation.state": {"dtype": "float32", "shape": (14,), "names": joints},
            "action": {"dtype": "float32", "shape": (14,), "names": joints}}


def record_demos(items: list[Assembled], root: Path | str, repo_id: str = "local/aloha_extraction",
                 replay_steps: int = 120, max_steps: int = 200) -> dict[str, int]:
    """Scripted extractions (reverse_replay) as a LeRobot dataset; only successful ones are kept."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(repo_id, fps=50, features=dataset_features(), root=root, use_videos=True)
    env = AssembledStart(make_single_env(aloha_config("AlohaInsertion-v0", max_steps)).envs[0], items)
    kept = 0
    for i, item in enumerate(items):
        obs, _ = env.reset(seed=i)
        plan = reverse_replay(item, replay_steps)
        frames, done_at = [], None
        for t in range(max_steps):
            a = plan[min(t, len(plan) - 1)] if done_at is None else frames[-1]["action"]
            frames.append({"observation.images.top": obs["pixels"]["top"],
                           "observation.state": obs["agent_pos"].astype(np.float32),
                           "action": np.asarray(a, dtype=np.float32), "task": EXTRACT_TASK})
            obs, r, _, _, _ = env.step(a)
            if done_at is None and r == AssembledStart.max_reward:
                done_at = t
            if done_at is not None and t - done_at >= HOLD_STEPS:
                break
        if done_at is not None:
            for f in frames:
                ds.add_frame(f)
            ds.save_episode()
            kept += 1
    ds.finalize()
    env.close()
    return {"demos": kept, "attempted": len(items)}


def extraction_task():
    """TaskSpec for evaluating a learned extraction policy with rollout.run_batch."""
    from .envs import TaskSpec, observe_insertion

    return TaskSpec("aloha_extraction", lambda n: aloha_config("AlohaInsertion-v0", n), 200, "",
                    progress=lambda r, info, i: float(r) / AssembledStart.max_reward,
                    observe=observe_insertion, near_miss=0.75,
                    note="progress = extraction stage / 4: held, out of contact, halfway, extracted")


def evaluate_extraction(bundle: PolicyBundle, items: list[Assembled], max_steps: int = 200, batch: int = 25):
    """A learned policy on extraction episodes starting from each assembled state once."""
    from .rollout import run_batch

    task = extraction_task()
    eps = []
    for first in range(0, len(items), batch):
        idx = list(range(first, min(first + batch, len(items))))
        venv = make_extraction_envs(items, len(idx), max_steps)
        eps += run_batch(bundle, venv, idx, task=task)
        venv.close()
    return eps


def glide_home(item: Assembled, steps: int) -> np.ndarray:
    """Trivial extractor: move both arms linearly to their start pose, grippers kept closed.

    If this extracts, the task does not test disassembly skill (F-009): nothing constrains
    the pull direction, so any motion apart succeeds.
    """
    from gym_aloha.constants import START_ARM_POSE

    last = item.actions[-1]
    home = np.r_[START_ARM_POSE[:6], last[6], START_ARM_POSE[8:14], last[13]]  # 6 joints + gripper, per arm
    alpha = (np.arange(1, steps + 1) / steps)[:, None]
    return last + alpha * (home - last)
