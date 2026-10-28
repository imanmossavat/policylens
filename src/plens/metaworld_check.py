"""Meta-World disassembly tasks: does a trivial controller already solve them? (CLAUDE.md rule; plens-metaworld env)

Controllers, from dumb to expert:
  zero      no motion
  random    uniform random actions
  grab_pull go to the object with the scripted expert's grasp, then pull straight along one axis with the
            gripper closed (no alignment, no feedback): if this wins, the task does not test alignment
  expert    Meta-World's scripted policy
"""

from __future__ import annotations

from typing import Any

import numpy as np

TASKS = {"disassemble-v3": ("SawyerDisassembleV3Policy", np.array([0, 0, 1.0])),      # pull the ring up
         "peg-unplug-side-v3": ("SawyerPegUnplugSideV3Policy", np.array([1.0, 0, 0]))}  # pull the peg out sideways


def make(task: str, seed: int):
    import gymnasium as gym
    import metaworld  # noqa: F401  registers Meta-World/*

    env = gym.make("Meta-World/MT1", env_name=task, seed=seed)
    return env


def run(task: str, controller: str, n: int = 50, max_steps: int = 500, seed: int = 0) -> dict[str, Any]:
    import metaworld.policies as P

    expert = getattr(P, TASKS[task][0])()
    pull = TASKS[task][1]
    rng = np.random.default_rng(seed)
    wins, steps, switched = 0, [], 0
    for ep in range(n):
        env = make(task, seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        grasped_at, closing, grip = None, 0, 0.0
        for t in range(max_steps):
            if controller == "zero":
                a = np.zeros(4)
            elif controller == "random":
                a = rng.uniform(-1, 1, 4)
            elif controller == "expert":
                a = expert.get_action(obs)
            else:  # grab_pull: expert until it has commanded a grip for 10 steps, then a fixed pull, same grip
                a = expert.get_action(obs)
                closing = closing + 1 if a[3] > 0 else 0
                if grasped_at is None and closing >= 10:
                    grasped_at, grip = t, a[3]
                if grasped_at is not None:
                    a = np.concatenate([pull, [grip]])
            obs, _, term, trunc, info = env.step(np.clip(a, -1, 1))
            if info.get("success", 0) > 0.5:
                wins += 1
                steps.append(t + 1)
                break
            if term or trunc:
                break
        switched += grasped_at is not None
        env.close()
    return {"task": task, "controller": controller, "n": n, "wins": wins,
            # WHY: report it, so a grab_pull that never switched to pulling (== expert) is visible
            "switched_to_pull": switched if controller == "grab_pull" else None,
            "median_steps": float(np.median(steps)) if steps else None}
