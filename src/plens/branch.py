"""Branch an episode: copy the simulator at a step and run many continuations from it.

The question this answers (Q-011): is an episode's outcome decided at identifiable
moments, or spread over the whole episode? From a saved state, K continuations with
fresh sampling noise estimate p(success | state); along an episode that is its fate curve.

A branch must restore everything the future depends on: the physics (positions,
velocities and the solver's contact cache), the time-limit counter, and the policy's
observation history. gym-pusht's reset_to_state restores none of the hidden parts (G-009).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lerobot.envs import preprocess_observation
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES

from .policy import PolicyBundle
from .envs import PUSHT
from .rollout import Episode


@dataclass
class Snapshot:
    """A PushT env frozen at one step, plus the last n_obs_steps raw observations."""
    bodies: tuple[Any, Any, Any]  # (pymunk.Space, agent, block), copied together (handlers detached)
    kinematics: dict[str, tuple]  # per body: cog, position, angle, velocity, angular velocity
    elapsed_steps: int
    attrs: dict[str, Any]      # small scalars the env carries between steps
    obs_history: list[dict[str, np.ndarray]]
    step: int
    rng: dict[str, torch.Tensor] | None = None  # torch RNG state, to replay the original exactly


def rng_state() -> dict[str, torch.Tensor]:
    state = {"cpu": torch.random.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, torch.Tensor]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _pusht(env):
    return env.unwrapped


def snapshot_pusht(env, obs_history: Sequence[dict[str, np.ndarray]], step: int) -> Snapshot:
    """Freeze one gym-pusht env (a TimeLimit-wrapped env from a vector env's .envs)."""
    u = _pusht(env)
    # GOTCHA(G-009): pymunk cannot copy a space with Python collision callbacks attached;
    # detach them for the copy, re-attach on restore.
    # WHY one deepcopy of the tuple: a copied space does not keep its body order, so the
    # agent and block must be copied in the same call to stay linked to their copies.
    handlers, u.space._handlers = u.space._handlers, {}
    try:
        bodies = copy.deepcopy((u.space, u.agent, u.block))
    finally:
        u.space._handlers = handlers
    kin = {name: (tuple(b.center_of_gravity), tuple(b.position), b.angle, tuple(b.velocity), b.angular_velocity)
           for name, b in (("agent", u.agent), ("block", u.block))}
    return Snapshot(bodies=bodies, kinematics=kin, elapsed_steps=env._elapsed_steps,
                    attrs={"n_contact_points": u.n_contact_points,
                           "_last_action": getattr(u, "_last_action", None)},
                    obs_history=[{k: np.copy(v) for k, v in o.items()} for o in obs_history], step=step,
                    rng=rng_state())


def restore_pusht(env, snap: Snapshot) -> None:
    """Put a (reset) gym-pusht env into the snapshot's exact state."""
    u = _pusht(env)
    u.space, u.agent, u.block = copy.deepcopy(snap.bodies)
    # GOTCHA(G-009): an unpickled pymunk body gets its position before its centre of gravity,
    # which shifts the block by its CoG offset on the next step. Re-set, CoG first.
    for name, body in (("agent", u.agent), ("block", u.block)):
        cog, pos, angle, vel, ang_vel = snap.kinematics[name]
        body.center_of_gravity, body.position, body.angle = cog, pos, angle
        body.velocity, body.angular_velocity = vel, ang_vel
    u._block_shapes = list(u.block.shapes)
    u.collision_handeler = u.space.add_collision_handler(0, 0)  # (sic) gym-pusht's name
    u.collision_handeler.post_solve = u._handle_collision
    for k, v in snap.attrs.items():
        setattr(u, k, v)
    env._elapsed_steps = snap.elapsed_steps


def _stack(obs_list: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _fill_obs_queue(bundle: PolicyBundle, history: Sequence[dict[str, np.ndarray]], n: int) -> None:
    """Load the policy's observation queue with the snapshot's history, for n parallel branches.

    select_action would re-plan on its first call; here we only populate the queue, exactly
    as the original episode's earlier select_action calls did.
    """
    policy = bundle.policy
    policy.reset()
    for obs in history:
        batch = preprocess_observation(_stack([obs] * n))
        batch["task"] = [""] * n
        batch = bundle.preprocessor(bundle.env_preprocessor(batch))
        if policy.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
        # WHY exclude ACTION: the processed batch carries an empty action key; select_action pops it
        # before queueing, and so must we, or the action queue fills with None.
        policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])


def run_branches(bundle: PolicyBundle, venv, snap: Snapshot, noise_seed: int | None,
                 first_plan: bool = True) -> tuple[list[Episode], np.ndarray | None]:
    """Run venv.num_envs continuations from snap, each with its own sampling noise.

    Returns the continuation episodes (steps counted from the snapshot) and, if first_plan,
    the first plan each branch made: [n_branches, n_action_steps, action_dim], in env units.
    noise_seed=None restores the original episode's RNG state instead (one branch then
    replays the original exactly; used to verify a snapshot).
    """
    n = venv.num_envs
    venv.reset(seed=list(range(n)))  # builds fresh envs; their state is replaced next
    for env in venv.envs:
        restore_pusht(env, snap)
    obs = _stack([snap.obs_history[-1]] * n)
    _fill_obs_queue(bundle, snap.obs_history[:-1], n)
    if noise_seed is None:
        set_rng_state(snap.rng)
    else:
        torch.manual_seed(noise_seed)

    traces = [{k: [] for k in ("action", "reward", "progress", *PUSHT.trace_keys)} for _ in range(n)]
    success, n_steps, active = [False] * n, [0] * n, [True] * n
    plan = []
    step = 0
    while any(active):
        batch = preprocess_observation(obs)
        batch["task"] = [""] * n
        batch = bundle.preprocessor(bundle.env_preprocessor(batch))
        with torch.inference_mode():
            action = bundle.policy.select_action(batch)
        action = bundle.env_postprocessor({"action": bundle.postprocessor(action)})["action"].to("cpu").numpy()
        if first_plan and step < bundle.policy.config.n_action_steps:
            plan.append(action)
        obs, reward, terminated, truncated, info = venv.step(action)
        for i in range(n):
            if not active[i]:
                continue
            done = bool(terminated[i] or truncated[i])
            src = info["final_info"] if done else info  # GOTCHA(G-004)
            traces[i]["action"].append(action[i])
            traces[i]["reward"].append(float(reward[i]))
            for k in PUSHT.trace_keys:
                traces[i][k].append(np.asarray(src[k][i]))
            traces[i]["progress"].append(PUSHT.progress(float(reward[i]), src, i))
            success[i] = success[i] or bool(src["is_success"][i])
            n_steps[i] = step + 1
            active[i] = not done
        step += 1

    eps = []
    for i in range(n):
        arrays = {k: np.asarray(v) for k, v in traces[i].items()}
        cov = arrays["coverage"].astype(float)
        eps.append(Episode(seed=-1, success=success[i], n_steps=n_steps[i], max_coverage=float(cov.max()),
                           final_coverage=float(cov[-1]), trace=arrays))
    plans = np.stack(plan, axis=1) if plan else None
    return eps, plans


class SnapshotTaker:
    """on_step callback for rollout.run_batch (one-env venv): snapshots at every re-plan step.

    every: snapshot each `every` steps; use the policy's n_action_steps so each snapshot
    falls where the original re-planned (its action queue is empty there).
    """

    def __init__(self, n_obs_steps: int, every: int, n_action_steps: int | None = None):
        if n_action_steps and every % n_action_steps:
            # WHY: between re-plans the parent still holds queued actions that a branch would not
            raise ValueError(f"every={every} must be a multiple of n_action_steps={n_action_steps}")
        self.n_obs_steps, self.every = n_obs_steps, every
        self.history: list[dict[str, np.ndarray]] = []
        self.snapshots: list[Snapshot] = []

    def __call__(self, step: int, venv, obs: dict[str, np.ndarray]) -> None:
        self.history = (self.history + [{k: v[0] for k, v in obs.items()}])[-self.n_obs_steps:]
        if step % self.every == 0:
            self.snapshots.append(snapshot_pusht(venv.envs[0], self.history, step))


def plan_spread(plans: np.ndarray) -> float:
    """RMS distance (env units, pixels for PushT) of K sampled plans from their mean plan.

    plans: [K, steps, action_dim]. A cheap uncertainty signal: how much the policy's own
    samples disagree about what to do next from this state.
    """
    return float(np.sqrt(((plans - plans.mean(axis=0)) ** 2).sum(axis=-1).mean()))


def fate_curve(bundle: PolicyBundle, env_cfg, seed: int, n_branches: int, every: int,
               noise_seed: int = 0) -> tuple[Episode, list[dict[str, Any]]]:
    """Run one episode, then from every `every`-th step run n_branches continuations.

    Returns the parent episode and one row per snapshot: step, successes out of n_branches,
    the plan spread of the branches' first plans, and the parent's state there. Branch noise
    is seeded with noise_seed + step, so rows are reproducible.
    """
    from .envs import make_envs, make_single_env
    from .rollout import run_batch
    from .stats import wilson

    cfg = bundle.policy.config
    taker = SnapshotTaker(cfg.n_obs_steps, every, cfg.n_action_steps)
    parent = run_batch(bundle, make_single_env(env_cfg), [seed], on_step=taker)[0]
    venv = make_envs(env_cfg, n_branches)
    rows = []
    for snap in taker.snapshots:
        eps, plans = run_branches(bundle, venv, snap, noise_seed + snap.step)
        k = sum(e.success for e in eps)
        lo, hi = wilson(k, n_branches)
        s = snap.step
        rows.append({"seed": seed, "step": s, "k": k, "n": n_branches, "p": k / n_branches,
                     "ci95": [round(lo, 4), round(hi, 4)], "plan_spread": round(plan_spread(plans), 3),
                     "mean_max_cov": round(float(np.mean([e.max_coverage for e in eps])), 4),
                     "parent_coverage": round(float(parent.trace["coverage"][max(s - 1, 0)]), 4),
                     "parent_contacts": int(parent.trace["n_contacts"][max(s - 1, 0)])})
    venv.close()
    return parent, rows


# ---------------------------------------------------------------- reading fate runs

def load_fate(run_dir) -> dict[int, dict[str, Any]]:
    """{parent idx: {"seed", "label", "step", "p", "k", "n", "plan_spread", ...arrays}} from a fate run."""
    import json
    from pathlib import Path

    from . import store

    run_dir = Path(run_dir)
    labels = {line["idx"]: line for line in store.read_index(run_dir)}
    rows: dict[int, list[dict]] = {}
    for line in (run_dir / "fate.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["idx"], []).append(r)
    out = {}
    for idx, rs in rows.items():
        rs.sort(key=lambda r: r["step"])
        out[idx] = {"seed": rs[0]["seed"], "label": labels[idx]["label"] if idx in labels else "?",
                    "parent_steps": labels[idx]["n_steps"] if idx in labels else None,
                    **{k: np.array([r[k] for r in rs]) for k in
                       ("step", "p", "k", "n", "plan_spread", "parent_coverage", "parent_contacts")}}
    return out


def fate_stats(curve: dict[str, Any], sealed: float = 0.1) -> dict[str, Any]:
    """How an episode's fate was decided, from its fate curve.

    largest_drop / largest_rise: biggest change of p between consecutive snapshots.
    decisiveness: |largest change| / sum of |changes|; near 1 = one moment decides,
      near 1/n_intervals = spread evenly. Binomial noise in p (n branches) inflates the sum,
      so compare against noise_floor_sum: the sum expected from noise alone.
    sealed_step: first snapshot after which p never leaves [0, sealed] or [1 - sealed, 1].
    """
    p, steps, n = curve["p"], curve["step"], curve["n"]
    d = np.diff(p)
    sealed_step = None
    for j in range(len(p)):
        rest = p[j:]
        if (rest <= sealed).all() or (rest >= 1 - sealed).all():
            sealed_step = int(steps[j])
            break
    big = int(np.argmax(np.abs(d))) if len(d) else 0
    noise = float(np.sum(np.sqrt(2 * p[:-1] * (1 - p[:-1]) / n[:-1]) * np.sqrt(2 / np.pi))) if len(d) else 0.0
    return {"p0": float(p[0]), "p_end": float(p[-1]), "largest_change": float(d[big]) if len(d) else 0.0,
            "largest_change_step": int(steps[big + 1]) if len(d) else None,
            "sum_abs_change": float(np.abs(d).sum()), "noise_floor_sum": noise,
            "decisiveness": float(abs(d[big]) / np.abs(d).sum()) if len(d) and np.abs(d).sum() else 0.0,
            "sealed_step": sealed_step}
