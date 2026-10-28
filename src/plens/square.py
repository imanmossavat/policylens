"""robosuite Square (nut on peg) with DPPO's pretrained state-based diffusion policy (D-014).

Runs in the plens-dppo env: robosuite 1.4.1 + mujoco 3.1.6 + torch 2.4, the versions the
checkpoint was trained with (DPPO pyproject and cfg/robomimic/env_meta/square.json).
Assets (checkpoint, normalization, demos, DPPO source) are listed in docs/assets.md.

Pieces:
  make_env        robosuite env with the exact kwargs of DPPO's env_meta/square.json
  obs_vector      the 23 numbers the policy sees: eef pos (3), eef quat (4), gripper (2), object (14)
  Normalizer      min/max scaling to [-1, 1] from DPPO's normalization.npz
  load_policy     DPPO Unet1D + DDPM sampler (20 denoising passes, plan of 4 actions)
  rollout_batch   N envs in lockstep, one batched policy call per re-plan; traces incl. wrist force
  replay_demos    open-loop replay of robomimic demo actions from their recorded start states
  render_states   re-render saved sim states to frames (no renderer needed during rollouts)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DPPO_SRC = ROOT / ".cache/dppo/src"
CKPT = ROOT / ".cache/dppo/ckpt_square_unet/state_8000.pt"
NORM = ROOT / ".cache/dppo/norm_square/normalization.npz"
DEMOS = ROOT / ".cache/robomimic/square_mh/low_dim_v141.hdf5"

# Copied from DPPO cfg/robomimic/env_meta/square.json (env_version 1.4.1); robomimic forces the
# renderer flags, ignore_done and use_object_obs the same way (robomimic v0.3 env_robosuite.py).
ENV_KWARGS: dict[str, Any] = {
    "robots": ["Panda"], "control_freq": 20, "reward_shaping": False,
    "has_renderer": False, "has_offscreen_renderer": False, "ignore_done": True,
    "use_object_obs": True, "use_camera_obs": False, "camera_depths": False,
    "controller_configs": {
        "type": "OSC_POSE", "input_max": 1, "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150, "damping": 1, "impedance_mode": "fixed", "kp_limits": [0, 300],
        "damping_limits": [0, 10], "position_limits": None, "orientation_limits": None,
        "uncouple_pos_ori": True, "control_delta": True, "interpolation": None, "ramp_ratio": 0.2,
    },
}
OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state")
# DPPO checkpoint config (cfg/robomimic/eval/square/eval_diffusion_unet.yaml)
POLICY_CFG = {"obs_dim": 23, "action_dim": 7, "horizon_steps": 4, "denoising_steps": 20}
UNET_CFG = {"diffusion_step_embed_dim": 16, "dim": 64, "dim_mults": [1, 2], "kernel_size": 5,
            "n_groups": 8, "smaller_encoder": False, "cond_predict_scale": True}
MAX_STEPS = 400  # DPPO eval max_episode_steps
TABLE_Z = 0.82   # env.table_offset[2]; a resting nut sits at ~0.83
# GOTCHA(G-014): the nut is placed ~6 cm above the table and drops in the first steps, so measure
# lifting from the table, never from the nut's height at reset.
LIFT_Z = TABLE_Z + 0.04  # lifted = nut 3 cm above resting height
SETTLE_STEPS = 10  # nut has landed by then (it lands in ~5)
NEAR_XY_M = 0.03  # same xy tolerance robosuite's on_peg uses for success


def make_env(**overrides: Any):
    import robosuite as suite

    return suite.make("NutAssemblySquare", **{**ENV_KWARGS, **overrides})


def obs_vector(raw: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(raw[k], dtype=np.float64) for k in OBS_KEYS])


class Normalizer:
    """DPPO's RobomimicLowdimWrapper scaling: obs and actions <-> [-1, 1] by dataset min/max."""

    def __init__(self, path: Path | str = NORM):
        d = np.load(path)
        self.obs_min, self.obs_max = d["obs_min"], d["obs_max"]
        self.act_min, self.act_max = d["action_min"], d["action_max"]

    def obs(self, x: np.ndarray) -> np.ndarray:
        return 2 * ((x - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def action(self, a: np.ndarray) -> np.ndarray:
        return (a + 1) / 2 * (self.act_max - self.act_min) + self.act_min


def load_policy(ckpt: Path | str | None = CKPT, device: str = "cpu", weights: str = "ema"):
    """DPPO DiffusionModel with the pretrained weights, loaded strictly.

    weights="ema" is what DPPO fine-tuning starts from (DiffusionModel loads "ema" when present);
    DPPO's own eval script (DiffusionEval) loads "model" instead.
    """
    import torch

    if str(DPPO_SRC) not in sys.path:
        sys.path.insert(0, str(DPPO_SRC))
    from model.diffusion.diffusion import DiffusionModel
    from model.diffusion.unet import Unet1D

    class _Model(DiffusionModel):
        # GOTCHA(G-012): DPPO's DiffusionModel.forward passes deterministic= to its own p_mean_var,
        # which does not take it (only DiffusionEval's override does).
        def p_mean_var(self, x, t, cond, index=None, deterministic=False):
            return super().p_mean_var(x, t, cond, index=index)

    net = Unet1D(cond_dim=POLICY_CFG["obs_dim"], action_dim=POLICY_CFG["action_dim"], **UNET_CFG)
    model = _Model(net,device=device, predict_epsilon=True, denoised_clip_value=1.0,
                           randn_clip_value=3, **POLICY_CFG)
    if ckpt is None:  # untrained, for plens.train_square
        return model
    state = torch.load(ckpt, map_location=device, weights_only=True)[weights]
    # WHY: DPPO loads with strict=False, which would silently run random weights on a key mismatch.
    model.load_state_dict(state, strict=True)
    return model.eval()


@dataclass
class SquareEpisode:
    seed: int
    success: bool = False
    success_step: int | None = None
    lifted: bool = False
    over_peg: bool = False
    trace: dict[str, list] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Coarse stage as a number, so labels.summarize works: 0 start, .5 lifted, .8 over peg, 1 done."""
        return 1.0 if self.success else 0.8 if self.over_peg else 0.5 if self.lifted else 0.0

    def summary(self) -> dict[str, Any]:
        return {"seed": self.seed, "success": self.success, "success_step": self.success_step,
                "n_steps": len(self.trace.get("action", [])), "lifted": self.lifted,
                "over_peg": self.over_peg, "max_coverage": self.progress}

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.trace.items()}


def _nut_peg(env) -> tuple[np.ndarray, np.ndarray]:
    nut = np.array(env.sim.data.body_xpos[env.obj_body_id[env.nuts[0].name]])
    peg = np.array(env.sim.data.body_xpos[env.peg1_body_id])  # Square nut goes on peg1
    return nut, peg


def stage_flags(nut: np.ndarray, peg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per step (or for one step): nut lifted off the table; nut centre within NEAR_XY_M of the peg in x and y.

    Works on saved traces too (trace/nut_pos, trace/peg_pos), so stages can be recomputed for old runs;
    ignore lifted for the first SETTLE_STEPS records (the nut starts in the air, G-014).
    """
    return nut[..., 2] > LIFT_Z, np.all(np.abs(nut[..., :2] - peg[..., :2]) < NEAR_XY_M, axis=-1)


def nut_peg_contact(env) -> tuple[float, int]:
    """Summed normal contact force (N) and contact count between the square nut and its peg (MuJoCo contacts)."""
    import mujoco

    m, d = env.sim.model._model, env.sim.data._data
    nut, peg = env.obj_body_id[env.nuts[0].name], env.peg1_body_id
    total, n, f6 = 0.0, 0, np.zeros(6)
    for i in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        if {b1, b2} == {nut, peg}:
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(f6[0])
            n += 1
    return total, n


def _record(env, ep: SquareEpisode, action: np.ndarray | None) -> None:
    robot = env.robots[0]
    nut, peg = _nut_peg(env)
    row = {
        "sim_state": env.sim.get_state().flatten(),
        "nut_pos": nut, "peg_pos": peg,
        "eef_pos": np.array(env.sim.data.site_xpos[robot.eef_site_id]),  # world frame, like nut/peg "gripper_qpos": np.array(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]),
        "wrist_force": np.array(robot.ee_force), "wrist_torque": np.array(robot.ee_torque),
    }
    row["nut_peg_force"], row["nut_peg_contacts"] = nut_peg_contact(env)
    if action is not None:
        row["action"] = action
    for k, v in row.items():
        ep.trace.setdefault(k, []).append(v)
    lifted, over_peg = stage_flags(nut, peg)
    ep.lifted |= bool(lifted) and len(ep.trace["nut_pos"]) > SETTLE_STEPS
    ep.over_peg |= bool(over_peg)


def rollout_batch(model, envs: Sequence, seeds: Sequence[int], norm: Normalizer | None = None,
                  noise_seed: int | None = None, max_steps: int = MAX_STEPS,
                  act_steps: int = POLICY_CFG["horizon_steps"]) -> list[SquareEpisode]:
    """Run len(seeds) episodes in lockstep; one batched policy call per re-plan of act_steps actions.

    Start layout: np.random.seed(seed) then env.reset(), as DPPO's wrapper does. A finished env
    (success) stops stepping but stays in the batch. Sampling noise comes from torch's global
    generator, seeded once per batch, so a seed's episode depends on its batch (as G-008).
    """
    import torch

    assert len(envs) >= len(seeds)
    norm = norm or Normalizer()
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    eps, obs = [], []
    for env, s in zip(envs, seeds):
        np.random.seed(s)
        raw = env.reset()
        ep = SquareEpisode(seed=s)
        _record(env, ep, None)
        eps.append(ep)
        obs.append(obs_vector(raw))
    device = next(model.parameters()).device
    active = list(range(len(seeds)))
    t = 0
    while active and t < max_steps:
        cond = torch.tensor(norm.obs(np.stack([obs[i] for i in active])), dtype=torch.float32,
                            device=device)[:, None, :]
        plans = norm.action(model({"state": cond}).trajectories.cpu().numpy())  # (B, 4, 7)
        still = []
        for b, i in enumerate(active):
            env, ep = envs[i], eps[i]
            for k in range(min(act_steps, max_steps - t)):
                raw, reward, _, _ = env.step(plans[b, k])
                _record(env, ep, plans[b, k])
                if reward >= 1:  # DPPO: best_reward_threshold_for_success = 1
                    ep.success, ep.success_step = True, t + k
                    break
            obs[i] = obs_vector(raw)
            if not ep.success:
                still.append(i)
        active = still
        t += act_steps
    return eps


def replay_demos(path: Path | str = DEMOS, n: int | None = None, env=None) -> list[dict[str, Any]]:
    """Replay each demo's recorded actions open loop from its recorded first sim state.

    Returns per demo: success (any step), and the largest object-position drift from the recorded
    states. Mostly-successful replays mean our env matches the one the demos (and policy) came from.
    """
    import h5py

    env = env or make_env()
    out = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"], key=lambda s: int(s.split("_")[1]))[:n]
        for name in names:
            g = f["data"][name]
            states, actions = g["states"][()], g["actions"][()]
            env.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            success, drift = False, 0.0
            for t, a in enumerate(actions):
                _, reward, _, _ = env.step(a)
                success |= reward >= 1
                if t + 1 < len(states):
                    drift = max(drift, float(np.abs(env.sim.get_state().flatten()[1:] - states[t + 1][1:]).max()))
            out.append({"demo": name, "success": bool(success), "n_steps": len(actions),
                        "max_state_drift": round(drift, 4)})
    return out


def render_states(states: np.ndarray, camera: str = "agentview", size: int = 256, every: int = 1) -> list[np.ndarray]:
    """Frames for saved sim states (trace/sim_state), via mujoco.Renderer on the env's model."""
    import mujoco

    env = make_env()
    env.reset()
    m, d = env.sim.model._model, env.sim.data._data
    renderer = mujoco.Renderer(m, size, size)
    frames = []
    for s in states[::every]:
        env.sim.set_state_from_flattened(s)
        env.sim.forward()
        renderer.update_scene(d, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


CONTACT_N = 20.0  # wrist force above this = pressing on something (free carry with the nut is ~5 N)


def peg_phase_force(trace: dict[str, np.ndarray], early_steps: int = 20) -> dict[str, Any] | None:
    """Wrist force once the nut first gets over the peg; None if it never does.

    early_*: only the first early_steps steps there (1 s), i.e. before the outcome is usually visible.
    """
    nut, peg, f = trace["nut_pos"], trace["peg_pos"], np.linalg.norm(trace["wrist_force"], axis=1)
    over = stage_flags(nut, peg)[1]
    if not over.any():
        return None
    t0 = int(np.argmax(over))
    ph, early = f[t0:], f[t0:t0 + early_steps]
    return {"t_over_peg": t0, "steps_at_peg": len(ph), "median_n": float(np.median(ph)), "max_n": float(ph.max()),
            "contact_frac": float((ph > CONTACT_N).mean()), "early_max_n": float(early.max()),
            "early_contact_frac": float((early > CONTACT_N).mean())}


def contact_from_states(states: np.ndarray, env=None) -> np.ndarray:
    """Nut-peg contact force (N) per saved sim state; recovers it for runs recorded before it was traced."""
    env = env or make_env()
    env.reset()
    out = np.zeros(len(states))
    for i, s in enumerate(states):
        env.sim.set_state_from_flattened(s)
        env.sim.forward()
        out[i] = nut_peg_contact(env)[0]
    return out
