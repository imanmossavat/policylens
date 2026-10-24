"""End-to-end with the real env and the real architecture (random weights, offline).

The legacy-checkpoint test fakes the old Hub format (G-001) so the migration path
is exercised without network access.
"""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from plens import store
from plens.envs import make_single_env, pusht_config
from plens.policy import DIFFUSION_PUSHT_LAYERS as LAYERS
from plens.policy import load_pretrained, load_random_diffusion, resolve_checkpoint
from plens.record import ActivationRecorder
from plens.rollout import run_episode

pytestmark = pytest.mark.smoke
MAX_STEPS = 12


@pytest.fixture(scope="module")
def env_cfg():
    return pusht_config(MAX_STEPS)


@pytest.fixture(scope="module")
def bundle(env_cfg):
    return load_random_diffusion(env_cfg, "cpu", num_inference_steps=3)


def test_episode_trace_and_activations(bundle, env_cfg):
    venv = make_single_env(env_cfg)
    with ActivationRecorder(bundle.policy, LAYERS) as rec:
        ep = run_episode(bundle, venv, seed=0, recorder=rec, render=True)
    venv.close()
    assert ep.n_steps == MAX_STEPS
    assert ep.trace["action"].shape == (MAX_STEPS, 2)
    assert ep.trace["block_pose"].shape == (MAX_STEPS, 3)
    assert len(ep.frames) == MAX_STEPS
    acts = rec.to_arrays()
    # Re-planning every n_action_steps: the image encoder fires on steps 0, 8, ...
    n_action_steps = bundle.policy.config.n_action_steps
    assert set(acts["diffusion.rgb_encoder"]["step"]) == set(range(0, MAX_STEPS, n_action_steps))
    # The denoiser fires num_inference_steps times per re-plan.
    unet = acts["diffusion.unet.mid_modules.1"]
    assert unet["call"].max() == bundle.policy.config.num_inference_steps - 1


def test_same_seed_same_episode(bundle, env_cfg):
    venv = make_single_env(env_cfg)
    a = run_episode(bundle, venv, seed=5)
    b = run_episode(bundle, venv, seed=5)
    venv.close()
    np.testing.assert_allclose(a.trace["action"], b.trace["action"])


# config.json of lerobot/diffusion_pusht at Hub commit 84a7c23 (checked 2026-10-20),
# except num_inference_steps (null -> 100 there) lowered to keep the test fast.
HUB_CONFIG = {
    "beta_end": 0.02, "beta_schedule": "squaredcos_cap_v2", "beta_start": 0.0001,
    "clip_sample": True, "clip_sample_range": 1.0, "crop_is_random": True, "crop_shape": [84, 84],
    "diffusion_step_embed_dim": 128, "do_mask_loss_for_padding": False,
    "down_dims": [512, 1024, 2048], "drop_n_last_frames": 7, "horizon": 16,
    "input_features": {"observation.image": {"shape": [3, 96, 96], "type": "VISUAL"},
                       "observation.state": {"shape": [2], "type": "STATE"}},
    "kernel_size": 5, "n_action_steps": 8, "n_groups": 8, "n_obs_steps": 2,
    "device": "cuda", "use_amp": False, "noise_scheduler_type": "DDPM",
    "normalization_mapping": {"ACTION": "MIN_MAX", "STATE": "MIN_MAX", "VISUAL": "MEAN_STD"},
    "num_inference_steps": 3, "num_train_timesteps": 100, "optimizer_betas": [0.95, 0.999],
    "optimizer_eps": 1e-08, "optimizer_lr": 0.0001, "optimizer_weight_decay": 1e-06,
    "output_features": {"action": {"shape": [2], "type": "ACTION"}},
    "prediction_type": "epsilon", "pretrained_backbone_weights": None, "scheduler_name": "cosine",
    "scheduler_warmup_steps": 500, "spatial_softmax_num_keypoints": 32, "type": "diffusion",
    "use_film_scale_modulation": True, "use_group_norm": True,
    "use_separate_rgb_encoder_per_camera": False, "vision_backbone": "resnet18",
}


def test_legacy_checkpoint_is_migrated_and_runs(tmp_path, env_cfg):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "config.json").write_text(json.dumps(HUB_CONFIG))
    arch = PreTrainedConfig.from_pretrained(legacy)  # exactly the Hub architecture
    arch.device = "cpu"
    # Old format: normalization stats stored as buffers inside the weights.
    sd = DiffusionPolicy(arch).state_dict()
    sd.update({
        "normalize_inputs.buffer_observation_image.mean": torch.full((3, 1, 1), 0.5),
        "normalize_inputs.buffer_observation_image.std": torch.full((3, 1, 1), 0.25),
        "normalize_inputs.buffer_observation_state.min": torch.zeros(2),
        "normalize_inputs.buffer_observation_state.max": torch.full((2,), 512.0),
        "normalize_targets.buffer_action.min": torch.zeros(2),
        "normalize_targets.buffer_action.max": torch.full((2,), 512.0),
        "unnormalize_outputs.buffer_action.min": torch.zeros(2),
        "unnormalize_outputs.buffer_action.max": torch.full((2,), 512.0),
    })
    save_file({k: v.contiguous() for k, v in sd.items()}, legacy / "model.safetensors")

    local, info = resolve_checkpoint(str(legacy), cache=tmp_path / "cache")
    assert info["migrated"] and any(local.glob("policy_preprocessor*.json"))
    _, again = resolve_checkpoint(str(legacy), cache=tmp_path / "cache")  # cached, no re-migration
    assert again["loaded_from"] == info["loaded_from"]

    b = load_pretrained(str(legacy), env_cfg, "cpu")
    ActivationRecorder(b.policy, LAYERS)  # default layer names exist in the Hub architecture
    venv = make_single_env(env_cfg)
    ep = run_episode(b, venv, seed=0)
    venv.close()
    # Unnormalized actions land in PushT's 0..512 pixel range.
    assert ep.trace["action"].min() >= 0 and ep.trace["action"].max() <= 512


def test_saved_run_reloads(tmp_path, bundle, env_cfg):
    venv = make_single_env(env_cfg)
    with ActivationRecorder(bundle.policy, LAYERS) as rec:
        ep = run_episode(bundle, venv, seed=1, recorder=rec)
    venv.close()
    store.save_episode(tmp_path, 0, ep, "fail", rec.to_arrays())
    loaded = store.load_episode(tmp_path, 0)
    assert set(loaded["act"]) == set(LAYERS)
    assert loaded["trace"]["action"].shape == (MAX_STEPS, 2)


def test_batch_of_one_equals_run_episode(bundle, env_cfg):
    from plens.rollout import run_batch

    venv = make_single_env(env_cfg)
    a = run_episode(bundle, venv, seed=3)
    b = run_batch(bundle, venv, [3])[0]
    venv.close()
    np.testing.assert_array_equal(a.trace["action"], b.trace["action"])


def test_batched_rollout_records_each_env(bundle, env_cfg):
    from plens.envs import make_envs
    from plens.record import select_env
    from plens.rollout import run_batch

    venv = make_envs(env_cfg, 3)
    with ActivationRecorder(bundle.policy, LAYERS) as rec:
        eps = run_batch(bundle, venv, [10, 11, 12], rec, render=(1,))
    venv.close()
    assert [ep.seed for ep in eps] == [10, 11, 12] and all(ep.n_steps == MAX_STEPS for ep in eps)
    assert len(eps[1].frames) == MAX_STEPS and not eps[0].frames
    # Different seeds start in different states, so the envs really are separate.
    assert not np.allclose(eps[0].trace["block_pose"][0], eps[1].trace["block_pose"][0])
    one = select_env(rec.to_arrays(), env=2, n_envs=3, n_steps=eps[2].n_steps)
    n_obs = bundle.policy.config.n_obs_steps
    assert one["diffusion.rgb_encoder"]["value"].shape[1] == n_obs
    assert one["diffusion.unet.mid_modules.1"]["value"].shape[1] == 1


def test_set_sampler_switches_to_ddim(env_cfg):
    from plens.policy import set_sampler

    b = load_random_diffusion(env_cfg, "cpu", num_inference_steps=3)
    s = set_sampler(b, "ddim", steps=2, n_action_steps=4)
    assert s["sampler"] == "ddim" and s["steps"] == 2 and s["n_action_steps"] == 4
    venv = make_single_env(env_cfg)
    with ActivationRecorder(b.policy, LAYERS) as rec:
        run_episode(b, venv, seed=0, recorder=rec)
    venv.close()
    unet = rec.to_arrays()["diffusion.unet.mid_modules.1"]
    assert unet["call"].max() == 1 and set(unet["step"]) == set(range(0, MAX_STEPS, 4))
    with pytest.raises(ValueError):
        set_sampler(b, n_action_steps=99)


def test_branch_replays_parent_exactly_and_runs_k_branches(env_cfg):
    from plens.branch import SnapshotTaker, fate_curve, run_branches
    from plens.envs import make_envs
    from plens.policy import set_sampler
    from plens.rollout import run_batch

    bundle = load_random_diffusion(env_cfg, "cpu", num_inference_steps=3)
    set_sampler(bundle, n_action_steps=4)  # 3 re-plans in a 12-step episode
    cfg = bundle.policy.config
    taker = SnapshotTaker(cfg.n_obs_steps, cfg.n_action_steps, cfg.n_action_steps)
    parent = run_batch(bundle, make_single_env(env_cfg), [2], on_step=taker)[0]
    snap = taker.snapshots[1]  # the second re-plan: physics, time limit, obs history and RNG restored
    s = snap.step
    replay, _ = run_branches(bundle, make_single_env(env_cfg), snap, noise_seed=None)
    np.testing.assert_allclose(replay[0].trace["block_pose"], parent.trace["block_pose"][s:], atol=1e-9)
    np.testing.assert_array_equal(replay[0].trace["action"], parent.trace["action"][s:])
    eps, plans = run_branches(bundle, make_envs(env_cfg, 3), snap, noise_seed=1)
    assert len(eps) == 3 and all(e.n_steps == MAX_STEPS - s for e in eps)
    assert plans.shape[0] == 3 and plans.shape[2] == 2
    _, rows = fate_curve(bundle, env_cfg, seed=2, n_branches=2, every=cfg.n_action_steps)
    assert [r["step"] for r in rows] == list(range(0, MAX_STEPS, cfg.n_action_steps))
    with pytest.raises(ValueError):
        SnapshotTaker(cfg.n_obs_steps, cfg.n_action_steps + 1, cfg.n_action_steps)


def test_diffusion_chunk_averaging_replans_every_step(env_cfg):
    from plens.policy import set_sampler

    b = load_random_diffusion(env_cfg, "cpu", num_inference_steps=2)
    s = set_sampler(b, temporal_ensemble=0.01)
    assert s["temporal_ensemble"] == 0.01 and s["n_action_steps"] == b.policy.config.horizon - 1
    venv = make_single_env(env_cfg)
    with ActivationRecorder(b.policy, LAYERS) as rec:
        ep = run_episode(b, venv, seed=0, recorder=rec)
    venv.close()
    assert set(rec.to_arrays()["diffusion.rgb_encoder"]["step"]) == set(range(MAX_STEPS))
    assert ep.trace["action"].shape == (MAX_STEPS, 2)
