"""Load a LeRobot policy and its processors, and record exactly which weights were used."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env_pre_post_processors
from lerobot.envs.configs import EnvConfig
from lerobot.policies import make_policy, make_pre_post_processors

DEFAULT_MIGRATION_CACHE = Path(".cache/migrated")

# WHY these two (D-005): image features = what the policy sees; denoiser bottleneck =
# what it plans. Names valid for lerobot/diffusion_pusht and load_random_diffusion().
DIFFUSION_PUSHT_LAYERS = ["diffusion.rgb_encoder", "diffusion.unet.mid_modules.1"]


@dataclass
class PolicyBundle:
    """Everything needed to turn an env observation into an env action."""

    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any
    source: dict[str, Any] = field(default_factory=dict)  # goes into meta.json


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _has_processors(path: Path) -> bool:
    return any(path.glob("policy_preprocessor*.json"))


def _tuples_where_declared(cfg: Any) -> None:
    """JSON turns tuples into lists; LeRobot's config encoder then fails on save (G-005)."""
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if isinstance(value, list) and "tuple" in str(f.type):
            setattr(cfg, f.name, tuple(value))


def migrate_legacy_checkpoint(src: Path, dst: Path) -> None:
    """Convert a pre-processor-era checkpoint, using LeRobot's own migration helpers.

    GOTCHA(G-001): lerobot/diffusion_pusht on the Hub predates LeRobot's processor
    pipelines; its normalization stats live inside model.safetensors.
    WHY in-process: LeRobot's migration CLI crashes on these configs (G-005).
    """
    from lerobot.policies.factory import get_policy_class, make_policy_config
    from lerobot.processor import migrate_policy_normalization as mig
    from safetensors.torch import load_file

    state_dict = load_file(src / "model.safetensors")
    config = json.loads((src / "config.json").read_text())
    stats = mig.extract_normalization_stats(state_dict)
    _, norm_map = mig.detect_features_and_norm_modes(config, stats)
    clean = mig.clean_state_dict(mig.remove_normalization_layers(state_dict), remove_str="._orig_mod")

    kwargs = {k: v for k, v in config.items() if k not in ("normalization_mapping", "type")}
    for key in ("input_features", "output_features"):
        if key in kwargs:
            kwargs[key] = mig.convert_features_to_policy_features(kwargs[key])
    kwargs["normalization_mapping"] = norm_map
    kwargs["device"] = "cpu"  # Hub config says "cuda"; the real device is chosen at load time
    cfg = make_policy_config(config["type"], **kwargs)
    _tuples_where_declared(cfg)

    policy = get_policy_class(config["type"])(cfg)
    missing, unexpected = policy.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Weights do not match the architecture: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]}")
    pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=stats)

    dst.mkdir(parents=True, exist_ok=True)
    pre.save_pretrained(dst)
    post.save_pretrained(dst)
    policy.to(torch.float32).save_pretrained(dst)
    if not _has_processors(dst):
        raise RuntimeError(f"Migration of {src} produced no processor files in {dst}")


def resolve_checkpoint(
    repo_or_path: str, revision: str | None = None, cache: Path = DEFAULT_MIGRATION_CACHE
) -> tuple[Path, dict[str, Any]]:
    """Return a local checkpoint dir with processors, plus provenance for meta.json."""
    local = Path(repo_or_path)
    if local.is_dir():
        info: dict[str, Any] = {"repo_id": None, "local_path": str(local.resolve()), "commit": None}
    else:
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(repo_or_path, revision=revision))
        # The snapshot folder name is the resolved commit hash: exact weights, even if "main" moves.
        info = {"repo_id": repo_or_path, "requested_revision": revision, "commit": local.name}

    info["migrated"] = False
    if not _has_processors(local):
        tag = f"{repo_or_path.replace('/', '__')}@{info['commit'] or 'local'}"
        migrated = cache / tag
        if not _has_processors(migrated):
            migrate_legacy_checkpoint(local, migrated)
        local, info["migrated"] = migrated, True
    info["loaded_from"] = str(local.resolve())
    return local, info


def _bundle(cfg: PreTrainedConfig, env_cfg: EnvConfig, pretrained: Path | None, device: str,
            source: dict[str, Any]) -> PolicyBundle:
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(pretrained) if pretrained else None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    source = {**source, "device": device, "policy_type": cfg.type,
              "n_params": sum(p.numel() for p in policy.parameters())}
    return PolicyBundle(policy, pre, post, env_pre, env_post, source)


def load_pretrained(repo_or_path: str, env_cfg: EnvConfig, device: str = "auto",
                    revision: str | None = None) -> PolicyBundle:
    from lerobot.utils.import_utils import register_third_party_plugins

    # WHY: policy types from installed lerobot_policy_* packages (e.g. flow_matching) only
    # exist once registered; LeRobot's own CLIs do the same.
    register_third_party_plugins()
    device = pick_device(device)
    local, source = resolve_checkpoint(repo_or_path, revision)
    cfg = PreTrainedConfig.from_pretrained(local)
    cfg.pretrained_path = str(local)
    cfg.device = device
    return _bundle(cfg, env_cfg, local, device, {"kind": "pretrained", **source})


def load_random_diffusion(env_cfg: EnvConfig, device: str = "auto", **overrides: Any) -> PolicyBundle:
    """Same architecture as the PushT diffusion policy, untrained. For tests and as a baseline.

    No ImageNet backbone download, so it works offline.
    """
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    device = pick_device(device)
    # WHY: match lerobot/diffusion_pusht's architecture so layer names are identical
    # (it uses one shared image encoder; LeRobot's current default is one per camera).
    kwargs = {"pretrained_backbone_weights": None, "crop_shape": (84, 84),
              "use_separate_rgb_encoder_per_camera": False, **overrides}
    cfg = DiffusionConfig(device=device, **kwargs)
    return _bundle(cfg, env_cfg, None, device, {"kind": "random_init", "overrides": kwargs})


SAMPLERS = ("ddpm", "ddim")


def set_sampler(bundle: PolicyBundle, sampler: str | None = None, steps: int | None = None,
                n_action_steps: int | None = None, temporal_ensemble: float | None = None) -> dict[str, Any]:
    """Change how a loaded policy samples, without retraining. Returns what is now set (for meta.json).

    sampler: "ddpm" or "ddim", diffusion only; the trained denoiser is unchanged, only the
      update rule between passes. steps: denoising passes (diffusion) or ODE steps (flow)
      per re-plan. n_action_steps: actions executed per re-plan, at most
      horizon - n_obs_steps + 1. Call before the first episode; reset() reads it.
    temporal_ensemble: re-plan every step and average the overlapping chunks with weights
      exp(-coeff * age) (the ACT paper uses 0.01). ACT natively; diffusion via _ensemble_diffusion.
    """
    policy, cfg = bundle.policy, bundle.policy.config
    model = getattr(policy, "diffusion", None)
    if sampler is not None:
        if model is None or not hasattr(model, "noise_scheduler"):
            raise ValueError(f"sampler={sampler!r} applies to diffusion policies, not {cfg.type!r}")
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        kw = dict(num_train_timesteps=cfg.num_train_timesteps, beta_start=cfg.beta_start,
                  beta_end=cfg.beta_end, beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
                  clip_sample_range=cfg.clip_sample_range, prediction_type=cfg.prediction_type)
        model.noise_scheduler = (DDPMScheduler if sampler == "ddpm" else DDIMScheduler)(**kw)
    if steps is not None:
        if model is not None:
            model.num_inference_steps = steps  # diffusion copies it from the config at init
        cfg.num_inference_steps = steps        # flow matching reads the config at call time
    if temporal_ensemble is not None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        if cfg.type == "act":
            cfg.temporal_ensemble_coeff, cfg.n_action_steps = temporal_ensemble, 1
            policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble, cfg.chunk_size)
        elif model is not None:
            _ensemble_diffusion(policy, temporal_ensemble)
        else:
            raise ValueError(f"temporal_ensemble supports ACT and diffusion, not {cfg.type!r}")
    if n_action_steps is not None:
        horizon = getattr(cfg, "horizon", None) or cfg.chunk_size  # ACT calls it chunk_size
        limit = horizon - cfg.n_obs_steps + 1
        if not 1 <= n_action_steps <= limit:
            raise ValueError(f"n_action_steps must be in 1..{limit} for horizon {horizon}")
        cfg.n_action_steps = n_action_steps
    return describe_sampling(bundle)


def _ensemble_diffusion(policy, coeff: float) -> None:
    """Re-plan a diffusion policy every step and average overlapping chunks, as ACT does.

    Each step samples a full usable chunk (horizon - n_obs_steps + 1 actions) and blends it
    with the still-valid parts of earlier chunks, weights exp(-coeff * age). WHY: tests whether
    chunk blending, which rescues ACT at short re-plan intervals (F-013), also rescues a
    sampling policy, whose chunks may disagree on the mode (F-014).
    """
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    cfg = policy.config
    cfg.n_action_steps = cfg.horizon - cfg.n_obs_steps + 1
    ensembler = ACTTemporalEnsembler(coeff, cfg.n_action_steps)
    reset = policy.reset

    def select_action(batch, noise=None):
        batch = dict(batch)
        batch.pop(ACTION, None)
        if cfg.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in cfg.image_features], dim=-4)
        policy._queues = populate_queues(policy._queues, batch)
        return ensembler.update(policy.predict_action_chunk(batch, noise=noise))

    def reset_all():
        reset()
        ensembler.reset()

    policy.select_action, policy.reset = select_action, reset_all
    cfg.temporal_ensemble_coeff = coeff


def describe_sampling(bundle: PolicyBundle) -> dict[str, Any]:
    """Sampler, passes per re-plan, horizon and re-plan interval actually in effect."""
    policy, cfg = bundle.policy, bundle.policy.config
    model = getattr(policy, "diffusion", None)
    if model is not None and hasattr(model, "noise_scheduler"):
        sampler = type(model.noise_scheduler).__name__.removesuffix("Scheduler").lower()
        steps = model.num_inference_steps
    else:  # flow matching: ODE solver; ACT: one deterministic pass, no sampler
        sampler = getattr(cfg, "ode_solver", cfg.type)
        steps = getattr(cfg, "num_inference_steps", 1)
    horizon = getattr(cfg, "horizon", None) or getattr(cfg, "chunk_size", None)
    return {"sampler": sampler, "steps": steps, "horizon": horizon,
            "n_action_steps": cfg.n_action_steps, "n_obs_steps": cfg.n_obs_steps,
            "temporal_ensemble": getattr(cfg, "temporal_ensemble_coeff", None)}
