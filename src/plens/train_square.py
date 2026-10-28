"""Train Square policies like DPPO's pretrained one, on fewer demos or with flow matching (D-017).

Data: DPPO's train.npz (300 robomimic MH demos, already normalized; the first n_demos are used, as
DPPO's max_n_episodes does). Network: the same Unet1D; plan of 4 actions from 1 observation.

  diffusion  DPPO DiffusionModel loss (predict noise, 20 DDPM steps); sampled by square.rollout_batch
  flow       same network predicts velocity along the straight noise->action path; 10 Euler steps

Recipe from DPPO cfg/robomimic/pretrain/square/pre_diffusion_unet.yaml: AdamW lr 1e-4, weight decay
1e-6, batch 256, cosine to 1e-5; EMA decay 0.995 updated every 10 batches. We train for a fixed
number of gradient steps instead of epochs, so different demo counts get the same compute.
"""

from __future__ import annotations

import copy
import math
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from . import square

DATA = square.ROOT / ".cache/dppo/data_square/train.npz"
Sample = namedtuple("Sample", "trajectories chains")


def select_demos(spec: str, seed: int = 0) -> list[int]:
    """Demo indices into train.npz (= robomimic demo_<i>): "first:N", "random:N", or an operator group
    "better" / "okay" / "worse" from the robomimic file's masks. GOTCHA(G-017): the first 50 demos are all "better"."""
    kind, _, n = spec.partition(":")
    if kind == "first":
        return list(range(int(n)))
    if kind == "random":
        return sorted(np.random.default_rng(seed).choice(300, int(n), replace=False).tolist())
    import json

    # masks.json = the hdf5's mask/<group> lists, exported once (h5py is not in the training env)
    return json.loads((square.DEMOS.parent / "masks.json").read_text())[kind]


def load_chunks(demos: int | list[int], path: Path | str = DATA, horizon: int = 4):
    """(obs, action chunk) pairs: obs at step t, actions t..t+horizon-1, never crossing a demo end.

    demos: an int N = the first N demos (DPPO's max_n_episodes), or explicit demo indices.
    """
    d = np.load(path)
    all_lens = d["traj_lengths"]
    ids = list(range(demos)) if isinstance(demos, int) else list(demos)
    offsets = np.concatenate([[0], np.cumsum(all_lens)[:-1]])
    S, A = d["states"].astype(np.float32), d["actions"].astype(np.float32)
    starts = []
    for i in ids:
        cur, n = int(offsets[i]), int(all_lens[i])
        starts += range(cur, cur + n - horizon + 1)
    idx = np.asarray(starts)
    chunks = np.stack([A[idx + k] for k in range(horizon)], axis=1)  # (N, horizon, act)
    return S[idx], chunks


def _flow_policy_class():
    import torch
    from torch import nn

    class FlowPolicy(nn.Module):
        """Rectified-flow policy with the same network and call signature as DPPO's DiffusionModel."""

        def __init__(self, network, horizon_steps, action_dim, flow_steps=10, device="cpu", **_):
            super().__init__()
            self.network = network.to(device)
            self.horizon_steps, self.action_dim, self.flow_steps = horizon_steps, action_dim, flow_steps

        def _v(self, x, t, cond):
            # WHY: Unet1D embeds time with a sinusoid tuned for integer steps; scale [0,1] to [0,100]
            return self.network(x, t * 100, cond=cond)

        def loss(self, x1, cond):
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), device=x1.device)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
            return nn.functional.mse_loss(self._v(xt, t, cond), x1 - x0)

        @torch.no_grad()
        def forward(self, cond, deterministic=True):
            B = len(cond["state"])
            dev = next(self.parameters()).device
            x = torch.randn((B, self.horizon_steps, self.action_dim), device=dev)
            dt = 1.0 / self.flow_steps
            for i in range(self.flow_steps):
                x = x + dt * self._v(x, torch.full((B,), i * dt, device=dev), cond)
            return Sample(x.clamp(-1, 1), None)

    return FlowPolicy


def make_model(kind: str, device: str = "cpu"):
    if kind == "diffusion":
        return square.load_policy(ckpt=None, device=device).train()
    import sys

    if str(square.DPPO_SRC) not in sys.path:
        sys.path.insert(0, str(square.DPPO_SRC))
    from model.diffusion.unet import Unet1D

    net = Unet1D(cond_dim=square.POLICY_CFG["obs_dim"], action_dim=square.POLICY_CFG["action_dim"], **square.UNET_CFG)
    return _flow_policy_class()(net, device=device, **square.POLICY_CFG)


def load_trained(path: Path | str, device: str = "cpu", weights: str = "ema"):
    """A checkpoint written by train(); kind is stored inside."""
    import torch

    ck = torch.load(path, map_location=device, weights_only=True)
    model = make_model(ck["kind"], device)
    model.load_state_dict(ck[weights], strict=True)
    return model.eval()


def train(kind: str, n_demos: int | list[int], steps: int, out: Path | str, device: str = "mps", batch: int = 256,
          seed: int = 0, log_every: int = 2000, log=print) -> dict:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    S, A = load_chunks(n_demos)
    S, A = torch.from_numpy(S).to(device), torch.from_numpy(A).to(device)
    model = make_model(kind, device)
    model.train()
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    warm, ema_start = 100, steps // 50
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    t0, losses = time.time(), []
    for step in range(1, steps + 1):
        i = torch.randint(0, len(S), (batch,), device=device)
        loss = model.loss(A[i], {"state": S[i][:, None, :]})
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item()) if step % 50 == 0 else None
        if step % 10 == 0:
            with torch.no_grad():
                if step < ema_start:
                    ema.load_state_dict(model.state_dict())
                else:
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(0.995).add_(pm, alpha=0.005)
        if step % log_every == 0:
            log(f"{kind} n={n_demos if isinstance(n_demos, int) else len(n_demos)} step {step}/{steps} loss {np.mean(losses[-40:]):.4f} {time.time() - t0:.0f}s")
    torch.save({"kind": kind, "n_demos": n_demos, "steps": steps, "model": model.state_dict(),
                "ema": ema.state_dict()}, out)
    return {"n_pairs": len(S), "final_loss": float(np.mean(losses[-40:])), "train_s": round(time.time() - t0)}
