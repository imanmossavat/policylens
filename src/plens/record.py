"""Record layer outputs during rollouts with forward hooks.

Each call of a hooked layer becomes one record: (env step, call index within that
step, output). Diffusion policies call the denoiser many times per action chunk and
only re-plan every few env steps, so "one record per step" would be wrong.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn


def list_layers(model: nn.Module, max_depth: int | None = None) -> list[tuple[str, str]]:
    """(name, class) for hookable modules, i.e. modules whose forward() actually runs."""
    out = []
    for name, mod in model.named_modules():
        if not name or isinstance(mod, (nn.ModuleList, nn.ModuleDict)):
            continue
        if max_depth is not None and name.count(".") >= max_depth:
            continue
        out.append((name, type(mod).__name__))
    return out


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Layer output has no tensor: {type(output).__name__}")


def pool(x: torch.Tensor) -> torch.Tensor:
    """Average every dim after (batch, channel). Keeps files small; spatial/time detail is lost."""
    return x.mean(dim=tuple(range(2, x.ndim))) if x.ndim > 2 else x


class ActivationRecorder:
    """Context manager. Set `.step` before each env step; read `.to_arrays()` afterwards."""

    def __init__(self, model: nn.Module, layers: list[str], pool_outputs: bool = True):
        modules = dict(model.named_modules())
        for name in layers:
            if name not in modules:
                raise KeyError(f"No layer named {name!r}. Use list_layers() to see options.")
            # GOTCHA(G-002): ModuleList/ModuleDict.forward never runs, so hooks on them stay silent.
            if isinstance(modules[name], (nn.ModuleList, nn.ModuleDict)):
                raise ValueError(f"{name!r} is a container; hook one of its children instead.")
        self.model, self.layers, self.pool_outputs = model, list(layers), pool_outputs
        self.step = 0
        self._records: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
        self._calls: dict[tuple[str, int], int] = defaultdict(int)
        self._handles: list[Any] = []

    def _hook(self, name: str):
        def fn(_module, _inputs, output):
            x = _first_tensor(output).detach()
            if self.pool_outputs:
                x = pool(x)
            key = (name, self.step)
            self._records[name].append((self.step, self._calls[key], x.float().cpu().numpy()))
            self._calls[key] += 1
        return fn

    def __enter__(self) -> "ActivationRecorder":
        modules = dict(self.model.named_modules())
        self._handles = [modules[n].register_forward_hook(self._hook(n)) for n in self.layers]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def to_arrays(self, calls: str = "all") -> dict[str, dict[str, np.ndarray]]:
        """{layer: {"step": [N], "call": [N], "value": [N, *shape]}} for every layer that fired.

        calls="last" keeps only the last call per (layer, step), e.g. the final
        denoising pass of a diffusion policy: its finished plan, 1/100th the size.
        """
        if calls not in ("all", "last"):
            raise ValueError(f"calls must be 'all' or 'last', got {calls!r}")
        out = {}
        for name, recs in self._records.items():
            if calls == "last":
                recs = [r for r in recs if r[1] == self._calls[(name, r[0])] - 1]
            out[name] = {
                "step": np.array([r[0] for r in recs], dtype=np.int32),
                "call": np.array([r[1] for r in recs], dtype=np.int32),
                "value": np.stack([r[2] for r in recs]),
            }
        return out
