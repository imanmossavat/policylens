import numpy as np
import pytest
import torch
from torch import nn

from plens.record import ActivationRecorder, list_layers, pool


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 4, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Linear(4, 3), nn.Linear(3, 3)])

    def forward(self, x):  # x: [B, 2, T]
        h = self.conv(x).mean(-1)
        for b in self.blocks:
            h = b(h)
        return h


def test_records_step_and_call_index():
    m = Toy()
    with ActivationRecorder(m, ["conv", "blocks.1"]) as rec:
        rec.step = 0
        m(torch.randn(1, 2, 5))
        m(torch.randn(1, 2, 5))  # second call within the same step
        rec.step = 3
        m(torch.randn(1, 2, 5))
    arrs = rec.to_arrays()
    assert arrs["conv"]["step"].tolist() == [0, 0, 3]
    assert arrs["conv"]["call"].tolist() == [0, 1, 0]
    assert arrs["conv"]["value"].shape == (3, 1, 4)  # pooled over time: [N, B, C]
    assert arrs["blocks.1"]["value"].shape == (3, 1, 3)


def test_calls_last_keeps_final_call_per_step():
    m = Toy()
    xs = [torch.randn(1, 2, 5) for _ in range(3)]
    with ActivationRecorder(m, ["conv"], pool_outputs=False) as rec:
        rec.step = 0
        m(xs[0])
        m(xs[1])
        rec.step = 1
        m(xs[2])
    last = rec.to_arrays(calls="last")["conv"]
    assert last["step"].tolist() == [0, 1] and last["call"].tolist() == [1, 0]
    np.testing.assert_allclose(last["value"][0], m.conv(xs[1]).detach().numpy(), rtol=1e-6)
    with pytest.raises(ValueError):
        rec.to_arrays(calls="first")


def test_values_match_layer_output():
    m = Toy()
    x = torch.randn(1, 2, 5)
    with ActivationRecorder(m, ["conv"], pool_outputs=False) as rec:
        m(x)
    np.testing.assert_allclose(rec.to_arrays()["conv"]["value"][0], m.conv(x).detach().numpy(), rtol=1e-6)


def test_hooks_removed_after_exit():
    m = Toy()
    with ActivationRecorder(m, ["conv"]) as rec:
        m(torch.randn(1, 2, 5))
    m(torch.randn(1, 2, 5))
    assert len(rec.to_arrays()["conv"]["step"]) == 1
    assert not m.conv._forward_hooks


def test_rejects_unknown_and_container_layers():
    m = Toy()
    with pytest.raises(KeyError):
        ActivationRecorder(m, ["nope"])
    with pytest.raises(ValueError):  # G-002
        ActivationRecorder(m, ["blocks"])


def test_list_layers_skips_containers():
    names = [n for n, _ in list_layers(Toy())]
    assert "blocks" not in names and "blocks.0" in names and "conv" in names


def test_pool_keeps_batch_and_channel():
    assert pool(torch.ones(2, 3, 4, 5)).shape == (2, 3)
    assert pool(torch.ones(2, 3)).shape == (2, 3)


def test_select_env_splits_env_major_blocks_and_drops_steps_after_done():
    from plens.record import select_env

    # 2 envs; "enc" has 2 rows per env (one per observation frame), "unet" 1 row per env.
    steps = np.array([0, 0, 8, 8])
    acts = {"enc": {"step": steps, "call": np.zeros(4, int), "value": np.arange(4 * 4 * 3).reshape(4, 4, 3)},
            "unet": {"step": steps, "call": np.array([0, 1, 0, 1]), "value": np.arange(4 * 2 * 3).reshape(4, 2, 3)}}
    one = select_env(acts, env=1, n_envs=2, n_steps=5)  # env 1 ended at step 5
    assert one["enc"]["value"].shape == (2, 2, 3) and one["unet"]["value"].shape == (2, 1, 3)
    np.testing.assert_array_equal(one["enc"]["value"], acts["enc"]["value"][:2, 2:4])
    np.testing.assert_array_equal(one["unet"]["step"], [0, 0])
