"""robosuite Square + DPPO policy (plens-dppo env; skipped elsewhere)."""

import numpy as np
import pytest

robosuite = pytest.importorskip("robosuite")
from plens import square  # noqa: E402

needs_assets = pytest.mark.skipif(not (square.CKPT.exists() and square.DEMOS.exists()),
                                  reason="assets not downloaded (docs/assets.md)")


@needs_assets
def test_obs_vector_matches_dataset_obs():
    import h5py

    with h5py.File(square.DEMOS, "r") as f:
        g = f["data/demo_1"]
        env = square.make_env()
        env.reset()
        env.sim.set_state_from_flattened(g["states"][0])
        env.sim.forward()
        ours = square.obs_vector(env._get_observations(force_update=True))
        theirs = np.concatenate([g["obs"][k][0] for k in
                                 ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")])
    assert ours.shape == (square.POLICY_CFG["obs_dim"],)
    np.testing.assert_allclose(ours, theirs, atol=1e-5)


@pytest.mark.smoke
@needs_assets
def test_policy_loads_and_rolls_out():
    model = square.load_policy()
    eps = square.rollout_batch(model, [square.make_env(), square.make_env()], [0, 1], noise_seed=0, max_steps=8)
    assert [len(e.trace["action"]) for e in eps] == [8, 8]
    assert eps[0].arrays()["wrist_force"].shape == (9, 3)
