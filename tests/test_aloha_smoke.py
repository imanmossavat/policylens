"""ALOHA task specs with the real MuJoCo env (needs gym-aloha: run in the plens-aloha env)."""

import numpy as np
import pytest

pytest.importorskip("gym_aloha")
pytestmark = pytest.mark.smoke

from plens.envs import ALOHA_INSERTION, make_single_env  # noqa: E402


def test_insertion_observes_ground_truth_contact():
    venv = make_single_env(ALOHA_INSERTION.make_config(10))
    venv.reset(seed=[0])
    obs = venv.step(np.zeros((1, 14), dtype=np.float32))[0]
    gt = ALOHA_INSERTION.observe(venv.envs[0])
    venv.close()
    assert set(gt) == {"peg_pos", "socket_pos", "peg_quat", "socket_quat", "peg_socket_force", "peg_socket_contacts",
                       "agent_pos"}
    assert gt["peg_pos"].shape == (3,) and gt["peg_socket_force"] >= 0
    assert "top" in obs["pixels"]


def test_start_pose_calibration_is_millimetre_accurate():
    from plens.resim import calibrate

    cal = calibrate(n=16)
    assert max(cal.rms_mm) < 3.0
