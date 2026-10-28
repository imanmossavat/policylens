"""Square policy training (plens.train_square); needs the DPPO data and source (docs/assets.md)."""

import pytest

torch = pytest.importorskip("torch")
from plens import square, train_square  # noqa: E402

pytestmark = pytest.mark.skipif(not (train_square.DATA.exists() and square.DPPO_SRC.exists()),
                                reason="assets not downloaded (docs/assets.md)")


def test_chunks_never_cross_a_demo_end():
    import numpy as np

    S, A = train_square.load_chunks(2)
    lens = np.load(train_square.DATA)["traj_lengths"][:2]
    assert len(S) == sum(n - 3 for n in lens)
    assert A.shape[1:] == (4, 7)


@pytest.mark.parametrize("kind", ["diffusion", "flow"])
def test_train_save_load_sample(kind, tmp_path):
    train_square.train(kind, n_demos=2, steps=20, out=tmp_path / "c.pt", device="cpu", batch=8, log_every=10**9)
    model = train_square.load_trained(tmp_path / "c.pt")
    plan = model({"state": torch.zeros(3, 1, 23)}).trajectories
    assert plan.shape == (3, 4, 7) and plan.abs().max() <= 1
