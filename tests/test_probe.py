import numpy as np

from plens import probe


def test_auc_known_values():
    assert probe.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert probe.auc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5


def test_probe_reads_a_linear_signal_and_not_noise():
    rng = np.random.default_rng(0)
    episode = np.repeat(np.arange(20), 50)
    x = rng.normal(size=(1000, 8))
    y_signal = (x[:, 0] + 0.3 * rng.normal(size=1000) > 0).astype(int)
    y_noise = rng.integers(0, 2, 1000)
    assert probe.probe(x, y_signal, episode)["all"] > 0.9
    assert abs(probe.probe(x, y_noise, episode)["all"] - 0.5) < 0.1
