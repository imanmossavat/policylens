import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plens import explore, explore_plots as P


def test_toy_switch_counts_follow_the_drawn_path():
    e = explore.toy_episode([0, 1.5, 0, 1.5], steps_per_leg=16)
    assert e.analyze(min_swing=0.5)["n_switches"] == 2
    assert e.analyze(min_swing=2.0)["n_switches"] == 0  # threshold above any single re-plan turn


def test_toy_jitter_below_threshold_is_not_counted():
    e = explore.toy_episode([0, 1.5, 3.0], steps_per_leg=16, wobble=0.02)
    assert e.analyze(min_swing=0.5)["n_switches"] == 0


def test_switch_steps_match_count_and_sweep_shape():
    e = explore.toy_episode([0, 1.5, 0, 1.5], steps_per_leg=16)
    assert len(e.switch_steps()) == e.analyze()["n_switches"]
    run = explore.RunData("x", [e, explore.toy_episode([0, 1.5], 16)])
    thr = np.array([0.2, 0.5, 1.0])
    assert explore.sweep(run, thr).shape == (3, 2)
    assert explore.sweep_by_label(run, thr)["toy"].shape == (3,)


def test_activation_projection_and_change():
    x = np.random.default_rng(0).normal(size=(12, 2, 8))
    xy, var = explore.activation_projection(x, 2)
    assert xy.shape == (12, 2) and 0 < var.sum() <= 1
    assert explore.activation_change(x).shape == (11,)


def test_plots_draw_without_error():
    e = explore.toy_episode([0, 1.5, 0], 16)
    e._act = {"L": {"step": np.arange(0, 8, 1) * 4, "call": np.arange(8), "value": np.random.rand(8, 1, 5)}}
    run = explore.RunData("x", [e])
    fig, axs = plt.subplots(1, 5)
    P.paths(axs[0], [e])
    P.threshold_sweep(axs[1], run, np.array([0.2, 0.5]), 0.5)
    P.scene(axs[2], e, 5)
    P.activation_view(axs[3], axs[4], e, "L", 6)
    plt.close(fig)
