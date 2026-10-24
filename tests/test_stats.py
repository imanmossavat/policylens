import numpy as np
import pytest

from plens import labels, stats


def test_wilson_matches_known_interval():
    lo, hi = stats.wilson(35, 50)
    assert lo == pytest.approx(0.5617, abs=1e-3) and hi == pytest.approx(0.8092, abs=1e-3)
    assert stats.wilson(0, 10)[0] == 0.0 and stats.wilson(10, 10)[1] == pytest.approx(1.0)


def test_exact_binomial_p():
    assert stats.binom_two_sided(0, 5) == pytest.approx(2 / 32)
    assert stats.binom_two_sided(5, 10) == 1.0
    assert stats.binom_two_sided(0, 0) == 1.0


def test_paired_counts_only_shared_seeds():
    a = {0: True, 1: True, 2: False, 3: False, 9: True}
    b = {0: True, 1: False, 2: True, 3: True}
    r = stats.paired(a, b)
    assert (r["n_shared"], r["both"], r["only_a"], r["only_b"], r["neither"]) == (4, 1, 1, 2, 0)
    assert r["diff"] == pytest.approx(0.25)
    assert r["diff_ci"][0] < r["diff"] < r["diff_ci"][1]


def test_episodes_for_halfwidth():
    assert stats.episodes_for_halfwidth(0.65, 0.05) == 350


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(1).normal(10, 1, 200)
    lo, hi = stats.bootstrap_ci(x)
    assert lo < x.mean() < hi and hi - lo < 1


def test_compare_pairs_against_first_run():
    line = lambda s, ok: {"seed": s, "success": ok, "max_coverage": 1.0 if ok else 0.5}
    rows = labels.compare({"ref": [line(0, True), line(1, False)], "new": [line(0, False), line(1, False)]})
    assert rows[0]["k"] == 1 and "vs_ref" not in rows[0]
    assert rows[1]["vs_ref"]["only_a"] == 1 and rows[1]["vs_ref"]["only_b"] == 0


def test_fate_stats_finds_the_deciding_moment():
    from plens.branch import fate_stats

    steps = np.arange(0, 80, 8)
    cliff = {"step": steps, "n": np.full(10, 50), "p": np.array([.6, .6, .62, .58, .6, .02, 0, 0, 0, 0])}
    s = fate_stats(cliff)
    assert s["largest_change_step"] == 40 and s["largest_change"] < -0.5
    assert s["decisiveness"] > 0.8 and s["sealed_step"] == 40
    drift = {"step": steps, "n": np.full(10, 50), "p": np.linspace(0.6, 0.0, 10)}
    assert fate_stats(drift)["decisiveness"] < 0.2
