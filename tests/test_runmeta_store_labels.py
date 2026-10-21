import numpy as np
import pytest

from plens import labels, store
from plens.rollout import Episode
from plens.runmeta import new_run, read_meta, update_meta


def make_episode(seed=0, success=False, cov=(0.1, 0.5, 0.3)):
    cov = np.array(cov)
    trace = {"action": np.zeros((len(cov), 2)), "coverage": cov, "block_pose": np.zeros((len(cov), 3))}
    return Episode(seed=seed, success=success, n_steps=len(cov), max_coverage=float(cov.max()),
                   final_coverage=float(cov[-1]), trace=trace)


def test_new_run_meta_has_provenance(tmp_path):
    run = new_run(tmp_path, "t", config={"n": 3, "seed": 7})
    meta = read_meta(run)
    assert meta["status"] == "running" and meta["config"]["seed"] == 7
    for key in ("python", "platform", "packages", "git"):
        assert key in meta["environment"]
    assert "torch" in meta["environment"]["packages"]


def test_update_meta_merges(tmp_path):
    run = new_run(tmp_path, "t", config={})
    update_meta(run, status="done", summary={"n_episodes": 1})
    meta = read_meta(run)
    assert meta["status"] == "done" and meta["config"] == {} and meta["summary"]["n_episodes"] == 1


def test_episode_roundtrip_with_activations(tmp_path):
    ep = make_episode(seed=4)
    acts = {"layer.x": {"step": np.array([0, 0, 1]), "call": np.array([0, 1, 0]),
                        "value": np.random.rand(3, 1, 8)}}
    line = store.save_episode(tmp_path, 0, ep, "fail", acts)
    assert line["seed"] == 4 and line["layers"] == ["layer.x"]
    loaded = store.load_episode(tmp_path, 0)
    np.testing.assert_array_equal(loaded["trace"]["coverage"], ep.trace["coverage"])
    np.testing.assert_array_equal(loaded["act"]["layer.x"]["value"], acts["layer.x"]["value"])


def test_index_appends_one_line_per_episode(tmp_path):
    for i in range(3):
        store.save_episode(tmp_path, i, make_episode(seed=i), "fail")
    assert [l["idx"] for l in store.read_index(tmp_path)] == [0, 1, 2]


@pytest.mark.parametrize("success,max_cov,expected", [
    (True, 0.96, "success"), (False, 0.85, "near_miss"), (False, 0.8, "near_miss"), (False, 0.3, "fail"),
])
def test_outcome_label(success, max_cov, expected):
    assert labels.outcome_label(success, max_cov) == expected


def test_summarize():
    index = [{"success": True, "label": "success", "max_coverage": 1.0, "n_steps": 100},
             {"success": False, "label": "fail", "max_coverage": 0.2, "n_steps": 300}]
    s = labels.summarize(index)
    assert s["success_rate"] == 0.5 and s["labels"] == {"fail": 1, "success": 1}
    assert s["mean_max_coverage"] == 0.6 and labels.summarize([]) == {"n_episodes": 0}
