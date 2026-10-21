import numpy as np

from plens import sides, store
from plens.rollout import Episode


def circle(angles, radius=100.0, block=(256.0, 256.0, 0.0)):
    """Agent on a circle around a fixed block at the given world angles."""
    angles = np.asarray(angles, float)
    pos = np.stack([block[0] + radius * np.cos(angles), block[1] + radius * np.sin(angles)], 1)
    return {"pos_agent": pos, "block_pose": np.tile(block, (len(angles), 1))}


def test_circling_one_way_is_swings_without_switches():
    r = sides.analyze_episode(circle(np.linspace(0, -4 * np.pi, 48)))  # clockwise
    assert r["n_cw"] > 0 and r["n_ccw"] == 0 and r["n_switches"] == 0


def test_reversing_counts_one_switch():
    ang = np.concatenate([np.linspace(0, 2.0, 16), np.linspace(2.0, 0.0, 16)])
    r = sides.analyze_episode(circle(ang))
    assert [d for _, d in r["swings"]] == [1, 1, -1, -1] and r["n_switches"] == 1


def test_small_turns_are_not_swings():
    r = sides.analyze_episode(circle(np.linspace(0, 0.3, 32)))
    assert r["swings"] == []


def test_wrapping_past_pi_does_not_fake_a_swing_or_switch():
    r = sides.analyze_episode(circle(np.linspace(3.0, 3.0 + 2.5, 24)))  # crosses angle pi
    assert r["n_ccw"] > 0 and r["n_cw"] == 0 and r["n_switches"] == 0


def test_pushing_the_block_is_not_going_around_it():
    # agent stays fixed in the block frame while the block translates and rotates
    n = 32
    th = np.linspace(0, 2.0, n)
    center = np.stack([np.linspace(200, 300, n), np.full(n, 256.0)], 1)
    local = np.array([-60.0, 10.0])
    rot = np.stack([np.cos(th) * local[0] - np.sin(th) * local[1],
                    np.sin(th) * local[0] + np.cos(th) * local[1]], 1)
    trace = {"pos_agent": center + rot, "block_pose": np.column_stack([center, th])}
    assert sides.analyze_episode(trace)["swings"] == []


def test_analyze_run_joins_index_and_by_label(tmp_path):
    for idx, (ang, label) in enumerate([(np.linspace(0, 3, 32), "success"),
                                        (np.concatenate([np.linspace(0, 2, 16), np.linspace(2, 0, 16)]), "fail")]):
        trace = {**circle(ang), "action": np.zeros((32, 2)), "coverage": np.zeros(32)}
        ep = Episode(seed=idx, success=label == "success", n_steps=32, max_coverage=0.1,
                     final_coverage=0.1, trace=trace)
        store.save_episode(tmp_path, idx, ep, label)
    rows = sides.analyze_run(tmp_path)
    assert [r["label"] for r in rows] == ["success", "fail"]
    summary = sides.by_label(rows)
    assert summary["success"]["with_switch"] == 0 and summary["fail"]["with_switch"] == 1
