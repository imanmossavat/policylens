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


def _fake_run(tmp_path, with_meta=True):
    """A real-shaped run folder: meta.json, episodes.jsonl and one episode file."""
    import json

    from plens import runmeta, store

    run_dir = runmeta.new_run(tmp_path, "t", config={"policy": "x/y", "n": 1, "seed": 0, "max_steps": 40,
                                                     "layers": ["L"], "video": 0, "near_miss_coverage": 0.8})
    runmeta.update_meta(run_dir, status="done", policy={"repo_id": "x/y", "commit": "abc", "policy_type": "diffusion"})
    e = explore.toy_episode([0, 1.5, 0], 16)
    store.episode_path(run_dir, 0).parent.mkdir(parents=True)
    np.savez_compressed(store.episode_path(run_dir, 0), **{f"trace/{k}": v for k, v in e.trace.items()})
    (run_dir / store.INDEX_FILE).write_text(json.dumps({"idx": 0, "seed": 0, "label": "fail"}) + "\n")
    return run_dir


def test_provenance_reports_files_policy_and_code(tmp_path):
    run = explore.load(_fake_run(tmp_path))
    p = explore.provenance(run)
    assert p["run_id"].endswith("_t") and p["status"] == "done"
    assert p["policy"]["repo_id"] == "x/y" and p["policy"]["checkpoint_commit"] == "abc"
    assert p["settings"]["n"] == 1 and p["settings"]["layers"] == ["L"]
    assert set(p["files"]) == {"meta.json", "episodes.jsonl", "episodes/", "videos/"}
    assert explore.episode_file(run, 0).name == "ep_00000.npz" and explore.episode_file(run, 0).exists()


def test_provenance_survives_a_run_without_meta(tmp_path):
    run_dir = _fake_run(tmp_path)
    (run_dir / "meta.json").unlink()
    p = explore.provenance(explore.load(run_dir))
    assert p["policy"]["repo_id"] is None and p["code"]["commit"] is None


def test_load_missing_run_names_the_folder_and_what_is_there(tmp_path):
    import pytest

    _fake_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="No run folder at .*Runs found next to it"):
        explore.load(tmp_path / "nope")


def test_ui_panels_build_and_redraw_in_place(tmp_path):
    import pytest

    pytest.importorskip("ipywidgets")
    from plens import explore_ui

    run = explore.load(_fake_run(tmp_path))
    assert "x/y" in explore_ui.provenance_panel(run).value
    toy = explore_ui.toy_panel()
    img = toy.children[-1]
    first = img.value
    toy.children[0].value = "0, 1.5, 0"          # change the path: same widget, new bytes
    assert img.value and img.value != first and img.value[:8] == b"\x89PNG\r\n\x1a\n"
    panel = explore_ui.threshold_panel(run)
    assert panel.children[-2].value[:4] == b"\x89PNG" and "fail" in panel.children[-1].value
    assert explore_ui.replay_panel(run).children[2].value[:4] == b"\x89PNG"


def test_export_replay_writes_a_playable_mp4(tmp_path):
    import pytest

    pytest.importorskip("ipywidgets")
    from plens import explore_ui

    e = explore.toy_episode([0, 1.5, 0], 8)
    path = explore_ui.export_replay(explore.RunData(tmp_path / "myrun", [e]), -1, out_dir=tmp_path / "out")
    assert path.exists() and path.suffix == ".mp4" and path.stat().st_size > 1000
    import imageio.v2 as iio

    assert len(iio.mimread(str(path), memtest=False)) == e.n_steps
