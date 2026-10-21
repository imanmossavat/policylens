"""ipywidgets panels over explore.py, so notebooks only call these (D-010, D-007).

Each function takes a RunData and returns a widget to display. Requires the `explore` extra.
"""

from __future__ import annotations

import numpy as np

from . import explore, explore_plots as P, sides


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _widgets():
    import ipywidgets as w
    return w


def threshold_panel(run: explore.RunData):
    """Paths + threshold sweep; drag min_swing / chunk and watch switch counts per outcome."""
    plt, w = _plt(), _widgets()
    thr = np.round(np.arange(0.1, 1.55, 0.05), 2)
    s_thr = w.FloatSlider(value=sides.MIN_SWING, min=0.1, max=1.5, step=0.05, description="min swing", readout_format=".2f")
    s_chunk = w.IntSlider(value=sides.CHUNK, min=2, max=24, description="chunk")
    c_sw = w.Checkbox(value=True, description="mark switches")
    c_ch = w.Checkbox(value=False, description="show re-plan lines")
    out = w.Output()

    def draw(*_):
        with out:
            out.clear_output(wait=True)
            fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2))
            P.paths(a, run.episodes, s_chunk.value, s_thr.value, c_sw.value, c_ch.value)
            P.threshold_sweep(b, run, thr, s_thr.value, s_chunk.value)
            counts = {e.idx: e.analyze(s_chunk.value, s_thr.value)["n_switches"] for e in run.episodes}
            a.set_title("switches per episode: " + " ".join(f"{i}:{n}" for i, n in counts.items()), fontsize=8)
            fig.tight_layout()
            plt.show()

    for c in (s_thr, s_chunk, c_sw, c_ch):
        c.observe(draw, "value")
    draw()
    return w.VBox([w.HBox([s_thr, s_chunk]), w.HBox([c_sw, c_ch]), out])


def replay_panel(run: explore.RunData):
    """One episode: video (or schematic) beside its angle line, with a step slider."""
    plt, w = _plt(), _widgets()
    ep_pick = w.Dropdown(options=[(f"{e.idx}  {e.label}  seed {e.seed}", e.idx) for e in run.episodes], description="episode")
    s_t = w.IntSlider(value=0, min=0, max=run.episodes[0].n_steps - 1, description="step", layout=w.Layout(width="600px"))
    s_thr = w.FloatSlider(value=sides.MIN_SWING, min=0.1, max=1.5, step=0.05, description="min swing")
    play = w.Play(interval=120, min=0, max=s_t.max, step=1)
    w.jslink((play, "value"), (s_t, "value"))
    out = w.Output()
    frames: dict[int, np.ndarray] = {}

    def video_frames(e):
        if e.video is None:
            return None
        if e.idx not in frames:
            import imageio.v2 as iio
            frames[e.idx] = np.stack(iio.mimread(e.video, memtest=False))
        return frames[e.idx]

    def on_episode(*_):
        e = run.get(ep_pick.value)
        s_t.max = play.max = e.n_steps - 1
        draw()

    def draw(*_):
        e = run.get(ep_pick.value)
        fr = video_frames(e)
        with out:
            out.clear_output(wait=True)
            fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
            P.scene(a, e, s_t.value, None if fr is None else fr[min(s_t.value, len(fr) - 1)])
            P.angle_with_cursor(b, e, s_t.value, sides.CHUNK, s_thr.value)
            fig.tight_layout()
            plt.show()

    ep_pick.observe(on_episode, "value")
    s_t.observe(draw, "value")
    s_thr.observe(draw, "value")
    draw()
    return w.VBox([w.HBox([ep_pick, s_thr]), w.HBox([play, s_t]), out])


def toy_panel():
    """A fake dot circling a fake T. Set the waypoints (rad) and jitter; see what gets counted."""
    plt, w = _plt(), _widgets()
    txt = w.Text(value="0, 1.5, 0, 1.5, 3.0", description="angles (rad)", layout=w.Layout(width="420px"))
    s_leg = w.IntSlider(value=16, min=2, max=48, description="steps/leg")
    s_wob = w.FloatSlider(value=0.0, min=0.0, max=0.3, step=0.01, description="jitter")
    s_thr = w.FloatSlider(value=sides.MIN_SWING, min=0.1, max=1.5, step=0.05, description="min swing")
    s_chunk = w.IntSlider(value=sides.CHUNK, min=2, max=24, description="chunk")
    out = w.Output()

    def draw(*_):
        with out:
            out.clear_output(wait=True)
            try:
                wp = [float(x) for x in txt.value.split(",")]
                assert len(wp) >= 2
            except (ValueError, AssertionError):
                print("give at least two comma-separated angles, e.g. 0, 1.5, 0")
                return
            e = explore.toy_episode(wp, s_leg.value, wobble=s_wob.value)
            r = e.analyze(s_chunk.value, s_thr.value)
            fig, ax = plt.subplots(figsize=(8, 3.6))
            P.paths(ax, [e], s_chunk.value, s_thr.value, show_chunks=True)
            ax.set_title(f"counted: {r['n_ccw']} ccw + {r['n_cw']} cw swings, {r['n_switches']} switches "
                         f"(you drew {sum((a - b) * (b - c) < 0 for a, b, c in zip(wp, wp[1:], wp[2:]))} reversals)",
                         fontsize=9)
            plt.show()

    for c in (txt, s_leg, s_wob, s_thr, s_chunk):
        c.observe(draw, "value")
    draw()
    return w.VBox([txt, w.HBox([s_leg, s_wob]), w.HBox([s_thr, s_chunk]), out])


def activation_panel(run: explore.RunData):
    """Pick an episode and a recorded layer; see how the layer moves over the episode."""
    plt, w = _plt(), _widgets()
    layers = sorted({l for e in run.episodes for l in e.activations()})
    ep_pick = w.Dropdown(options=[(f"{e.idx}  {e.label}", e.idx) for e in run.episodes], description="episode")
    l_pick = w.Dropdown(options=layers, description="layer", layout=w.Layout(width="420px"))
    s_t = w.IntSlider(value=0, min=0, max=run.episodes[0].n_steps - 1, description="step", layout=w.Layout(width="600px"))
    out = w.Output()

    def draw(*_):
        e = run.get(ep_pick.value)
        s_t.max = e.n_steps - 1
        with out:
            out.clear_output(wait=True)
            fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
            P.activation_view(a, b, e, l_pick.value, s_t.value)
            P.angle_with_cursor(c, e, s_t.value)
            fig.tight_layout()
            plt.show()

    for c in (ep_pick, l_pick, s_t):
        c.observe(draw, "value")
    draw()
    return w.VBox([w.HBox([ep_pick, l_pick]), s_t, out])
