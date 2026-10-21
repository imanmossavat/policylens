"""ipywidgets panels over explore.py, so notebooks only call these (D-010, D-007, D-011).

Each function takes a RunData and returns a widget to display. Requires the `explore` extra.

WHY figures are drawn off-screen and shown through one image widget whose bytes are replaced:
re-creating a pyplot figure on every slider tick clears the output first, which flashes. Swapping
the image in place does not. Each figure also carries a footer naming its source file and settings,
so a screenshot or an exported MP4 still says where it came from.
"""

from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any

import numpy as np

from . import explore, explore_plots as P, sides

# Plain-language names for the layers recorded by default (D-005).
LAYER_NAMES = {
    "diffusion.rgb_encoder": "what it SEES (image encoder)",
    "diffusion.unet.mid_modules.1": "what it PLANS (denoiser, middle)",
}
FIG_DPI = 80  # 12 x 4.2 in at 80 dpi = 960 x 336 px: both multiples of 16, as video codecs like


def _w():
    import ipywidgets as w
    return w


class _Figure:
    """An off-screen matplotlib figure shown through one ipywidgets.Image, redrawn in place."""

    def __init__(self, nrows: int, ncols: int, figsize: tuple[float, float], **subplot_kw: Any) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=figsize, dpi=FIG_DPI)
        self.canvas = FigureCanvasAgg(self.fig)
        self.axes = np.atleast_1d(self.fig.subplots(nrows, ncols, **subplot_kw))
        self.footer = ""
        self._laid_out = False
        self.image = _w().Image(format="png", layout=_w().Layout(max_width="100%"))

    def _prepare(self, relayout: bool) -> None:
        # tight_layout costs ~40 ms: only redo it when the axes' contents changed shape, not per frame.
        if relayout or not self._laid_out:
            self.fig.tight_layout(rect=(0, 0.05, 1, 1))
            self._laid_out = True
        for t in list(self.fig.texts):
            t.remove()
        self.fig.text(0.005, 0.008, self.footer, fontsize=6.5, color="0.35", va="bottom")

    def render(self, relayout: bool = True) -> None:
        self._prepare(relayout)
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", pil_kwargs={"compress_level": 1})  # savefig draws; no extra draw
        self.image.value = buf.getvalue()

    def rgb(self, relayout: bool = False) -> np.ndarray:
        self._prepare(relayout)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


def _note(text: str):
    """A small grey line of explanation or provenance under a figure."""
    return _w().HTML(f"<div style='font-size:12px;opacity:.75;line-height:1.4'>{text}</div>")


def _short(run: explore.RunData, *parts: str) -> str:
    return "/".join([Path(run.run_dir).name, *parts])


def _settings_text(chunk: int, min_swing: float) -> str:
    return f"chunk = {chunk} steps per re-plan, min swing = {min_swing:.2f} rad"


# ---------------------------------------------------------------------------------------------
# Where did this data come from?

def provenance_panel(run: explore.RunData):
    """Which files and which settings produced this run, straight from its meta.json."""
    p = explore.provenance(run)
    esc = html.escape

    def row(k: str, v: Any) -> str:
        v = "(not recorded)" if v is None or v == "" else v
        return f"<tr><td style='padding:2px 14px 2px 0;opacity:.7;vertical-align:top'>{esc(k)}</td><td>{v}</td></tr>"

    code = lambda s: f"<code>{esc(str(s))}</code>"  # noqa: E731
    pol, st, cd = p["policy"], p["settings"], p["code"]
    dirty = ("<b style='color:#c0392b'> with uncommitted changes: this run may not be exactly "
             "reproducible from the commit</b>") if cd["dirty"] else " (clean)" if cd["dirty"] is False else ""
    summ = p["summary"] or {}
    rows = [
        row("Run", code(p["run_id"])),
        row("Folder", code(p["folder"])),
        row("Files", "<br>".join(f"{code(k)} → {code(v)}" for k, v in p["files"].items())),
        row("Created / duration", f"{esc(str(p['created_utc']))} (UTC), {p['duration_s']} s" if p["duration_s"]
            else esc(str(p["created_utc"]))),
        row("Status", esc(str(p["status"]))),
        row("Command that made it", code(p["command"]) if p["command"] else None),
        row("Policy", f"{code(pol['repo_id'])} ({esc(str(pol['type']))}, {pol['n_params']:,} parameters)"
            if pol["n_params"] else code(pol["repo_id"])),
        row("Exact weights (checkpoint commit)", code(pol["checkpoint_commit"])),
        row("Ran on", esc(str(pol["device"]))),
        row("Episodes", f"{st['n']} episodes; episode i used seed {st['seed']} + i; at most {st['max_steps']} steps each"),
        row("Layers recorded", ", ".join(code(x) for x in st["layers"]) if st["layers"] else None),
        row("Videos", f"episodes {p['with_video']}" if p["with_video"] else "none recorded"),
        row("Outcome labels", f"success = target covered ≥ 95%; near_miss = not a success but covered ≥ "
            f"{st['near_miss_coverage']}; fail = the rest" if st["near_miss_coverage"] else None),
        row("Result", esc(str(summ)) if summ else None),
        row("Code version", f"{code(cd['commit'])}{dirty}" if cd["commit"] else None),
        row("Package versions", esc(", ".join(f"{k} {v}" for k, v in p["packages"].items() if v)) or None),
    ]
    import json
    raw = esc(json.dumps(p["raw_meta"], indent=2, default=str))
    body = (f"<table style='font-size:13px;border-collapse:collapse'>{''.join(rows)}</table>"
            f"<details style='margin-top:6px'><summary style='cursor:pointer;opacity:.75'>"
            f"Show the complete meta.json</summary><pre style='font-size:11px'>{raw}</pre></details>")
    return _w().HTML(body)


# ---------------------------------------------------------------------------------------------
# Panels

def toy_panel():
    """A fake dot circling a fake T. Set the waypoints (rad) and jitter; see what gets counted."""
    w = _w()
    txt = w.Text(value="0, 1.5, 0, 1.5, 3.0", description="angles (rad)", layout=w.Layout(width="420px"))
    s_leg = w.IntSlider(value=16, min=2, max=48, description="steps/leg")
    s_wob = w.FloatSlider(value=0.0, min=0.0, max=0.3, step=0.01, description="jitter")
    s_thr = w.FloatSlider(value=sides.MIN_SWING, min=0.1, max=1.5, step=0.05, description="min swing")
    s_chunk = w.IntSlider(value=sides.CHUNK, min=2, max=24, description="chunk")
    fig = _Figure(1, 1, (9.6, 3.9))
    msg = w.HTML()

    def draw(*_):
        try:
            wp = [float(x) for x in txt.value.split(",")]
            assert len(wp) >= 2
        except (ValueError, AssertionError):
            msg.value = "<b>Give at least two comma-separated angles, e.g. 0, 1.5, 0</b>"
            return
        msg.value = ""
        e = explore.toy_episode(wp, s_leg.value, wobble=s_wob.value)
        r = e.analyze(s_chunk.value, s_thr.value)
        drawn = sum((a - b) * (b - c) < 0 for a, b, c in zip(wp, wp[1:], wp[2:]))
        ax = fig.axes[0]
        ax.clear()
        P.paths(ax, [e], s_chunk.value, s_thr.value, show_chunks=True,
                title=f"You drew {drawn} reversal(s); the measurement counted {r['n_switches']} switch(es) "
                      f"({r['n_ccw']} counter-clockwise + {r['n_cw']} clockwise swings)")
        fig.footer = f"toy data made up from the angles you typed, nothing recorded · {_settings_text(s_chunk.value, s_thr.value)}"
        fig.render()

    for c in (txt, s_leg, s_wob, s_thr, s_chunk):
        c.observe(draw, "value")
    draw()
    return w.VBox([txt, w.HBox([s_leg, s_wob]), w.HBox([s_thr, s_chunk]), msg, fig.image])


def _episode_table(run: explore.RunData, chunk: int, min_swing: float) -> str:
    rows = []
    for e in run.episodes:
        r = e.analyze(chunk, min_swing)
        rows.append(f"<tr><td style='padding:1px 12px'>{e.idx}</td><td style='padding:1px 12px'>{e.seed}</td>"
                    f"<td style='padding:1px 12px;color:{P.color(e.label)}'><b>{e.label}</b></td>"
                    f"<td style='padding:1px 12px'>{r['n_ccw']} ↺ / {r['n_cw']} ↻</td>"
                    f"<td style='padding:1px 12px'><b>{r['n_switches']}</b></td></tr>")
    head = "".join(f"<th style='padding:1px 12px;text-align:left'>{h}</th>" for h in
                   ("episode", "seed", "outcome", "swings (↺ counter-clockwise / ↻ clockwise)", "switches"))
    return f"<table style='font-size:12px;border-collapse:collapse'><tr>{head}</tr>{''.join(rows)}</table>"


def threshold_panel(run: explore.RunData):
    """Paths + threshold sweep; drag min_swing / chunk and watch switch counts per outcome."""
    w = _w()
    thr = np.round(np.arange(0.1, 1.55, 0.05), 2)
    s_thr = w.FloatSlider(value=sides.MIN_SWING, min=0.1, max=1.5, step=0.05, description="min swing", readout_format=".2f")
    s_chunk = w.IntSlider(value=sides.CHUNK, min=2, max=24, description="chunk")
    c_sw = w.Checkbox(value=True, description="mark switches")
    c_ch = w.Checkbox(value=False, description="show re-plan lines")
    fig = _Figure(1, 2, (13, 4.4))
    table = w.HTML()

    def draw(*_):
        a, b = fig.axes
        a.clear()
        b.clear()
        P.paths(a, run.episodes, s_chunk.value, s_thr.value, c_sw.value, c_ch.value,
                title="Each line = one episode; open dot = a switch")
        P.threshold_sweep(b, run, thr, s_thr.value, s_chunk.value)
        fig.footer = (f"data: {_short(run, 'episodes', 'ep_*.npz')} (agent and T position per step; index: "
                      f"{_short(run, 'episodes.jsonl')}) · {_settings_text(s_chunk.value, s_thr.value)}")
        fig.render()
        table.value = _episode_table(run, s_chunk.value, s_thr.value)

    for c in (s_thr, s_chunk, c_sw, c_ch):
        c.observe(draw, "value")
    draw()
    return w.VBox([w.HBox([s_thr, s_chunk]), w.HBox([c_sw, c_ch]), fig.image, table])


# ---------------------------------------------------------------------------------------------
# Replay

def _load_video(path: Path, max_side: int = 240) -> np.ndarray:
    """All frames of an mp4, shrunk so one episode stays small in memory."""
    import imageio.v2 as iio

    frames = []
    reader = iio.get_reader(str(path))
    try:
        for fr in reader:
            k = max(1, max(fr.shape[:2]) // max_side)
            frames.append(np.ascontiguousarray(fr[::k, ::k, :3]))
    finally:
        reader.close()
    return np.stack(frames)


def replay_panel(run: explore.RunData):
    """One episode: video (or schematic) beside its angle line, with a step slider."""
    w = _w()
    ep_pick = w.Dropdown(options=[(f"{e.idx}  {e.label}  seed {e.seed}" + ("" if e.video else "  (no video)"), e.idx)
                                  for e in run.episodes], description="episode")
    s_t = w.IntSlider(value=0, min=0, max=run.episodes[0].n_steps - 1, description="step", layout=w.Layout(width="600px"))
    s_thr = w.FloatSlider(value=sides.MIN_SWING, min=0.1, max=1.5, step=0.05, description="min swing")
    play = w.Play(interval=120, min=0, max=s_t.max, step=1)
    w.jslink((play, "value"), (s_t, "value"))
    fig = _Figure(1, 2, (12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    cache: dict[int, np.ndarray] = {}
    last_key: list = [None]

    def frames_for(e):
        if e.video is None:
            return None
        if e.idx not in cache:
            cache.clear()  # keep one episode in memory
            cache[e.idx] = _load_video(e.video)
        return cache[e.idx]

    def draw(*_):
        e = run.get(ep_pick.value)
        fr = frames_for(e)
        t = min(s_t.value, e.n_steps - 1)
        P.scene(fig.axes[0], e, t, None if fr is None else fr[min(t, len(fr) - 1)])
        P.angle_with_cursor(fig.axes[1], e, t, sides.CHUNK, s_thr.value)
        src = (f"video: {_short(run, 'videos', f'ep_{e.idx:05d}.mp4')}" if e.video
               else "no video for this episode")
        fig.footer = (f"{src} · angle from {_short(run, 'episodes', f'ep_{e.idx:05d}.npz')} · "
                      f"{_settings_text(sides.CHUNK, s_thr.value)}")
        key = (e.idx, s_thr.value)
        fig.render(relayout=key != last_key[0])  # moving only the step keeps the layout
        last_key[0] = key

    def on_episode(*_):
        e = run.get(ep_pick.value)
        s_t.max = play.max = e.n_steps - 1
        draw()

    ep_pick.observe(on_episode, "value")
    s_t.observe(draw, "value")
    s_thr.observe(draw, "value")
    draw()
    return w.VBox([w.HBox([ep_pick, s_thr]), w.HBox([play, s_t]), fig.image,
                   _note("▶ plays frame by frame, each redrawn on demand, so it can lag on a slow machine. "
                         "For smooth playback make an MP4 (next cell).")])


def export_replay(run: explore.RunData, idx: int, min_swing: float = sides.MIN_SWING,
                  chunk: int = sides.CHUNK, out_dir: Path | str = "exports", fps: int = 10) -> Path:
    """Write the replay (video or schematic beside the angle line) as one smooth MP4.

    WHY not inside the run folder: runs are never edited after they finish (D-004).
    """
    import imageio.v2 as iio

    e = run.get(idx)
    frames = _load_video(e.video) if e.video else None
    fig = _Figure(1, 2, (12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(run.run_dir).name}_ep{idx:05d}_swing{min_swing:.2f}_chunk{chunk}.mp4"
    imgs = []
    for t in range(e.n_steps):
        P.scene(fig.axes[0], e, t, None if frames is None else frames[min(t, len(frames) - 1)])
        P.angle_with_cursor(fig.axes[1], e, t, chunk, min_swing)
        fig.footer = (f"{_short(run, 'videos', f'ep_{idx:05d}.mp4') if e.video else 'schematic'} · "
                      f"{_short(run, 'episodes', f'ep_{idx:05d}.npz')} · {_settings_text(chunk, min_swing)}")
        imgs.append(fig.rgb())
    iio.mimwrite(str(path), imgs, fps=fps, macro_block_size=16)
    return path


def mp4_panel(run: explore.RunData, out_dir: Path | str = "exports"):
    """Pick an episode and a threshold, press the button, get a smooth MP4 shown right here."""
    w = _w()
    from IPython.display import Video, display

    ep_pick = w.Dropdown(options=[(f"{e.idx}  {e.label}  seed {e.seed}", e.idx) for e in run.episodes], description="episode")
    s_thr = w.FloatSlider(value=sides.MIN_SWING, min=0.1, max=1.5, step=0.05, description="min swing")
    btn = w.Button(description="Make MP4", button_style="primary", icon="film")
    out = w.Output()

    def go(_):
        btn.disabled = True
        with out:
            out.clear_output()
            print("Rendering (about 20-60 s for 300 steps)...")
        try:
            path = export_replay(run, ep_pick.value, s_thr.value, out_dir=out_dir)
            with out:
                out.clear_output()
                print(f"Saved {path.resolve()}")
                display(Video(str(path), embed=True, html_attributes="controls loop width=960"))
        except Exception as err:  # show it in the notebook instead of a silent button
            with out:
                out.clear_output()
                print(f"Could not make the MP4: {err!r}")
        finally:
            btn.disabled = False

    btn.on_click(go)
    return w.VBox([w.HBox([ep_pick, s_thr, btn]), out])


# ---------------------------------------------------------------------------------------------

def activation_panel(run: explore.RunData):
    """Pick an episode and a recorded layer; see how the layer moves over the episode."""
    w = _w()
    layers = sorted({l for e in run.episodes for l in e.activations()})
    ep_pick = w.Dropdown(options=[(f"{e.idx}  {e.label}", e.idx) for e in run.episodes], description="episode")
    l_pick = w.Dropdown(options=[(f"{LAYER_NAMES.get(l, l)}  [{l}]", l) for l in layers], description="layer",
                        layout=w.Layout(width="560px"))
    s_t = w.IntSlider(value=0, min=0, max=run.episodes[0].n_steps - 1, description="step", layout=w.Layout(width="600px"))
    fig = _Figure(1, 3, (15, 4.4))
    last_key: list = [None]

    def draw(*_):
        e = run.get(ep_pick.value)
        s_t.max = e.n_steps - 1
        a, b, c = fig.axes
        P.activation_view(a, b, e, l_pick.value, s_t.value, LAYER_NAMES.get(l_pick.value))
        P.angle_with_cursor(c, e, s_t.value)
        fig.footer = (f"activations: {_short(run, 'episodes', f'ep_{e.idx:05d}.npz')} → act/{l_pick.value} "
                      f"(one value per re-plan, last denoising pass) · angle: same file, trace/pos_agent, trace/block_pose")
        key = (e.idx, l_pick.value)
        fig.render(relayout=key != last_key[0])
        last_key[0] = key

    for c in (ep_pick, l_pick, s_t):
        c.observe(draw, "value")
    draw()
    return w.VBox([w.HBox([ep_pick, l_pick]), s_t, fig.image])
