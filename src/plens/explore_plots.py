"""Matplotlib drawings for explore.py data. Each takes an Axes and draws on it (D-010).

Every plot says in its own title and axis labels what one line, dot or colour means, so a
figure can be read without the notebook text next to it.
"""

from __future__ import annotations

import numpy as np

from . import sides
from .explore import EpisodeView, RunData, activation_change, activation_projection, sweep_by_label

LABEL_COLORS = {"success": "tab:green", "near_miss": "tab:orange", "fail": "tab:red", "toy": "tab:blue"}
ANGLE_LABEL = "angle around the T (rad)\n↑ counter-clockwise   ↓ clockwise"


def color(label: str) -> str:
    return LABEL_COLORS.get(label, "tab:gray")


def paths(ax, eps: list[EpisodeView], chunk: int = sides.CHUNK, min_swing: float = sides.MIN_SWING,
          show_switches: bool = True, show_chunks: bool = False, title: str | None = None) -> None:
    """Angle around the T vs step, one line per episode; open dots mark switches.

    The start height of a line only reflects where the robot happened to start; what matters
    is its shape: climbing, falling, and where it turns around.
    """
    many = len(eps) > 1
    for e in eps:
        ax.plot(e.phi, color=color(e.label), alpha=0.85, lw=1.6 if e.label != "success" else 1.2)
        if many:  # name the line at its right end so a line can be traced back to its episode
            ax.annotate(str(e.idx), (e.n_steps - 1, e.phi[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=color(e.label), va="center")
        if show_switches:
            s = e.switch_steps(chunk, min_swing)
            ax.plot(s, e.phi[[min(i, e.n_steps - 1) for i in s]], "o", color=color(e.label),
                    mfc="white", ms=7, mew=1.5)
    if show_chunks:
        for x in range(0, max(e.n_steps for e in eps), chunk):
            ax.axvline(x, color="0.8", lw=0.5, zorder=0)
    ax.margins(x=0.05)
    ax.set_xlabel("environment step (time →)")
    ax.set_ylabel(ANGLE_LABEL, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    handles = [ax.plot([], [], color=color(l), label=l)[0] for l in dict.fromkeys(e.label for e in eps)]
    if show_switches:
        handles.append(ax.plot([], [], "o", color="0.3", mfc="white", mew=1.5, label="switch")[0])
    if show_chunks:
        handles.append(ax.plot([], [], color="0.7", lw=1, label=f"re-plan (every {chunk} steps)")[0])
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.8)


def threshold_sweep(ax, run: RunData, thresholds: np.ndarray, current: float, chunk: int = sides.CHUNK,
                    title: str | None = "Mean switches per episode, by outcome") -> None:
    """Mean switches per outcome vs the swing threshold; vertical line = the slider's value."""
    for lab, y in sweep_by_label(run, thresholds, chunk).items():
        ax.plot(thresholds, y, "-o", ms=3, color=color(lab), label=f"{lab} (n={len(run.by_label()[lab])})")
    ax.axvline(current, color="k", ls="--", lw=1, label="slider")
    ax.set(xlabel="swing threshold (rad): a turn must be at least this big to count",
           ylabel="switches per episode (mean)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)


def scene(ax, e: EpisodeView, t: int, video_frame: np.ndarray | None = None) -> None:
    """The robot at step t: the recorded video frame if there is one, else a schematic."""
    ax.clear()
    ax.set_axis_off()
    if video_frame is not None:
        ax.imshow(video_frame)
        ax.set_title("recorded video", fontsize=8)
        return
    pa, bp = e.trace["pos_agent"], e.trace["block_pose"]
    ax.plot(*pa[: t + 1].T, color="0.8", lw=1)
    ax.plot(*bp[: t + 1, :2].T, color="tab:blue", lw=1, alpha=0.4)
    ax.plot(*bp[t, :2], "s", color="tab:blue", ms=14, label="T centre")
    d = 60 * np.array([np.cos(bp[t, 2]), np.sin(bp[t, 2])])
    ax.plot([bp[t, 0], bp[t, 0] + d[0]], [bp[t, 1], bp[t, 1] + d[1]], color="tab:blue", lw=2)
    ax.plot(*pa[t], "o", color="k", ms=9, label="agent")
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect("equal")
    ax.set_title("schematic (no video was recorded for this episode)\nblack dot = robot, blue square = centre of the T\n"
                 "blue line = direction the T points, grey = path so far", fontsize=7)


def angle_with_cursor(ax, e: EpisodeView, t: int, chunk: int = sides.CHUNK,
                      min_swing: float = sides.MIN_SWING):
    """One episode's angle line with a vertical cursor at step t. Returns the cursor line."""
    ax.clear()
    paths(ax, [e], chunk, min_swing, show_chunks=True)
    cursor = ax.axvline(t, color="k", lw=1.5)
    ax.set_title(f"episode {e.idx} ({e.label}, seed {e.seed}), step {t}", fontsize=9)
    return cursor


def activation_view(ax_proj, ax_jump, e: EpisodeView, layer: str, t: int | None = None,
                    layer_name: str | None = None) -> None:
    """Layer trajectory over the episode: PCA path (left) and per-re-plan jump size (right)."""
    act = e.activations()[layer]
    order = np.argsort(act["step"], kind="stable")
    steps, vals = act["step"][order], act["value"][order]
    xy, var = activation_projection(vals, 2)
    if xy.shape[1] < 2:
        xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])))
    ax_proj.clear()
    ax_proj.scatter(xy[:, 0], xy[:, 1], c=steps, cmap="viridis", s=18)
    ax_proj.plot(xy[:, 0], xy[:, 1], color="0.7", lw=0.8, zorder=0)
    if t is not None:
        i = int(np.searchsorted(steps, t, side="right") - 1)
        ax_proj.plot(*xy[max(i, 0)], "o", mfc="none", mec="k", ms=12, mew=2)
    ax_proj.set(title=f"{layer_name or layer}: its path through the episode\n"
                      f"2 summary axes (PCA) keep {var.sum():.0%} of the variation; dark → light = time; ring = now",
                xlabel="summary axis 1", ylabel="summary axis 2")
    ax_proj.title.set_fontsize(8)
    ax_jump.clear()
    ax_jump.plot(steps[1:], activation_change(vals), "-o", ms=3, color=color(e.label))
    ax_jump.set(title="how much the layer changed at each re-plan\n(high = it saw or planned something different)",
                xlabel="environment step", ylabel="size of change")
    ax_jump.title.set_fontsize(8)
