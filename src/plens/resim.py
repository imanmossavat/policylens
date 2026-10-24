"""Re-simulate ALOHA human demonstrations to add what they never recorded: contact force.

Round 2 of the force question (Q-012, F-011/F-012) needs demonstrations with force. The
LeRobot ALOHA datasets hold images, joint states and actions only; not even the starting
object poses. Steps:

1. calibrate: render the scene with known peg/socket positions and fit, per object, an
   affine map from its colour-blob centroid in the top camera to its table (x, y). The
   objects start flat at a fixed height and orientation (gym_aloha sample_insertion_pose),
   so two numbers per object fully define the start.
2. estimate_start: apply the maps to a demonstration's first frame.
3. replay: start the sim there and play the recorded actions open-loop, recording force.
   If the sim reproduces the recorded joint trajectory and outcome, the force trace is the
   force that demonstration would have produced; if it drifts, the demonstration is not
   reproducible in today's simulator (itself worth knowing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .envs import ALOHA_INSERTION, make_single_env

PEG_XY_RANGE = ((0.1, 0.2), (0.4, 0.6))
SOCKET_XY_RANGE = ((-0.2, -0.1), (0.4, 0.6))
START_Z = 0.05


def masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Red peg and blue socket pixels in an RGB top-camera frame."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 150) & (g < 80) & (b < 80), (b > 120) & (r < 80) & (g < 100)


def _features(mask: np.ndarray) -> np.ndarray:
    yx = np.argwhere(mask)
    cy, cx = yx.mean(0)
    return np.array([cx, cy, 1.0])


@dataclass
class Calibration:
    peg: np.ndarray     # [3, 2] affine map, [cx, cy, 1] -> (x, y)
    socket: np.ndarray
    rms_mm: tuple[float, float]  # held-out error per object

    def estimate(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        red, blue = masks(img)
        return _features(red) @ self.peg, _features(blue) @ self.socket


def _render_start(env, peg_xy, socket_xy) -> np.ndarray:
    from gym_aloha.tasks import sim

    pose = np.r_[peg_xy, START_Z, 1, 0, 0, 0, socket_xy, START_Z, 1, 0, 0, 0]
    u = env.unwrapped
    sim.BOX_POSE[0] = pose
    ts = u._env.reset()  # GOTCHA(G-011): gym-aloha's own reset overwrites BOX_POSE from its seed
    return u._format_raw_obs(ts.observation)["pixels"]["top"]


def calibrate(n: int = 40, seed: int = 0) -> Calibration:
    """Fit pixel -> table maps on n random renders; report error on n/4 held-out renders."""
    env = make_single_env(ALOHA_INSERTION.make_config(10)).envs[0]
    env.reset(seed=0)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n + n // 4):
        peg = rng.uniform(*np.array(PEG_XY_RANGE).T)
        sock = rng.uniform(*np.array(SOCKET_XY_RANGE).T)
        red, blue = masks(_render_start(env, peg, sock))
        rows.append((_features(red), peg, _features(blue), sock))
    env.close()
    fit, test = rows[:n], rows[n:]
    maps, errs = [], []
    for k in (0, 2):
        A = np.stack([r[k] for r in fit]); Y = np.stack([r[k + 1] for r in fit])
        M = np.linalg.lstsq(A, Y, rcond=None)[0]
        At = np.stack([r[k] for r in test]); Yt = np.stack([r[k + 1] for r in test])
        maps.append(M)
        errs.append(float(np.sqrt(((At @ M - Yt) ** 2).sum(1).mean()) * 1000))
    return Calibration(maps[0], maps[1], (errs[0], errs[1]))


def replay(peg_xy, socket_xy, actions: np.ndarray) -> dict[str, np.ndarray]:
    """Play recorded actions open-loop from the given start; per-step ground truth and stage."""
    env = make_single_env(ALOHA_INSERTION.make_config(len(actions) + 1)).envs[0]
    env.reset(seed=0)
    _render_start(env, peg_xy, socket_xy)
    out: dict[str, list] = {}
    for a in actions:
        for k, v in ALOHA_INSERTION.observe(env).items():
            out.setdefault(k, []).append(v)
        _, reward, _, _, _ = env.step(np.asarray(a, dtype=np.float64))
        out.setdefault("reward", []).append(reward)
    env.close()
    return {k: np.asarray(v) for k, v in out.items()}
