"""Run a policy on PushT for N episodes; save traces, activations, videos and metadata.

  python scripts/rollouts.py --n 100 --name baseline
  python scripts/rollouts.py --list-layers
  python scripts/rollouts.py --random-weights --n 3 --max-steps 30   # offline check
"""

from __future__ import annotations

import argparse
import time
import traceback

from plens import labels, store
from plens.envs import make_single_env, pusht_config
from plens.policy import DIFFUSION_PUSHT_LAYERS, load_pretrained, load_random_diffusion
from plens.record import ActivationRecorder, list_layers
from plens.rollout import run_episode
from plens.runmeta import new_run, update_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/diffusion_pusht", help="Hub repo id or local dir")
    p.add_argument("--revision", default=None, help="Hub revision; resolved commit goes in meta.json")
    p.add_argument("--random-weights", action="store_true", help="Untrained policy, no download")
    p.add_argument("--n", type=int, default=50, help="Number of episodes")
    p.add_argument("--seed", type=int, default=0, help="Episode i uses seed + i")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--layers", nargs="*", default=DIFFUSION_PUSHT_LAYERS, help="Layers to record; none to skip")
    p.add_argument("--no-pool", action="store_true", help="Keep full activation shapes (large files)")
    p.add_argument("--all-calls", action="store_true",
                   help="Keep every denoising pass, not just the last (about 100x larger files)")
    p.add_argument("--video", type=int, default=5, help="Save videos of the first N episodes")
    p.add_argument("--device", default="auto")
    p.add_argument("--name", default="pusht")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--list-layers", action="store_true", help="Print hookable layers and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = pusht_config(args.max_steps)
    bundle = (load_random_diffusion(env_cfg, args.device) if args.random_weights
              else load_pretrained(args.policy, env_cfg, args.device, args.revision))

    if args.list_layers:
        for name, cls in list_layers(bundle.policy, max_depth=4):
            print(f"{name}  ({cls})")
        return

    run_dir = new_run(args.runs_dir, args.name, config={
        **vars(args), "near_miss_coverage": labels.NEAR_MISS_COVERAGE})
    update_meta(run_dir, policy=bundle.source)
    print(f"Run: {run_dir}")
    venv = make_single_env(env_cfg)
    started = time.time()
    try:
        for i in range(args.n):
            seed = args.seed + i
            render = i < args.video
            if args.layers:
                with ActivationRecorder(bundle.policy, args.layers, not args.no_pool) as rec:
                    ep = run_episode(bundle, venv, seed, rec, render)
                acts = rec.to_arrays(calls="all" if args.all_calls else "last")
            else:
                ep, acts = run_episode(bundle, venv, seed, None, render), None
            label = labels.outcome_label(ep.success, ep.max_coverage)
            store.save_episode(run_dir, i, ep, label, acts)
            if render:
                store.save_video(run_dir, i, ep.frames, fps=env_cfg.fps)
            print(f"  ep {i:3d} seed {seed:4d}  {label:9s} max_cov {ep.max_coverage:.3f}  steps {ep.n_steps}")
        summary = labels.summarize(store.read_index(run_dir))
        update_meta(run_dir, status="done", summary=summary,
                    duration_s=round(time.time() - started, 1))
        print(summary)
    except BaseException as e:  # record failures too, incl. Ctrl-C
        update_meta(run_dir, status="failed", error=repr(e), traceback=traceback.format_exc(),
                    duration_s=round(time.time() - started, 1))
        raise
    finally:
        venv.close()


if __name__ == "__main__":
    main()
