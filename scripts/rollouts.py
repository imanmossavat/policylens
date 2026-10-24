"""Run a policy on a task for N episodes; save traces, activations, videos and metadata.

  python scripts/rollouts.py --n 100 --name baseline
  python scripts/rollouts.py --n 500 --batch-size 50 --layers --video 0 --name eval   # fast stats
  python scripts/rollouts.py --n 500 --batch-size 50 --sampler ddim --inference-steps 10 --name ddim10
  python scripts/rollouts.py --list-layers
  python scripts/rollouts.py --task aloha_insertion --n 50 --batch-size 10 --name act_insertion
  python scripts/rollouts.py --random-weights --n 3 --max-steps 30   # offline check

--batch-size 1 (default) reproduces earlier runs exactly; larger batches share each
policy pass between envs and agree with them statistically, not seed for seed (G-008).
"""

from __future__ import annotations

import argparse
import time
import traceback

from plens import labels, store
from plens.envs import TASKS, make_envs
from plens.policy import SAMPLERS, load_pretrained, load_random_diffusion, set_sampler
from plens.record import ActivationRecorder, list_layers, select_env
from plens.rollout import run_batch
from plens.runmeta import new_run, update_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=sorted(TASKS), default="pusht", help="ALOHA tasks need the plens-aloha env")
    p.add_argument("--policy", default=None, help="Hub repo id or local dir; default: the task's (envs.TASKS)")
    p.add_argument("--revision", default=None, help="Hub revision; resolved commit goes in meta.json")
    p.add_argument("--random-weights", action="store_true", help="Untrained policy, no download")
    p.add_argument("--n", type=int, default=50, help="Number of episodes")
    p.add_argument("--seed", type=int, default=0, help="Episode i uses seed + i")
    p.add_argument("--batch-size", type=int, default=1, help="Envs stepped together through one policy call")
    p.add_argument("--noise-seed-offset", type=int, default=0,
                   help="Torch seed = first seed of the batch + offset; change it to redraw sampling noise")
    p.add_argument("--sampler", choices=SAMPLERS, default=None, help="Diffusion only; default: checkpoint's")
    p.add_argument("--inference-steps", type=int, default=None, help="Denoising passes / ODE steps per re-plan")
    p.add_argument("--n-action-steps", type=int, default=None, help="Actions executed per re-plan")
    p.add_argument("--temporal-ensemble", type=float, default=None,
                   help="ACT or diffusion: re-plan every step and average chunks (ACT paper: 0.01)")
    p.add_argument("--max-steps", type=int, default=None, help="Default: the task's episode length")
    p.add_argument("--layers", nargs="*", default=None, help="Layers to record; default: the task's; none to skip")
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
    task = TASKS[args.task]
    args.policy = args.policy or task.policy
    args.max_steps = args.max_steps or task.max_steps
    args.layers = list(task.layers) if args.layers is None else args.layers
    env_cfg = task.make_config(args.max_steps)
    bundle = (load_random_diffusion(env_cfg, args.device) if args.random_weights
              else load_pretrained(args.policy, env_cfg, args.device, args.revision))
    sampling = set_sampler(bundle, args.sampler, args.inference_steps, args.n_action_steps,
                           args.temporal_ensemble)

    if args.list_layers:
        for name, cls in list_layers(bundle.policy, max_depth=4):
            print(f"{name}  ({cls})")
        return

    run_dir = new_run(args.runs_dir, args.name, config={
        **vars(args), "near_miss_progress": task.near_miss, "progress": task.note})
    update_meta(run_dir, policy=bundle.source, sampling=sampling)
    print(f"Run: {run_dir}  {sampling}  batch {args.batch_size}")
    started = time.time()
    lockstep_steps = env_steps = 0
    venv = None
    # GOTCHA(G-007): on mps this process's RAM grows with every denoiser pass; for thousands
    # of episodes run several processes (e.g. one per 100 episodes) rather than one long one.
    try:
        for first in range(0, args.n, args.batch_size):
            idxs = list(range(first, min(first + args.batch_size, args.n)))
            seeds = [args.seed + i for i in idxs]
            if venv is None or venv.num_envs != len(seeds):  # the last batch may be smaller
                if venv is not None:
                    venv.close()
                venv = make_envs(env_cfg, len(seeds))
            render = [j for j, i in enumerate(idxs) if i < args.video]
            noise_seed = seeds[0] + args.noise_seed_offset
            if args.layers:
                with ActivationRecorder(bundle.policy, args.layers, not args.no_pool) as rec:
                    eps = run_batch(bundle, venv, seeds, rec, render, noise_seed, task=task)
                acts = rec.to_arrays(calls="all" if args.all_calls else "last")
            else:
                eps, acts = run_batch(bundle, venv, seeds, None, render, noise_seed, task=task), None
            lockstep_steps += max(ep.n_steps for ep in eps)
            for j, (i, ep) in enumerate(zip(idxs, eps)):
                env_steps += ep.n_steps
                label = labels.outcome_label(ep.success, ep.max_coverage, task.near_miss)
                ep_acts = select_env(acts, j, len(seeds), ep.n_steps) if acts else None
                store.save_episode(run_dir, i, ep, label, ep_acts)
                if ep.frames:
                    store.save_video(run_dir, i, ep.frames, fps=env_cfg.fps)
                print(f"  ep {i:3d} seed {ep.seed:4d}  {label:9s} max_progress {ep.max_coverage:.3f}  "
                      f"steps {ep.n_steps}")
        duration = time.time() - started
        summary = labels.summarize(store.read_index(run_dir))
        # WHY: per env step, not per episode: successful episodes end early (Q-009).
        timing = {"batch_size": args.batch_size, "env_steps": env_steps, "lockstep_steps": lockstep_steps,
                  "ms_per_env_step": round(1e3 * duration / env_steps, 2),
                  "episodes_per_min": round(60 * args.n / duration, 2)}
        update_meta(run_dir, status="done", summary=summary, timing=timing, duration_s=round(duration, 1))
        print(summary, timing)
    except BaseException as e:  # record failures too, incl. Ctrl-C
        update_meta(run_dir, status="failed", error=repr(e), traceback=traceback.format_exc(),
                    duration_s=round(time.time() - started, 1))
        raise
    finally:
        if venv is not None:
            venv.close()


if __name__ == "__main__":
    main()
