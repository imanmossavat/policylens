"""robosuite Square with DPPO's pretrained diffusion policy (plens-dppo env; D-014).

  python scripts/square_eval.py replay --n 300 --name square_demo_replay     # env check, no policy
  python scripts/square_eval.py rollouts --n 20 --batch-size 10 --name square_dppo_pilot
  python scripts/square_eval.py rollouts --n 200 --batch-size 50 --video 6 --name square_dppo_n200
  python scripts/square_eval.py rollouts --n 10 --repeat 8 --name square_dppo_dice   # 10 starts x 8 noise draws
  python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/<id>/checkpoint.pt --name sq_eval_x

Episode i uses start seed --seed + i. Batches of --batch-size share one policy call per re-plan
and one torch noise seed (first seed of the batch + --noise-seed-offset), as in G-008.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from plens import labels, square, store
from plens.runmeta import new_run, update_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["replay", "rollouts"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--repeat", type=int, default=1,
                   help="Run each start this many times in one batch (only the sampling noise differs); overrides --batch-size")
    p.add_argument("--noise-seed-offset", type=int, default=0)
    p.add_argument("--weights", choices=["ema", "model"], default="ema", help="Which checkpoint weights (load_policy)")
    p.add_argument("--checkpoint", default=None, help="A plens checkpoint from square_train.py; default: DPPO's (D-014)")
    p.add_argument("--max-steps", type=int, default=square.MAX_STEPS)
    p.add_argument("--video", type=int, default=0, help="Render the first N episodes to videos/")
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = new_run("runs", args.name or f"square_{args.mode}", vars(args) | {
        "checkpoint": args.checkpoint or str(square.CKPT.relative_to(square.ROOT)), "env_kwargs": square.ENV_KWARGS,
        "policy_cfg": square.POLICY_CFG})
    print(run_dir)
    t0 = time.time()
    try:
        if args.mode == "replay":
            rows = square.replay_demos(n=args.n)
            with open(run_dir / store.INDEX_FILE, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            k = sum(r["success"] for r in rows)
            summary = {"n_demos": len(rows), "replay_success": k, "rate": round(k / len(rows), 4)}
        else:
            if args.checkpoint:
                from plens.train_square import load_trained
                model = load_trained(args.checkpoint, device=args.device, weights=args.weights)
            else:
                model = square.load_policy(device=args.device, weights=args.weights)
            size = args.repeat if args.repeat > 1 else args.batch_size
            envs = [square.make_env() for _ in range(size)]
            if args.repeat > 1:
                batches = [[s] * args.repeat for s in range(args.seed, args.seed + args.n)]
            else:
                batches = [list(range(b, min(b + size, args.seed + args.n)))
                           for b in range(args.seed, args.seed + args.n, size)]
            idx = 0
            for seeds in batches:
                eps = square.rollout_batch(model, envs, seeds, noise_seed=seeds[0] + args.noise_seed_offset,
                                           max_steps=args.max_steps)
                for ep in eps:
                    store.save_npz_episode(run_dir, idx, ep.arrays(), {
                        **ep.summary(), "label": labels.outcome_label(ep.success, ep.progress)})
                    if idx < args.video:
                        store.save_video(run_dir, idx, square.render_states(ep.trace["sim_state"], every=2), fps=10)
                    idx += 1
                done = store.read_index(run_dir)
                print(f"{idx}/{args.n * args.repeat} episodes, success {sum(r['success'] for r in done)}, {time.time() - t0:.0f}s",
                      flush=True)
            summary = labels.summarize(store.read_index(run_dir))
        update_meta(run_dir, status="done", summary=summary, wall_s=round(time.time() - t0))
        print(json.dumps(summary))
    except BaseException as e:
        update_meta(run_dir, status="failed", error=traceback.format_exc(), wall_s=round(time.time() - t0))
        raise e


if __name__ == "__main__":
    main()
