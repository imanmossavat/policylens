"""Fate curves: from every re-plan of an episode, branch K continuations and count successes (Q-011).

  python scripts/fate.py --seeds 0 1 2 --branches 50 --sampler ddim --inference-steps 10 --name fate_ddim10
  python scripts/summarize.py runs/<run>                      # parent episodes, as for rollouts
  python scripts/fate_plot.py runs/<run>                      # the fate curves as one figure

Cost: about n_snapshots x branches x remaining steps env steps per parent episode;
~15-20 min per 300-step parent at DDIM-10 with 50 branches on an M4 Max.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from plens import labels, store
from plens.branch import fate_curve
from plens.envs import pusht_config
from plens.policy import SAMPLERS, load_pretrained, set_sampler
from plens.runmeta import new_run, update_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/diffusion_pusht")
    p.add_argument("--revision", default=None)
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="Parent episode seeds")
    p.add_argument("--branches", type=int, default=50, help="Continuations per snapshot")
    p.add_argument("--every", type=int, default=None, help="Snapshot interval; default n_action_steps")
    p.add_argument("--noise-seed", type=int, default=0, help="Branch noise = this + snapshot step")
    p.add_argument("--sampler", choices=SAMPLERS, default=None)
    p.add_argument("--inference-steps", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--name", default="fate")
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()

    env_cfg = pusht_config(args.max_steps)
    bundle = load_pretrained(args.policy, env_cfg, args.device, args.revision)
    sampling = set_sampler(bundle, args.sampler, args.inference_steps)
    every = args.every or sampling["n_action_steps"]
    run_dir = new_run(args.runs_dir, args.name, config={**vars(args), "every": every})
    update_meta(run_dir, policy=bundle.source, sampling=sampling)
    print(f"Run: {run_dir}  {sampling}")
    started = time.time()
    try:
        for i, seed in enumerate(args.seeds):
            parent, rows = fate_curve(bundle, env_cfg, seed, args.branches, every, args.noise_seed)
            store.save_episode(run_dir, i, parent, labels.outcome_label(parent.success, parent.max_coverage))
            with open(run_dir / "fate.jsonl", "a") as f:
                for row in rows:
                    f.write(json.dumps({"idx": i, **row}) + "\n")
            print(f"  seed {seed}: parent {'success' if parent.success else 'fail'} in {parent.n_steps} steps; "
                  f"p(success) by snapshot: {' '.join(f'{r['p']:.2f}' for r in rows)}", flush=True)
        update_meta(run_dir, status="done", summary=labels.summarize(store.read_index(run_dir)),
                    duration_s=round(time.time() - started, 1))
    except BaseException as e:
        update_meta(run_dir, status="failed", error=repr(e), traceback=traceback.format_exc(),
                    duration_s=round(time.time() - started, 1))
        raise


if __name__ == "__main__":
    main()
