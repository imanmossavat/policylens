"""Train a Square policy (diffusion or flow) on the first N DPPO demos (D-017).

Run with the lerobot-gestures python (torch 2.11): torch 2.4's mps backward is wrong (G-016).
Evaluate the result with the plens-dppo python: square_eval.py rollouts --checkpoint runs/<id>/checkpoint.pt

  python scripts/square_train.py --kind diffusion --n-demos 100 --steps 60000 --name sq_diff_n100
  python scripts/square_train.py --kind diffusion --demos worse --steps 100000 --name sq_diff_worse
"""

from __future__ import annotations

import argparse
import time
import traceback

from plens import train_square
from plens.runmeta import new_run, update_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["diffusion", "flow"], default="diffusion")
    p.add_argument("--n-demos", type=int, default=300, help="First N demos (ignored with --demos)")
    p.add_argument("--demos", default=None, help='"first:N", "random:N" or an operator group: better / okay / worse')
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    run_dir = new_run("runs", args.name or f"sq_{args.kind}_n{args.n_demos}", vars(args))
    print(run_dir, flush=True)
    t0 = time.time()
    try:
        demos = train_square.select_demos(args.demos, args.seed) if args.demos else args.n_demos
        if not isinstance(demos, int):
            update_meta(run_dir, demo_ids=demos)
        out = train_square.train(args.kind, demos, args.steps, run_dir / "checkpoint.pt", device=args.device,
                                 seed=args.seed, log=lambda s: print(s, flush=True))
        update_meta(run_dir, status="done", summary=out, wall_s=round(time.time() - t0))
    except BaseException:
        update_meta(run_dir, status="failed", error=traceback.format_exc(), wall_s=round(time.time() - t0))
        raise


if __name__ == "__main__":
    main()
