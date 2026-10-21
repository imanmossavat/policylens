"""Which way around the T does each episode go, and does it switch between re-plans? (Q-001)

  python scripts/sides.py
  python scripts/sides.py runs/20261021-151911_first --min-swing 0.8
"""

from __future__ import annotations

import argparse
import json

from plens import sides, store


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="Run folder (default: latest in runs/)")
    p.add_argument("--chunk", type=int, default=sides.CHUNK, help="Steps per re-plan")
    p.add_argument("--min-swing", type=float, default=sides.MIN_SWING, help="Radians for a chunk to count")
    args = p.parse_args()

    run_dir = args.run or store.latest_run()
    rows = sides.analyze_run(run_dir, args.chunk, args.min_swing)
    print(f"run {run_dir}  chunk={args.chunk}  min_swing={args.min_swing}")
    for r in rows:
        path = "".join("+" if d > 0 else "-" for _, d in r["swings"]) or "(none)"
        print(f"  {r['idx']:4d} seed {r['seed']:4d}  {r['label']:9s} ccw {r['n_ccw']:2d} cw {r['n_cw']:2d} "
              f"switches {r['n_switches']:2d}  {path}")
    print(json.dumps(sides.by_label(rows)))


if __name__ == "__main__":
    main()
