"""Print what a run was and how it went. Defaults to the latest run.

  python scripts/summarize.py
  python scripts/summarize.py runs/20261020-120000_baseline --episodes
"""

from __future__ import annotations

import argparse
import json

from plens import labels, store
from plens.runmeta import read_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="Run folder (default: latest in runs/)")
    p.add_argument("--episodes", action="store_true", help="Also list every episode")
    args = p.parse_args()

    run_dir = args.run or store.latest_run()
    meta = read_meta(run_dir)
    index = store.read_index(run_dir)
    pol = meta.get("policy", {})
    git = (meta["environment"].get("git") or {})
    print(f"run      {meta['run_id']}  [{meta['status']}]")
    print(f"policy   {pol.get('repo_id') or pol.get('kind')}  commit={pol.get('commit')}  device={pol.get('device')}")
    print(f"code     git={git.get('commit', 'n/a')}{' (dirty)' if git.get('dirty') else ''}")
    print(f"summary  {json.dumps(labels.summarize(index))}")
    if args.episodes:
        for line in index:
            print(f"  {line['idx']:4d} seed {line['seed']:4d}  {line['label']:9s} "
                  f"max_cov {line['max_coverage']:.3f}  steps {line['n_steps']}")


if __name__ == "__main__":
    main()
