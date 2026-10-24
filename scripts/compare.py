"""Compare runs on shared seeds: success with 95% intervals, paired against the first run.

  python scripts/compare.py runs/A_ddpm100 runs/B_ddim10 runs/C_flow
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plens import labels, store
from plens.runmeta import read_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="Run folders; the first is the reference")
    args = p.parse_args()

    indexes = {Path(r).name: store.read_index(r) for r in args.runs}
    metas = {Path(r).name: read_meta(r) for r in args.runs}
    print(f"{'run':45s} {'success':>12s} {'95% CI':>15s} {'ms/step':>8s}  vs first run (paired)")
    for row in labels.compare(indexes):
        lo, hi = row["ci95"]
        ms = (metas[row["name"]].get("timing") or {}).get("ms_per_env_step", float("nan"))
        line = f"{row['name']:45s} {row['k']:4d}/{row['n']:<4d}{row['rate']:5.1%} [{lo:5.1%},{hi:5.1%}] {ms:8.1f}"
        if "vs_ref" in row:
            v = row["vs_ref"]
            dlo, dhi = v["diff_ci"]
            line += (f"  diff {v['diff']:+.1%} [{dlo:+.1%},{dhi:+.1%}]  +{v['only_b']}/-{v['only_a']}"
                     f"  p={v['p_mcnemar']:.3f}  (n={v['n_shared']})")
        print(line)


if __name__ == "__main__":
    main()
