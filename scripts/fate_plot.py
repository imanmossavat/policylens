"""Draw a fate run (scripts/fate.py) and print how each episode's fate was decided.

  python scripts/fate_plot.py runs/<fate run>        # writes exports/<run>_fate.png (D-011)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plens import explore_plots as ep  # noqa: E402
from plens.branch import fate_stats, load_fate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--out", default="exports")
    args = p.parse_args()
    run = Path(args.run)
    curves = load_fate(run)
    print(f"{'seed':>5} {'parent':9} {'p0':>5} {'p_end':>5} {'largest change':>15} {'at step':>7} "
          f"{'decisive':>8} {'sum|dp|':>7} {'noise':>6} {'sealed at':>9}")
    for idx, c in sorted(curves.items()):
        s = fate_stats(c)
        print(f"{c['seed']:5d} {c['label']:9} {s['p0']:5.2f} {s['p_end']:5.2f} {s['largest_change']:+15.2f} "
              f"{s['largest_change_step']!s:>7} {s['decisiveness']:8.2f} {s['sum_abs_change']:7.2f} "
              f"{s['noise_floor_sum']:6.2f} {s['sealed_step']!s:>9}")
    sp = np.concatenate([c["plan_spread"] for c in curves.values()])
    un = np.concatenate([c["p"] * (1 - c["p"]) for c in curves.values()])
    rank = lambda x: np.argsort(np.argsort(x))
    print(f"Spearman(plan spread, p(1-p)) over {len(sp)} states: {np.corrcoef(rank(sp), rank(un))[0, 1]:.2f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ep.fate_curves(a, curves, "Fate curves: from each re-plan, the share of 50 fresh-noise branches that succeed")
    ep.spread_vs_uncertainty(b, curves, "Does the policy's own disagreement flag open fates?")
    fig.text(0.01, 0.01, f"source: {run}/fate.jsonl", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(args.out).mkdir(exist_ok=True)
    out = Path(args.out) / f"{run.name}_fate.png"
    fig.savefig(out, dpi=150)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
