"""Check wrist force sensing in robosuite Square (run in the plens-robosuite env).

  python scripts/square_force.py
"""

from __future__ import annotations

import argparse

import numpy as np

from plens.forcecheck import press_test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps-down", type=int, default=45)
    args = p.parse_args()
    f = press_test(args.steps_down)
    print("|wrist force| (N), every 6th step:", np.round(f[::6], 1))
    print(f"free (first 10 steps) median {np.median(f[:10]):.1f} N; pressing (last 15 of descent) median "
          f"{np.median(f[args.steps_down - 15:args.steps_down]):.1f} N")


if __name__ == "__main__":
    main()
