"""Does the policy's vision encode contact it never senses? Probe a run's activations (Q-012).

  python scripts/probe_contact.py runs/<aloha_insertion run with per-step activations>

Needs per-step records (e.g. --temporal-ensemble 0.01, which re-plans every step). Target:
peg-socket contact from the simulator. Features: each recorded layer, the joint positions the
policy is given (agent_pos), and both. Test AUC by parent outcome, over the late phase only
(both parts grasped, progress >= 0.5), where contact decides success.
"""

from __future__ import annotations

import argparse

import numpy as np

from plens import store
from plens.probe import probe


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--min-progress", type=float, default=0.5)
    p.add_argument("--target", choices=["contact", "force"], default="contact",
                   help="contact: is the peg touching the socket; force: among contact steps, is the "
                        "contact force above its median (pushing hard vs touching)")
    args = p.parse_args()

    index = store.read_index(args.run)
    feats: dict[str, list] = {}
    target, episode, label = [], [], []
    for line in index:
        ep = store.load_episode(args.run, line["idx"])
        tr, act = ep["trace"], ep["act"]
        keep = np.flatnonzero(tr["progress"] >= args.min_progress)
        keep = keep[keep < len(tr["peg_socket_contacts"])]
        if args.target == "force":
            keep = keep[tr["peg_socket_contacts"][keep] > 0]
        if len(keep) == 0:
            continue
        for layer, a in act.items():
            by_step = dict(zip(a["step"].tolist(), a["value"][:, 0]))
            feats.setdefault(layer, []).append(np.stack([by_step[t] for t in keep]))
        feats.setdefault("agent_pos (joint state)", []).append(tr["agent_pos"][keep])
        target.append(tr["peg_socket_contacts"][keep] > 0 if args.target == "contact"
                      else tr["peg_socket_force"][keep])
        episode.append(np.full(len(keep), line["idx"]))
        label.append(np.full(len(keep), line["label"]))
    if not target:
        raise SystemExit(f"no steps with progress >= {args.min_progress} and ground truth in {args.run}")
    target, episode, label = map(np.concatenate, (target, episode, label))
    if args.target == "force":
        print(f"contact force: median {np.median(target):.1f} N, 90th percentile {np.percentile(target, 90):.1f} N")
        target = target > np.median(target)
    x = {k: np.concatenate(v) for k, v in feats.items()}
    vision = [k for k in x if k.startswith("model.")]
    if vision:
        x[f"{vision[0]} + agent_pos"] = np.concatenate([x[vision[0]], x["agent_pos (joint state)"]], axis=1)
    groups = {lab: label == lab for lab in ("success", "near_miss", "fail") if (label == lab).any()}
    print(f"{len(set(episode))} episodes, {len(target)} late-phase steps, positive in {target.mean():.0%} ({args.target})")
    print(f"{'features':45s} {'AUC all':>8s} " + " ".join(f"{g:>10s}" for g in groups))
    for name, f in x.items():
        r = probe(f, target.astype(int), episode, groups)
        print(f"{name:45s} {r['all']:8.3f} " + " ".join(f"{r[g]:10.3f}" for g in groups))


if __name__ == "__main__":
    main()
