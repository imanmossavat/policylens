"""ALOHA extraction (disassembly): assembled states, a scripted baseline, demos, and a learned policy.

  # 1. scripted baseline: freeze insertion successes, replay the last N insertion actions backwards
  python scripts/disassembly.py baseline --from-run runs/<act_insertion run> --name extraction
  python scripts/disassembly.py baseline --controller home --assembled runs/<baseline>/assembled.npz  # triviality check
  # 2. demonstrations for training (LeRobot dataset inside the run folder)
  python scripts/disassembly.py demos --from-run runs/<act_insertion run> --name extraction_demos
  # 3. train ACT on them (plens-aloha env has lerobot's training extras)
  lerobot-train --dataset.repo_id=local/aloha_extraction --dataset.root=runs/<demos run>/dataset \\
      --policy.type=act --policy.device=mps --policy.push_to_hub=false --steps=40000 --batch_size=8 \\
      --num_workers=0 --wandb.enable=false --output_dir=runs/<new>_train_act_extraction
  # 4. evaluate the learned policy on assembled states it never trained on
  python scripts/disassembly.py evaluate --policy runs/<train>/checkpoints/last/pretrained_model \\
      --assembled runs/<baseline run>/assembled.npz --name extraction_eval

Stages: 0 dropped a part, 1 held and in contact, 2 out of contact, 3 halfway, 4 extracted.
See plens/disassembly.py for why each step is built the way it is.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from plens import labels, store
from plens.disassembly import (EXTRACT_DISTANCE, collect_assembled, evaluate_extraction, extraction_task,
                               glide_home, load_assembled, make_extraction_envs, record_demos, reverse_replay,
                               save_assembled)
from plens.envs import ALOHA_INSERTION
from plens.policy import load_pretrained
from plens.runmeta import new_run, update_meta


def _seeds(args) -> list[int]:
    return args.seeds or [l["seed"] for l in store.read_index(args.from_run) if l["success"]]


def baseline(args, run_dir) -> dict:
    bundle = load_pretrained(args.policy, ALOHA_INSERTION.make_config(400))
    update_meta(run_dir, policy=bundle.source)
    if args.assembled:
        items = load_assembled(args.assembled)
    else:
        seeds = _seeds(args)
        items = collect_assembled(bundle, seeds)
        print(f"  assembled {len(items)} of {len(seeds)} seeds")
    save_assembled(run_dir / "assembled.npz", items)
    controller = {"reverse": reverse_replay, "home": glide_home}[args.controller]
    results = {}
    venv = make_extraction_envs(items, len(items), args.max_steps)
    for steps in args.replay_steps:
        venv.reset(seed=list(range(len(items))))
        plans = [controller(a, steps) for a in items]
        frames = [[] for _ in range(min(args.video, len(items)))]
        # WHY track only up to the first termination: an extracted env autoresets to its
        # assembled state in the same step (G-004), so later stages belong to a new episode.
        best = np.zeros(len(items), dtype=int)
        live = np.ones(len(items), dtype=bool)
        for t in range(min(steps, args.max_steps)):
            for j in range(len(frames)):
                if live[j]:
                    frames[j].append(venv.envs[j].unwrapped.render())
            _, r, term, trunc, _ = venv.step(np.stack([pl[min(t, len(pl) - 1)] for pl in plans]))
            best[live] = np.maximum(best[live], r[live].astype(int))
            live &= ~(term | trunc)
        results[steps] = {"extracted": int((best == 4).sum()), "n": len(items),
                          "best_stage_counts": np.bincount(best, minlength=5).tolist()}
        for j, fr in enumerate(frames):
            store.save_video(run_dir, steps * 1000 + j, fr, fps=50)  # videos/ep_<N><j>.mp4
        print(f"  {args.controller} {steps:3d} steps: {results[steps]}")
    venv.close()
    with open(run_dir / "extraction.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def demos(args, run_dir) -> dict:
    bundle = load_pretrained(args.policy, ALOHA_INSERTION.make_config(400))
    update_meta(run_dir, policy=bundle.source)
    items = collect_assembled(bundle, _seeds(args))
    save_assembled(run_dir / "assembled.npz", items)
    out = record_demos(items, run_dir / "dataset", replay_steps=max(args.replay_steps))
    print(f"  {out} -> {run_dir / 'dataset'}")
    return out


def evaluate(args, run_dir) -> dict:
    task = extraction_task()
    bundle = load_pretrained(args.policy, task.make_config(args.max_steps))
    update_meta(run_dir, policy=bundle.source)
    items = load_assembled(args.assembled)
    eps = evaluate_extraction(bundle, items, args.max_steps)
    for i, ep in enumerate(eps):
        store.save_episode(run_dir, i, ep, labels.outcome_label(ep.success, ep.max_coverage, task.near_miss))
    summary = labels.summarize(store.read_index(run_dir))
    print(f"  {summary}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["baseline", "demos", "evaluate"])
    p.add_argument("--from-run", default=None, help="Insertion run whose successful seeds to assemble")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--policy", default=ALOHA_INSERTION.policy,
                   help="baseline/demos: the insertion policy; evaluate: the extraction policy")
    p.add_argument("--assembled", default=None, help="assembled.npz to start from (skips re-assembling)")
    p.add_argument("--controller", choices=["reverse", "home"], default="reverse",
                   help="baseline: reverse = replay insertion backwards; home = glide arms home (triviality check)")
    p.add_argument("--replay-steps", type=int, nargs="+", default=[20, 40, 60, 80, 100, 120])
    p.add_argument("--max-steps", type=int, default=200, help="Extraction episode length")
    p.add_argument("--video", type=int, default=3)
    p.add_argument("--name", default="extraction")
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()

    run_dir = new_run(args.runs_dir, args.name, config={**vars(args), "extract_distance_m": EXTRACT_DISTANCE})
    print(f"Run: {run_dir}")
    started = time.time()
    result = {"baseline": baseline, "demos": demos, "evaluate": evaluate}[args.command](args, run_dir)
    update_meta(run_dir, status="done", summary=result, duration_s=round(time.time() - started, 1))


if __name__ == "__main__":
    main()
