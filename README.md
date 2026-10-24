# policy-lens

Run pretrained robot policies, record what happens inside them, and find out why they fail.
End target: assembly and disassembly. Agents and contributors: start at CLAUDE.md.

## Setup (Mac or Linux, Python 3.12+)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not smoke"   # fast checks; plain `pytest` adds the real env
```

## Run
```bash
python scripts/rollouts.py --n 10 --name first      # pretrained Diffusion Policy on PushT
python scripts/summarize.py --episodes              # latest run: provenance, success rate, labels
python scripts/rollouts.py --list-layers            # layers you can record with --layers
python scripts/rollouts.py --n 500 --batch-size 50 --layers --video 0 --name eval   # many episodes, fast
python scripts/rollouts.py --n 500 --batch-size 50 --sampler ddim --inference-steps 10 --name ddim10
python scripts/compare.py runs/<A> runs/<B>         # success with 95% CIs, paired by seed vs the first
```
First run downloads `lerobot/diffusion_pusht` and converts it once (G-001).
One episode takes ~30-45 s on an M4 Max (G-006). `--batch-size` shares each policy pass between envs:
use it for statistics; keep batch 1 to reproduce a run seed for seed (G-008). Long runs on mps: G-007.
Another LeRobot policy plugin (e.g. `lerobot_policy_flow_matching`) loads with `--policy <repo>` once installed.
Results land in `runs/<run_id>/`; see `LAYOUT` in src/plens/store.py.

## Explore (no robot needed)
```bash
pip install -e ".[explore]" && jupyter lab notebooks/explore.ipynb
```
Sliders over recorded runs: paths around the T, the swing threshold, replay beside the angle plot, a toy, and layer activations.
The notebook starts with what the task and policy are, shows the run's `meta.json` (files, policy checkpoint, code commit, settings),
and every figure carries a footer naming its source file and settings. A replay can be saved as a smooth MP4 in `exports/`.
