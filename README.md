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
```
First run downloads `lerobot/diffusion_pusht` and converts it once (G-001).
Each episode takes minutes on a Mac (G-006); `--n 50` is better left for the 3090.
Results land in `runs/<run_id>/`; see `LAYOUT` in src/plens/store.py.
