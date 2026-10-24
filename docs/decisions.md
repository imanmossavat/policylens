# Decisions (append only, newest last; supersede, never edit)
D-001 2026-10-20 End target is assembly/disassembly; PushT only builds the analysis toolkit. Every stepping-stone task must isolate an assembly difficulty (docs/tasks.md).
D-002 2026-10-20 Analyze pretrained policies first, train later: faster to the interesting questions; LeRobot checkpoints are established and reproducible.
D-003 2026-10-20 Diffusion Policy on PushT first: small, fails ~35% (65.4% success on its model card), runs on a Mac. diff:A,B,C
D-004 2026-10-20 One folder per run with meta.json (code commit, versions, device, config, seeds, outcome); runs are never edited, so findings can cite them.
D-005 2026-10-20 Record diffusion.rgb_encoder (what the policy sees) and diffusion.unet.mid_modules.1 (what it plans), pooled, last denoising pass only. Full detail via --no-pool / --all-calls.
D-006 2026-10-20 Docs: one index (AGENTS.md) + one-line ID'd entries, enforced by tests/test_docs.py. Cheap to read, grep-able, hard to go stale.
D-007 2026-10-20 Library + tests + scripts first; notebooks later, for teaching only, importing from src/.
D-008 2026-10-21 Index lives in CLAUDE.md only (no AGENTS.md); supersedes the file name in D-006.
D-009 2026-10-21 Side = agent angle around the block, in the block frame (pushing is not circling); swing = >=0.5 rad in one 8-step re-plan; switch = swing opposite the previous one.
D-010 2026-10-21 Learning happens on recorded data, not robot runs: explore.py (numpy views) + explore_plots/explore_ui (sliders) + notebooks/explore.ipynb that only calls them. Supersedes the "notebooks later" in D-007.
D-011 2026-10-21 Notebook figures are drawn off-screen and shown as an image replaced in place (no flicker), each with a footer naming its source files and settings; MP4 exports go to exports/, never into a run (D-004).
D-012 2026-10-23 Rollouts run N envs in lockstep through one batched policy call (run_batch); batch 1 is the old path and reproduces earlier runs exactly. Chosen over per-env noise generators, which would need LeRobot changes; the cost is G-008.
D-013 2026-10-24 ALOHA insertion before Square as the first assembly task: LeRobot ships a pretrained ACT for it and MuJoCo exposes peg-socket contact force as ground truth; Square has no pretrained policy here. Disassembly = ALOHA extraction from frozen insertion successes.
