# Square report: reproduction record (token-dense; for humans read the page). Index of all reports: docs/reports.md
Report (claude.ai, private): https://claude.ai/artifact/KzMdYvkqudvXAziN1SRiBo (v5, 2026-10-28). Page source: exports/square_demo/ (gitignored): _body.html + _style.html, build.py assembles index.html; make_clips.py renders clips; data_brains.json, data_contact.json, clips/force.json are its data.
Code: all runs below were made from uncommitted code; exact code = base commit + runs/<id>/git_diff.patch. All of it is committed in 4026018 (step 1) and c5c2255 (step 2); later code changed only recording (G-014 lift baseline, G-015 eef frame, nut_peg_force trace), not policy actions or env.
Machine: Apple M4 Max, macOS 26.6.2. Decisions D-014..D-019, findings F-020..F-032, gotchas G-012..G-017 (grep the ID).

## Envs (conda, /opt/miniconda3/envs/<env>/bin/python; plens installed with pip install -e . --no-deps)
plens-dppo: py3.12.16 torch==2.4.0 mujoco==3.1.6 robosuite==1.4.1 numpy==1.26.4 einops h5py imageio[ffmpeg] gdown. Rollouts, replay, rendering, analysis. Never train here (G-016).
lerobot-gestures: py3.12.13 torch==2.11.0 numpy==2.2.6 lerobot==0.6.1. Training only (mps).
plens-metaworld: py3.12.16 metaworld==3.1.1 mujoco==3.3.0 gymnasium==1.4.0. Meta-World check only.
Rebuild plens-dppo: conda create -n plens-dppo python=3.12; pip install "torch==2.4.0" "mujoco==3.1.6" "robosuite==1.4.1" "numpy<2" einops h5py "imageio[ffmpeg]" gdown pytest; pip install -e . --no-deps

## Assets (details + sha256 in docs/assets.md)
.cache/dppo/src = github.com/irom-lab/dppo @ cc7234ad (model code only). Checkpoint: gdown 11IEgQe0LFI23hn1Cwf6Z_YfJdDilVc0z -> .cache/dppo/ckpt_square_unet/state_8000.pt. Normalization: gdown 1FFMqWVv0145OJjbA_iglkWywmdK22Za- -> .cache/dppo/norm_square/normalization.npz. Train data: gdown --folder 1mVVNOJ6wt2EXoapF7PKkcqxbsB9gvK-B -> .cache/dppo/data_square/. Demos: curl http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/mh/low_dim_v141.hdf5 -> .cache/robomimic/square_mh/. masks.json = that hdf5's mask/{better,okay,worse} exported once (train_square.select_demos).

## Protocol (shared by all rollouts)
Env kwargs = DPPO cfg/robomimic/env_meta/square.json (square.ENV_KWARGS). Start seed s: np.random.seed(s); env.reset(). Obs = eef_pos, eef_quat, gripper_qpos, object-state (23), min-max to [-1,1]. Plan 4 actions per policy call; max 400 steps unless noted; success = env reward >= 1 (first time). Batch 50 envs lockstep; torch noise seeded once per batch with its first seed (G-008), so a seed's outcome depends on batching. Episode i uses seed i (dice: --seed + i, repeated --repeat times in one batch).
Stages (page sec 6, F-021): lifted = nut z > 0.86 after step 10; over peg = nut xy within 3 cm of peg xy; near miss = over peg, no win. Recompute old runs with square.stage_flags (stored "lifted" in runs before 04:30 UTC is wrong, G-014).
Training: plens/train_square.py, AdamW 1e-4, wd 1e-6, batch 256, warmup 100 + cosine to 0.1x, EMA 0.995 every 10 steps (copy before step 2000), 100k steps, seed 0. Diffusion = DPPO DiffusionModel (eps loss, 20 DDPM steps). Flow = linear path, velocity loss, 10 Euler steps. Eval uses EMA weights.

## Runs (runs/<id>/: meta.json, episodes.jsonl, episodes/*.npz with trace/sim_state etc., git_diff.patch; trained ones: checkpoint.pt)
id | python env | code | command | result | wall
20261028-040352_square_dppo_pilot | plens-dppo | 6931498+patch | `python scripts/square_eval.py rollouts --n 20 --batch-size 10 --video 2 --name square_dppo_pilot` | 7/20 win; labels {'fail': 10, 'near_miss': 3, 'success': 7} | 57s
20261028-040459_square_dppo_ema_n300 | plens-dppo | 6931498+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --video 12 --name square_dppo_ema_n300` | 90/300 win; labels {'fail': 157, 'near_miss': 53, 'success': 90} | 1098s
20261028-040501_square_dppo_model_n300 | plens-dppo | 6931498+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --weights model --name square_dppo_model_n300` | 95/300 win; labels {'fail': 149, 'near_miss': 56, 'success': 95} | 1072s
20261028-040503_square_demo_replay_n300 | plens-dppo | 6931498+patch | `python scripts/square_eval.py replay --n 300 --name square_demo_replay_n300` | replay 211/300 | 462s
20261028-042110_square_dppo_dice_10x8 | plens-dppo | 6931498+patch | `python scripts/square_eval.py rollouts --n 10 --repeat 8 --name square_dppo_dice_10x8` | 21/80 win; labels {'fail': 48, 'near_miss': 11, 'success': 21} | 251s
20261028-044215_square_dppo_ema_n300_t800 | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --max-steps 800 --name square_dppo_ema_n300_t800` | 125/300 win; labels {'fail': 135, 'near_miss': 40, 'success': 125} | 2019s
20261028-044216_square_dppo_dice_40x8_s10 | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --seed 10 --n 40 --repeat 8 --name square_dppo_dice_40x8_s10` | 95/320 win; labels {'fail': 171, 'near_miss': 54, 'success': 95} | 1280s
20261028-045006_sq_diff_n300 | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind diffusion --n-demos 300 --steps 100000 --name sq_diff_n300` | pairs 79831, final loss 0.0319, 2387s | 2388s
20261028-045008_sq_diff_n100 | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind diffusion --n-demos 100 --steps 100000 --name sq_diff_n100` | pairs 26914, final loss 0.0180, 2385s | 2386s
20261028-045010_sq_diff_n50 | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind diffusion --n-demos 50 --steps 100000 --name sq_diff_n50` | pairs 7547, final loss 0.0146, 2392s | 2394s
20261028-045012_sq_flow_n300 | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind flow --n-demos 300 --steps 100000 --name sq_flow_n300` | pairs 79831, final loss 0.0869, 2399s | 2400s
20261028-053027_eval_sq_diff_n300 | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-045006_sq_diff_n300/checkpoint.pt --name eval_sq_diff_n300` | 178/300 win; labels {'fail': 74, 'near_miss': 48, 'success': 178} | 993s
20261028-053029_eval_sq_diff_n100 | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-045008_sq_diff_n100/checkpoint.pt --name eval_sq_diff_n100` | 171/300 win; labels {'fail': 98, 'near_miss': 31, 'success': 171} | 954s
20261028-053031_eval_sq_diff_n50 | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-045010_sq_diff_n50/checkpoint.pt --name eval_sq_diff_n50` | 170/300 win; labels {'fail': 100, 'near_miss': 30, 'success': 170} | 928s
20261028-053033_eval_sq_flow_n300 | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-045012_sq_flow_n300/checkpoint.pt --name eval_sq_flow_n300` | 153/300 win; labels {'fail': 104, 'near_miss': 43, 'success': 153} | 796s
20261028-054822_sq_diff_better | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind diffusion --demos better --steps 100000 --name sq_diff_better` | pairs 18227, final loss 0.0159, 2188s | 2189s
20261028-054824_sq_diff_okay | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind diffusion --demos okay --steps 100000 --name sq_diff_okay` | pairs 26156, final loss 0.0215, 2195s | 2196s
20261028-054826_sq_diff_worse | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind diffusion --demos worse --steps 100000 --name sq_diff_worse` | pairs 35448, final loss 0.0209, 2205s | 2206s
20261028-054828_sq_diff_rand25 | lerobot-gestures | 4026018+patch | `python scripts/square_train.py --kind diffusion --demos random:25 --steps 100000 --name sq_diff_rand25` | pairs 6861, final loss 0.0094, 2193s | 2194s
20261028-062523_eval_sq_diff_better | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-054822_sq_diff_better/checkpoint.pt --name eval_sq_diff_better` | 160/300 win; labels {'fail': 108, 'near_miss': 32, 'success': 160} | 956s
20261028-062525_eval_sq_diff_okay | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-054824_sq_diff_okay/checkpoint.pt --name eval_sq_diff_okay` | 72/300 win; labels {'fail': 155, 'near_miss': 73, 'success': 72} | 1145s
20261028-062527_eval_sq_diff_worse | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-054826_sq_diff_worse/checkpoint.pt --name eval_sq_diff_worse` | 90/300 win; labels {'fail': 145, 'near_miss': 65, 'success': 90} | 1150s
20261028-062529_eval_sq_diff_rand25 | plens-dppo | 4026018+patch | `python scripts/square_eval.py rollouts --n 300 --batch-size 50 --checkpoint runs/20261028-054828_sq_diff_rand25/checkpoint.pt --name eval_sq_diff_rand25` | 48/300 win; labels {'fail': 211, 'near_miss': 41, 'success': 48} | 1146s

## Analyses without their own run folder (all read the runs above; plens-dppo python)
- Stage breakdowns (F-021, F-029/F-032 stages, page sec 6): for each episode square.stage_flags(trace nut_pos, peg_pos); win > over_peg > lifted (steps >= 10) > not_lifted.
- Paired comparisons (F-020, F-029, F-032): labels.compare({name: store.read_index(run)}), first = reference; McNemar exact.
- 800-step check (F-026): every seed's (success and success_step<400) equals 20261028-040459 outcome: 300/300. Late wins grouped by stage in 20261028-040459 (first 401 steps).
- Dice (F-022, F-025, page sec 8): wins per seed summed over 20261028-042110 + 20261028-044216; variance ratio = var(wins, ddof=1) / (8 p (1-p)).
- Wrist force (F-027): square.peg_phase_force(trace) on 20261028-044215; AUC = P(stuck > win) with ties 1/2.
- Contact force (F-028, page sec 9 dots): square.contact_from_states(trace sim_state from first over-peg step) on 20261028-044215; per episode median over steps with force > 0. Saved as exports/square_demo/data_contact.json.
- Contact replication (F-030): same metrics from trace nut_peg_force in 20261028-053027_eval_sq_diff_n300, _n50, _flow_n300.
- Demo-replay detail (G-013, F-023): replay success by demo length from 20261028-040503; first-step error = state after env.step(actions[0]) vs states[1].
- Observation check: obs_vector at demo_1 state 0 equals hdf5 obs (max diff 1e-7); train.npz rows = robomimic demo_i in order, normalized (tests/test_square.py).
- Paper target (D-016): DPPO arXiv 2409.00588 Fig. 5 (MLP start ~0.4) and Fig. 7 (UNet start ~0.2-0.3), read by eye.
- ALOHA aim or stuck (F-024): /opt/miniconda3/envs/lerobot-gestures/bin/python -I logs/peg_in_socket.py runs/20261025-141810_act_insertion_n100_pose (logs/ is untracked; script is the only copy).
- Meta-World (F-031): plens-metaworld python; plens.metaworld_check.run(task, controller, n=50) for disassemble-v3 and peg-unplug-side-v3; controllers expert, grab_pull (switch after 10 grip-positive steps; failed on unplug because its expert grips weakly from the approach on). Expert source: metaworld/policies/sawyer_peg_unplug_side_v3_policy.py.

## Page clips (exports/square_demo/clips; make_clips.py clip RUN IDX NAME [SIZE], sideview 480 cropped, every 2nd step at 20 fps)
win=040459 ep0, win2=040459 ep7, near=040459 ep6, fail=040459 ep2, late=044215 ep6, stuck=044215 ep13, dice_0..7=042110 ep16..23 (seed 2), force.json=make_clips.py force 040459 7,6. Run ids abbreviated: 20261028-<hhmmss>_*.
