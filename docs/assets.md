# Assets on disk: models, checkpoints, data (never in git; all under gitignored .cache/ unless noted)
One line each: what, where, source, sha256 prefix, used by. Add a line when you download something.

| What | Path | Source | sha256 | Used by |
|---|---|---|---|---|
| DPPO Square pretrained diffusion, state, UNet, 20 denoising, plan 4 (epoch 8000) | .cache/dppo/ckpt_square_unet/state_8000.pt | DPPO Google Drive id 11IEgQe0LFI23hn1Cwf6Z_YfJdDilVc0z (script/download_url.py) | ae2f31de | plens/square.py (D-014) |
| DPPO Square obs/action min-max | .cache/dppo/norm_square/normalization.npz (same file in data_square/) | Drive id 1FFMqWVv0145OJjbA_iglkWywmdK22Za- | 68ec0abf | plens/square.py |
| DPPO Square training data (300 MH demos, normalized; = robomimic file below, same order) | .cache/dppo/data_square/train.npz | Drive folder 1mVVNOJ6wt2EXoapF7PKkcqxbsB9gvK-B | 443665cd | training a weaker policy (step 2) |
| robomimic Square MH demos, robosuite 1.4.1, with sim states | .cache/robomimic/square_mh/low_dim_v141.hdf5 | downloads.cs.stanford.edu/downloads/rt_benchmark/square/mh/ | (128 MB) | replay_demos |
| DPPO source (model code only is imported) | .cache/dppo/src | github.com/irom-lab/dppo @ cc7234ad | git | plens/square.py |
| LeRobot policies (PushT diffusion, ALOHA ACT) | ~/.cache/huggingface; migrated copies in .cache/migrated/ | Hugging Face Hub (G-001) | HF revisions in each run's meta.json | plens/policy.py |

Conda envs: lerobot-gestures (PushT, main), plens-aloha (ALOHA/ACT), plens-flow (flow matching), plens-robosuite (robosuite 1.5.2 + mujoco 3.3.0, force checks), plens-dppo (robosuite 1.4.1 + mujoco 3.1.6 + torch 2.4: Square + DPPO). Python: /opt/miniconda3/envs/<env>/bin/python.
