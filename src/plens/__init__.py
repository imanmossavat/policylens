"""plens: run pretrained robot policies, record their internals, analyze failures.

Index of modules (grep the name to jump):
  runmeta   - run folders + meta.json (what ran, with which code, versions, seeds)
  policy    - load a pretrained or random-weight LeRobot policy + processors; set_sampler
  envs      - build the PushT env
  record    - ActivationRecorder: forward hooks on named layers
  rollout   - run_batch: N envs in lockstep, one Episode each (D-012); run_episode = batch of 1
  store     - save/load episodes and the per-run episode index
  labels    - outcome labels, run summaries, paired run comparison
  stats     - Wilson intervals, exact McNemar, bootstrap, episodes needed for a precision
  explore   - re-computable views of recorded runs; explore_plots / explore_ui: figures and sliders (D-010)
  sides     - which way around the T the agent goes, and whether it switches (Q-001)
"""

__version__ = "0.1.0"
