"""plens: run pretrained robot policies, record their internals, analyze failures.

Index of modules (grep the name to jump):
  runmeta   - run folders + meta.json (what ran, with which code, versions, seeds)
  policy    - load a pretrained or random-weight LeRobot policy + processors
  envs      - build the PushT env
  record    - ActivationRecorder: forward hooks on named layers
  rollout   - run one episode, return an Episode
  store     - save/load episodes and the per-run episode index
  labels    - outcome labels and run summaries
"""

__version__ = "0.1.0"
