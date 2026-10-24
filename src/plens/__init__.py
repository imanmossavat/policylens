"""plens: run pretrained robot policies, record their internals, analyze failures.

Index of modules (grep the name to jump):
  runmeta   - run folders + meta.json (what ran, with which code, versions, seeds)
  policy    - load a pretrained or random-weight LeRobot policy + processors; set_sampler
  envs      - build envs; TaskSpec per task (PushT, ALOHA insertion / transfer cube): what to record, what progress means
  record    - ActivationRecorder: forward hooks on named layers
  rollout   - run_batch: N envs in lockstep, one Episode each (D-012); run_episode = batch of 1
  store     - save/load episodes and the per-run episode index
  labels    - outcome labels, run summaries, paired run comparison
  stats     - Wilson intervals, exact McNemar, bootstrap, episodes needed for a precision
  explore   - re-computable views of recorded runs; explore_plots / explore_ui: figures and sliders (D-010)
  sides     - which way around the T the agent goes, and whether it switches (Q-001)
  branch    - copy the simulator mid-episode, run K continuations: fate curves (Q-011)
  probe     - linear probes on activations for simulator ground truth, split by episode (Q-012)
  disassembly - ALOHA extraction: frozen assembled states, scripted extractors, demos, evaluation (F-008/F-009)
  forcecheck - robosuite Square wrist force sanity check (plens-robosuite env)
"""

__version__ = "0.1.0"
