# Research questions (one line each; status: open / active / answered by F-###)
Q-001 active (F-001, F-002, F-003: outcomes mostly set by sampling noise; needs more failures) diff:A In failures, does the policy commit to one way around the T, or switch between ways from one re-plan to the next?
Q-002 open diff:B In near-misses, is the alignment error already in the plan (denoiser output) or does it appear only during execution?
Q-003 open diff:B Can a linear probe on image features read block pose? If probes stay accurate in failures, failures are planning, not perception.
Q-004 open diff:F Can a probe on activations predict failure k steps before it happens (early warning)? How large can k be?
Q-005 open diff:D Do failures start at re-plan boundaries (every 8 steps) or inside a chunk?
Q-006 open Fine-tune on demos from failure states: which layers change most, and do the early-warning signals of Q-004 vanish?
Q-007 open diff:B Does a Square (nut-on-peg) policy encode "aligned with peg" the way PushT encodes "aligned with target"?
Q-008 open diff:F Do assembly and disassembly policies for Square share representations of alignment?
Q-009 answered by F-004, F-006 Is rollout time set by the chain of denoising passes (100 per re-plan, mostly fixed GPU overhead), and what do fewer steps, batched episodes or many plans per state buy? Plain-language: https://claude.ai/artifact/9BYR9vs69p9nkcs71eozGG
Q-010 open diff:A From one frozen state, sample 32 plans in one batched pass: how often does each go left or right around the T, and do failures start where the split is near 50/50? (F-003: noise decides most outcomes.)
Q-011 answered by F-017 (probe part untested) diff:A,F Is an episode's outcome decided at identifiable moments or spread over the episode? Branch K continuations from each re-plan (fate curve); does plan spread or a probe see the drops coming? (scripts/fate.py)
Q-012 active (F-011: contact yes, force no) diff:B,C Does a vision-only policy (ACT, ALOHA insertion) encode peg-socket contact it never senses? Probe activations for simulator contact; if near-misses lack it, force sensing (round 2) should fix exactly those.
Q-013 open diff:D Why does re-planning ACT more often make it crawl (F-013)? Test: does the first part of each chunk lag the demonstrated speed, and does giving it 2 frames (velocity) remove the collapse?
Q-014 open diff:A,F Is the late fade of PushT near-misses (F-017) the 300-step limit or the policy? Re-branch their late snapshots with max_steps 600.
Q-015 open diff:B,D Does ALOHA insertion flip outcomes under tiny start or numeric perturbations as PushT flips under noise (F-018)? ACT has no sampling noise (F-013), so test ~20 repeats on 25-30 stratified starts.
Q-016 open diff:D,F Are Square near misses partly timeouts (F-021)? Ep 6 of 20261028-040459_square_dppo_ema_n300 grasps at step ~280, then bumps the peg at 40-70 N until step 400. Rerun near-miss starts with max_steps 800; does force show the jam first?
