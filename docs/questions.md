# Research questions (one line each; status: open / active / answered by F-###)
Q-001 open diff:A In failures, does the policy commit to one way around the T, or switch between ways from one re-plan to the next?
Q-002 open diff:B In near-misses, is the alignment error already in the plan (denoiser output) or does it appear only during execution?
Q-003 open diff:B Can a linear probe on image features read block pose? If probes stay accurate in failures, failures are planning, not perception.
Q-004 open diff:F Can a probe on activations predict failure k steps before it happens (early warning)? How large can k be?
Q-005 open diff:D Do failures start at re-plan boundaries (every 8 steps) or inside a chunk?
Q-006 open Fine-tune on demos from failure states: which layers change most, and do the early-warning signals of Q-004 vanish?
Q-007 open diff:B Does a Square (nut-on-peg) policy encode "aligned with peg" the way PushT encodes "aligned with target"?
Q-008 open diff:F Do assembly and disassembly policies for Square share representations of alignment?
