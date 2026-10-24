# Tasks and the assembly difficulty each isolates (D-001)

Difficulties of assembly (tag as diff:X):
- diff:A multimodal choices: several valid approaches/grasps; averaging them does neither
- diff:B precise final alignment: millimetre fit before insertion
- diff:C contact dynamics: pushing, sliding, jamming; errors grow once parts touch
- diff:D compounding errors: small early mistakes lead to states never seen in training
- diff:E sequencing and memory: knowing which step you are in
- diff:F detecting being stuck: noticing a jam instead of pushing on

| Task | Status | Isolates | Why it is here |
|---|---|---|---|
| PushT, Diffusion Policy | active | diff:A diff:B diff:C | Go around the T left or right (A); success needs 95% overlap (B); pushing only (C) |
| ALOHA cube transfer, ACT | active | diff:D | ACT predicts action chunks to fight compounding error; attention is inspectable |
| ALOHA insertion (peg in socket), ACT | active | diff:B diff:C | First real assembly task: pretrained, bimanual, contact force readable in MuJoCo (D-013) |
| ALOHA extraction (disassembly) | needs redesign | diff:F diff:C | Insertion reversed from frozen assembled states; trivial as built (F-009): needs a constrained pull before training a policy |
| Meta-World assembly / disassemble (peg and ring) | planned | diff:F diff:B | Ready-made pair with scripted experts; lerobot/smolvla_metaworld covers both; run the triviality check first |
| robomimic Square (nut on peg) | planned | diff:B diff:C | Panda in robosuite; runs with mujoco 3.3.0, wrist force/torque available, but needs a trained policy |
| Square reversed (disassembly) | planned | diff:F | Remove nut from peg; compare with assembly representations |
| robomimic Tool Hang | optional | diff:E diff:B | Multi-step precise insertion |
| LIBERO-Long | optional | diff:E | Only if memory matters before Square |
