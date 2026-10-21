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
| ALOHA cube transfer, ACT | planned | diff:D | ACT predicts action chunks to fight compounding error; attention is inspectable |
| robomimic Square (nut on peg) | planned | diff:B diff:C | First real assembly task; same toolkit, Panda in robosuite |
| Square reversed (disassembly) | planned | diff:F | Remove nut from peg; compare with assembly representations |
| robomimic Tool Hang | optional | diff:E diff:B | Multi-step precise insertion |
| LIBERO-Long | optional | diff:E | Only if memory matters before Square |
