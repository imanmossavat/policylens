# Report index: every published report, its source and how to reproduce it
All claude.ai links are private to the owner until shared. Sources under exports/ are gitignored; reproduction records are in docs/.

| Report | Link | Source | Reproduce / record |
|---|---|---|---|
| Nut on a Peg: Square, borrowed DPPO policy, timeouts, dice, force, trained policies (part 2) | https://claude.ai/artifact/KzMdYvkqudvXAziN1SRiBo | exports/square_demo/ (build.py) | docs/report_square.md |
| Policy Lens: field guide for non-roboticists, PushT + ALOHA (part 1) | https://claude.ai/artifact/GUypTRrg3MMFE7gfVBkB8t | exports/presentation/ (index.html, make_clips.py) | clips are fresh PushT episodes, not run folders (see make_clips.py); numbers cite F-003, F-013, F-014, F-017, F-018 |
| Why PushT rollouts were slow (plain language, Q-009) | https://claude.ai/artifact/9BYR9vs69p9nkcs71eozGG | not in repo | F-004, F-006 |
| Fate-curve figures | exports/20261024-114140_fate_ddim10_16parents_fate.png, exports/20261024-154946_fate_ddim10_rest7_fate.png | scripts/fate_plot.py runs/<fate run> | F-017 |

The user's own reading index of private claude.ai docs: docs/readable.md (local, gitignored).
New report: add a row here; if it has numbers, give it a docs/report_<name>.md record like docs/report_square.md.
