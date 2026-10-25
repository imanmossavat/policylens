# policy-lens: index

Goal: understand why robot policies succeed or fail on assembly and disassembly.
PushT is only the training ground for the toolkit (D-001).

## Where things are
| Need | Look in | Grep |
|---|---|---|
| Setup and commands | README.md | |
| Code map | src/plens/__init__.py docstring | |
| Run folder layout | src/plens/store.py | `LAYOUT` |
| Tasks and what each records | src/plens/envs.py | `TASKS` |
| Why something is the way it is | docs/decisions.md | `D-0` |
| Traps already hit | docs/gotchas.md, plus code sites | `G-0`, `GOTCHA(` |
| Results so far | docs/findings.md | `F-0` |
| Open research questions | docs/questions.md | `Q-0` |
| Tasks and the assembly difficulty each isolates | docs/tasks.md | `diff:A` ... `diff:F` |
| Human-readable story docs (user's own, don't read unless asked) | docs/readable.md | |

Searching an ID (e.g. `G-001`) finds its doc line and every code line that cites it.

## Conventions
- Code comments: `WHY:` for intent the code can't show; `GOTCHA(G-###):` at a trap's site.
- Functions live in src/plens/ with tests in tests/. Scripts in scripts/ only parse args and call the library.
  Notebooks (later, notebooks/) are for teaching and exploring; they import, never define, library code.
- Every run writes runs/<run_id>/meta.json: code commit, versions, device, config, seeds, outcome.
  Run IDs are UTC; a run from uncommitted code also saves git_diff.patch.
- `pytest -m "not smoke"` is fast; plain `pytest` also runs the real env and policy architecture.
- Before building or training on a new task: check existing benchmarks for it, and run a trivial controller.
  If the trivial controller solves it, tell the user; never redesign an env without their permission.

## When and what to document
Only what the code can't tell you. One line per entry; if it needs more, it belongs in a docstring or notebook.
- Decision (D): chose between real alternatives. Append only. To change one, add a line "supersedes D-00x".
- Gotcha (G): a trap that cost real time. One line in gotchas.md and a `GOTCHA(G-###)` comment at the site.
- Finding (F): a run answered something. Cite the run_id; runs never change, so findings can't rot.
- Question (Q): something worth testing. Update its status in place: open / active / answered by F-###.
- Never document: arguments (`--help`), APIs (docstrings), file lists (`ls`), anything restated from code.
tests/test_docs.py fails if an ID is duplicated, an ID is cited but undefined, or a docs/ file is missing above.
