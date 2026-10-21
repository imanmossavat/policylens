"""Keep docs from going stale: unique IDs, no dangling references, index complete (D-006)."""

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"\b([DGFQ]-\d{3})\b")
DEFINITION = re.compile(r"^([DGFQ]-\d{3})\b", re.M)


def _docs():
    return sorted((ROOT / "docs").glob("*.md"))


def _defined():
    return [m for p in _docs() for m in DEFINITION.findall(p.read_text())]


def test_ids_are_unique():
    dupes = [i for i, n in Counter(_defined()).items() if n > 1]
    assert not dupes, f"Duplicate IDs: {dupes}"


def test_every_cited_id_is_defined():
    defined = set(_defined())
    files = [ROOT / "AGENTS.md", ROOT / "README.md", *_docs(),
             *(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py"),
             *(p for p in (ROOT / "tests").rglob("*.py") if p.name != "test_docs.py")]
    missing = {(i, p.relative_to(ROOT).as_posix()) for p in files if p.exists()
               for i in ID.findall(p.read_text()) if i not in defined}
    assert not missing, f"Cited but not defined: {sorted(missing)}"


def test_every_doc_is_in_the_index():
    index = (ROOT / "AGENTS.md").read_text()
    unlisted = [p.relative_to(ROOT).as_posix() for p in _docs()
                if p.relative_to(ROOT).as_posix() not in index]
    assert not unlisted, f"Add to AGENTS.md: {unlisted}"


def test_doc_entries_stay_short():
    long = [(p.name, line[:40]) for p in _docs() for line in p.read_text().splitlines()
            if DEFINITION.match(line) and len(line) > 300]
    assert not long, f"Entries over 300 chars; move detail to a docstring or notebook: {long}"
