"""Rule 8: outcomes are read only by analysis/."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = ["forecaster", "librarian", "controls", "serve", "ledger", "archive"]
ALLOWED = {"controls/build_known_truth.py"}  # writes outcomes once, from public data


def test_only_analysis_reads_outcomes():
    offenders = []
    for d in FORBIDDEN_DIRS:
        for path in (REPO / d).rglob("*.py"):
            rel = str(path.relative_to(REPO))
            if rel in ALLOWED:
                continue
            if re.search(r"""["'/]outcomes\b|\boutcomes/""", path.read_text()):
                offenders.append(rel)
    assert offenders == []


def test_outcomes_is_not_a_package():
    assert not (REPO / "outcomes" / "__init__.py").exists()
