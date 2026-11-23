"""Known-truth check: can the reading method tell true from false? (stage 0)

Closed book, 1900 to 1929 outcomes (``questions/known_truth.jsonl``, built by
``controls/build_known_truth.py``; version 2, rebuilt 23 Nov 2026 with coarser
questions). This module only produces log-odds; it never sees outcomes.
``analysis/stage0_report.py`` scores them (AUROC) against the held-apart
outcomes file.

Each question names its header set (``template_set``: ``dated`` by default,
``retrospective`` for the events family, whose asserted fact is a year).

Gate: no fixed threshold. The AUROC and its interval are reported and Alex
decides (Alex, 20 Nov 2026; the brief had proposed at least 0.65).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from forecaster import elicit
from forecaster.client import TalkieClient

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = REPO / "questions" / "known_truth.jsonl"
TEMPLATES = REPO / "questions" / "templates" / "known_truth.yaml"
QUESTION_SET = "v2.1"
_SET_KEYS = {"dated": "templates", "retrospective": "retrospective_templates"}


def _raw() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def load_sets():
    """Questions, header sets by name, and each question's set name."""
    raw = _raw()
    y = yaml.safe_load(TEMPLATES.read_text())
    sets = {name: [elicit.Template(t["id"], t["header"]) for t in y[key]]
            for name, key in _SET_KEYS.items()}
    qs = [elicit.BinaryQuestion.from_dict(d) for d in raw]
    which = {d["id"]: d.get("template_set", "dated") for d in raw}
    return qs, sets, which


def load():
    """Questions and the dated header set (kept for parity prompts and old callers)."""
    qs, sets, _ = load_sets()
    return qs, sets["dated"]


def build_all(qs, sets, which):
    reqs, rejected = [], []
    for name, ts in sets.items():
        sub = [q for q in qs if which[q.id] == name]
        if sub:
            r, rej = elicit.build_requests(sub, ts)
            reqs += r
            rejected += rej
    return reqs, rejected


def run(client: TalkieClient, limit: int | None = None) -> dict:
    qs, sets, which = load_sets()
    if limit:
        qs = qs[:limit]
    reqs, rejected = build_all(qs, sets, which)
    rows = elicit.run(client, reqs)
    agg = elicit.aggregate(rows)
    fam = {q.id: q.family for q in qs}
    return {
        "model": client.model,
        "question_set": QUESTION_SET,
        "n_questions": len(agg),
        "rejected_by_firewall": [{"origin": e.origin, "reasons": e.reasons} for e in rejected],
        "per_template": rows,
        "forecasts": [{"question_id": k, "family": fam[k], **v} for k, v in agg.items()],
    }
