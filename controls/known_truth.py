"""Known-truth check: can the reading method tell true from false? (stage 0)

Closed book, 1900 to 1929 outcomes (``questions/known_truth.jsonl``, built by
``controls/build_known_truth.py``). This module only produces log-odds; it
never sees outcomes. ``analysis/stage0_report.py`` scores them (AUROC) against
the held-apart outcomes file.

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


def load():
    qs = [elicit.BinaryQuestion.from_dict(json.loads(line))
          for line in QUESTIONS.read_text().splitlines() if line.strip()]
    ts = [elicit.Template(t["id"], t["header"])
          for t in yaml.safe_load(TEMPLATES.read_text())["templates"]]
    return qs, ts


def run(client: TalkieClient, limit: int | None = None) -> dict:
    qs, ts = load()
    if limit:
        qs = qs[:limit]
    reqs, rejected = elicit.build_requests(qs, ts)
    rows = elicit.run(client, reqs)
    agg = elicit.aggregate(rows)
    fam = {q.id: q.family for q in qs}
    return {
        "model": client.model,
        "n_questions": len(agg),
        "rejected_by_firewall": [{"origin": e.origin, "reasons": e.reasons} for e in rejected],
        "per_template": rows,
        "forecasts": [{"question_id": k, "family": fam[k], **v} for k, v in agg.items()],
    }
