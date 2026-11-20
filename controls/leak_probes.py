"""Leak probes (stage 0): what does Talkie know about the 1930s with no documents?

Three kinds:

1. Choice probes (``controls/leak_probes.yaml``): a prefix and options, one of
   which happened after 1930; plus controls about events before 1931.
2. Year probes: the probability Talkie gives each year from 1920 to 1950 as
   the present year, under several period headers.
3. Closed-book question probes: every target question asked with no passage
   (``probe_questions``). The pilot questions do not exist until stage 2, so
   this runs then; the function is here so stage 2 reuses it.

Optionally (4) a sample of the community probe set
``trumancai/talkie-1930-knowledge-bench`` (CC BY-SA 4.0, pinned revision),
multiple-choice items anchored in the 1930s, with a 1920s retention bucket as
the reference. Its stems were written by a modern pipeline, so every one goes
through the firewall and rejections are counted.

This module records scores only; outcomes are joined in analysis/.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

import yaml

import firewall
from forecaster import elicit
from forecaster.client import TalkieClient

REPO = Path(__file__).resolve().parents[1]
PROBES = REPO / "controls" / "leak_probes.yaml"
KB_REPO = "trumancai/talkie-1930-knowledge-bench"
KB_REVISION = "83cf6135fc242a5bbbaa47857295355c5ec651a8"
KB_SKIP_PREDICATES = {"genre", "publication_date", "inception", "date_of_birth"}


def _admit_choice(pid: str, prefix: str, options: list[str]):
    pre = firewall.admit(prefix, kind="probe", origin=pid, leak_probe=True)
    opts = [firewall.admit(o if o.startswith((" ", "\n")) else " " + o, kind="continuation",
                           origin=f"{pid}:opt{i}", leak_probe=True)
            for i, o in enumerate(options)]
    return pre, opts


def choice_and_year(client: TalkieClient) -> dict:
    spec = yaml.safe_load(PROBES.read_text())
    items, meta, rejected = [], [], []
    for p in spec["choice"]:
        try:
            pre, opts = _admit_choice(p["id"], p["prefix"], p["options"])
        except firewall.FirewallRejection as e:
            rejected.append({"origin": e.origin, "reasons": e.reasons})
            continue
        items.append((pre, opts))
        meta.append({"probe_id": p["id"], "kind": "choice", "options": p["options"],
                     "control": bool(p.get("control"))})
    years = [str(y) for y in range(spec["year"]["options_from"], spec["year"]["options_to"] + 1)]
    for p in spec["year"]["prefixes"]:
        pre, opts = _admit_choice(p["id"], p["prefix"], [" " + y for y in years])
        items.append((pre, opts))
        meta.append({"probe_id": p["id"], "kind": "year", "options": years})
    scores = client.score(items)
    rows = []
    for m, sc in zip(meta, scores):
        rows.append({**m, "logprobs": [s["logprob"] for s in sc],
                     "n_tokens": [s["n_tokens"] for s in sc]})
    return {"model": client.model, "rows": rows, "rejected_by_firewall": rejected}


def probe_questions(client: TalkieClient, questions, templates) -> dict:
    """Closed-book log-odds for target questions (run from stage 2 on)."""
    reqs, rejected = elicit.build_requests(questions, templates)
    rows = elicit.run(client, reqs)
    return {"model": client.model, "forecasts": elicit.aggregate(rows), "per_template": rows,
            "rejected_by_firewall": [{"origin": e.origin, "reasons": e.reasons} for e in rejected]}


def _kb_rows(bucket: str) -> list[dict]:
    url = (f"https://huggingface.co/datasets/{KB_REPO}/resolve/{KB_REVISION}"
           f"/by_bucket/{bucket}.jsonl")
    with urllib.request.urlopen(url, timeout=120) as r:
        return [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]


def knowledge_bench(client: TalkieClient, n: int = 300, seed: int = 0) -> dict:
    """Multiple choice, closed book: novel 1930s items against 1920s retention."""
    rng = random.Random(seed)
    out = {}
    for label, bucket in (("novel_1931s", "probes_eval_novel_1931s"),
                          ("retention_1920s", "probes_eval_retention_1920s")):
        pool = [r for r in _kb_rows(bucket)
                if r["predicate"] not in KB_SKIP_PREDICATES and r.get("probe_mc")]
        rng.shuffle(pool)
        items, meta, rejected = [], [], 0
        for r in pool:
            if len(items) >= n:
                break
            mc = r["probe_mc"]
            try:
                pre, opts = _admit_choice(f"kb:{r['qid']}:{r['predicate']}",
                                          f"Question. {mc['stem']}\nAnswer.", mc["options"])
            except firewall.FirewallRejection:
                rejected += 1
                continue
            items.append((pre, opts))
            meta.append({"qid": r["qid"], "predicate": r["predicate"],
                         "anchor_year": r["anchor_year"], "answer_index": mc["answer_index"],
                         "n_options": len(mc["options"])})
        scores = client.score(items)
        for m, sc in zip(meta, scores):
            lp = [s["logprob"] for s in sc]
            m["pred"] = max(range(len(lp)), key=lambda i: lp[i])
        out[label] = {"rows": meta, "rejected_by_firewall": rejected,
                      "accuracy": sum(m["pred"] == m["answer_index"] for m in meta) / max(1, len(meta)),
                      "chance": sum(1 / m["n_options"] for m in meta) / max(1, len(meta))}
    return out
