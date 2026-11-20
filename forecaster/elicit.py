"""Reading a forecast from Talkie's token probabilities (build brief 7.4).

Each binary question is a statement A and its negation (or matched opposite),
three or four paraphrases each. Under each period header (template), every
paraphrase is scored as a continuation; probability is summed over the
paraphrases of each outcome; the log-odds of A against not-A is taken; and
log-odds are averaged over four to six templates. No PMI normalisation.

The same machinery serves the known-truth check (retrospective headers) and,
later, the forecasting arms (prospective headers, passages in the prefix).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

import firewall
from forecaster.client import TalkieClient
from serve.scoring import logodds


@dataclass(frozen=True)
class BinaryQuestion:
    id: str
    family: str
    yes: tuple[str, ...]          # paraphrases of statement A
    no: tuple[str, ...]           # paraphrases of not-A
    fields: dict = field(default_factory=dict)   # values for template placeholders
    cluster: str = ""

    @staticmethod
    def from_dict(d: dict) -> "BinaryQuestion":
        return BinaryQuestion(id=d["id"], family=d["family"], yes=tuple(d["yes"]),
                              no=tuple(d["no"]), fields=dict(d.get("fields", {})),
                              cluster=d.get("cluster", ""))


@dataclass(frozen=True)
class Template:
    id: str
    header: str      # str.format template; ends where the statement begins

    def render(self, **fields) -> str:
        return self.header.format(**fields)


@dataclass
class Request:
    question_id: str
    template_id: str
    prefix: firewall.Admitted
    yes: list[firewall.Admitted]
    no: list[firewall.Admitted]


def build_requests(questions: Sequence[BinaryQuestion], templates: Sequence[Template],
                   *, passage: firewall.Admitted | None = None, header_kind: str = "header"
                   ) -> tuple[list[Request], list[firewall.FirewallRejection]]:
    """Admit every header and continuation through the firewall.

    ``passage``, when given, must already have passed ``firewall.admit(...,
    kind="passage", pub_date=...)``; it is composed with each admitted header
    via ``firewall.compose`` so Rule 1's date check is never bypassed and
    genuine pre-1931 forecast passages mentioning 1931 to 1940 are not
    rejected by ``header``'s stricter future-year ban.

    A question whose strings are rejected is dropped in full (no partial
    template rows are kept) and the rejection returned, so the caller can
    report it.
    """
    if passage is not None:
        if not isinstance(passage, firewall.Admitted):
            raise TypeError(
                "passage must be a firewall.Admitted object (admitted with kind='passage')"
            )
        if passage.kind != "passage":
            raise ValueError(f"passage must have kind='passage', got {passage.kind!r}")
        if passage.leak_probe:
            raise ValueError("passage cannot carry the leak-probe exemption")

    reqs, rejected = [], []
    for q in questions:
        q_reqs: list[Request] = []
        try:
            for t in templates:
                hdr = firewall.admit(t.render(**q.fields).rstrip(" "),
                                     kind=header_kind, origin=f"{q.id}:{t.id}")
                if passage is not None:
                    join_sep = "" if passage.text.endswith("\n") else "\n\n"
                    prefix = firewall.compose(
                        [passage, hdr], kind=header_kind,
                        origin=f"{q.id}:{t.id}:{passage.origin}", sep=join_sep,
                    )
                else:
                    prefix = hdr
                # After a line break the statement starts the line; otherwise
                # it follows a space. Either way prefix + continuation encodes
                # the same as the joined text (boundary_ok).
                sep = "" if prefix.text.endswith("\n") else " "
                conts = [[firewall.admit(sep + s.strip(), kind="continuation",
                                         origin=f"{q.id}:{side}{i}:{t.id}")
                          for i, s in enumerate(stmts)]
                         for side, stmts in (("yes", q.yes), ("no", q.no))]
                q_reqs.append(Request(q.id, t.id, prefix, conts[0], conts[1]))
        except firewall.FirewallRejection as e:
            rejected.append(e)
        else:
            reqs.extend(q_reqs)
    return reqs, rejected


def run(client: TalkieClient, requests: Sequence[Request]) -> list[dict]:
    """One row per (question, template): log-odds and the raw scores."""
    scores = client.score([(r.prefix, r.yes + r.no) for r in requests])
    rows = []
    for r, sc in zip(requests, scores):
        lp = [s["logprob"] for s in sc]
        ny = len(r.yes)
        rows.append({
            "question_id": r.question_id, "template_id": r.template_id,
            "logodds": logodds(lp[:ny], lp[ny:]),
            "yes_logprobs": lp[:ny], "no_logprobs": lp[ny:],
            "n_tokens": [s["n_tokens"] for s in sc],
            "boundary_ok": all(s["boundary_ok"] for s in sc),
        })
    return rows


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def aggregate(rows: Sequence[dict]) -> dict[str, dict]:
    """Average log-odds over templates for each question."""
    by_q: dict[str, list[float]] = {}
    for r in rows:
        by_q.setdefault(r["question_id"], []).append(r["logodds"])
    out = {}
    for qid, lo in by_q.items():
        m = statistics.mean(lo)
        out[qid] = {"logodds_raw": m, "p_raw": _sigmoid(m),
                    "n_templates": len(lo),
                    "template_sd": statistics.stdev(lo) if len(lo) > 1 else 0.0}
    return out
