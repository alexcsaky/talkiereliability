"""How much context can Talkie use? (Unknown 1 in the build brief.)

Two measurements, at total lengths of 1k, 2k, 3k and 4k tokens:

1. **Loss on a fixed span against context length.** For each long period
   text, take a 256-token target span and score it after the preceding
   768, 1,792, 2,816 or 3,840 tokens. If longer context is usable, the loss
   on the same span should not rise as context grows. A rise past about 2k
   would confirm the porter's report.
2. **Retrieval.** Invented facts (a county fair prize and its winner, plus a
   distractor fair) are planted at 10, 50 or 90 per cent depth in period
   filler text, and a question at the end asks for the winner. Accuracy is
   the share of trials where the true winner's name scores highest among four
   names. Chance is 25 per cent. The names are invented, so the model cannot
   know them.

Caveat: well-known books are probably in Talkie's training data, so absolute
losses are low; the comparison across context lengths is what matters.
"""

from __future__ import annotations

import math
import random
import statistics
import time

from serve.scoring import Engine
from serve.tokenizer import EOT_ID

TOTALS = (1024, 2048, 3072, 4096)
SPAN = 256
DEPTHS = (0.1, 0.5, 0.9)

_FIRST = ["Elias", "Josiah", "Ambrose", "Thaddeus", "Hiram", "Silas", "Obadiah", "Ezra",
          "Lemuel", "Cyrus", "Jethro", "Amos", "Barnabas", "Gideon", "Enoch", "Asa"]
_LAST = ["Thornbury", "Pettigrew", "Hollister", "Winslade", "Ashcombe", "Marbury",
         "Fenwick", "Gladstone", "Pruett", "Ockley", "Brantwood", "Kittredge",
         "Haverill", "Stockdale", "Wetherby", "Calloway", "Dunmore", "Ravenhill"]
_TOWNS = ["Ashford", "Millbrook", "Cedar Falls", "Bramwell", "Oakdale", "Hartwell",
          "Pine Ridge", "Elmhurst", "Fairhaven", "Stonebridge", "Willow Creek", "Granby"]
_CROPS = ["pumpkin", "ear of corn", "cabbage", "squash", "turnip", "watermelon"]


def _safe_cut(enc, tokens: list[int], pos: int) -> int:
    """Shift ``pos`` minimally so decoding ``tokens[:pos]`` does not split a UTF-8 character."""
    n = len(tokens)
    pos = max(0, min(pos, n))
    for delta in (0, -1, 1, -2, 2, -3, 3):
        p = pos + delta
        if 0 <= p <= n:
            left = enc.decode(tokens[:p])
            right = enc.decode(tokens[p:])
            if "\ufffd" not in left[-2:] and "\ufffd" not in right[:2]:
                return p
    return pos


def loss_vs_context(engine: Engine, texts: list[str]) -> dict:
    """Mean loss (nats/token) on the same span after each context length."""
    head = [EOT_ID] if engine.prepend_eot else []
    rows = []
    for ti, text in enumerate(texts):
        ids = engine.encode(text)
        need = TOTALS[-1] - len(head)
        if len(ids) < need:
            continue
        window = ids[:need]
        span_start = need - SPAN
        for total in TOTALS:
            ctx = total - len(head) - SPAN
            seq = head + window[span_start - ctx: span_start + SPAN]
            lp = engine.score_ids([(seq, len(seq) - SPAN)])[0]
            rows.append({"text": ti, "total_tokens": total, "context_tokens": ctx,
                         "loss": -sum(lp) / len(lp)})
    summary = {}
    for total in TOTALS:
        losses = [r["loss"] for r in rows if r["total_tokens"] == total]
        if losses:
            summary[total] = {"mean_loss": statistics.mean(losses),
                              "ppl": math.exp(statistics.mean(losses)), "n_texts": len(losses)}
    return {"by_total": summary, "rows": rows}


def loss_by_position(engine: Engine, texts: list[str], bins: int = 8) -> dict:
    """Mean loss per 512-token bin of a 4,096-token window (for the record)."""
    head = [EOT_ID] if engine.prepend_eot else []
    width = TOTALS[-1] // bins
    acc = [[] for _ in range(bins)]
    need = TOTALS[-1] - len(head)
    for text in texts:
        ids = engine.encode(text)[:need]
        if len(ids) < need:
            continue
        lp = engine.score_ids([(head + ids, 1)])[0]
        for i, x in enumerate(lp):
            acc[min((i + 1) // width, bins - 1)].append(-x)
    return {f"{b * width}-{(b + 1) * width}": (statistics.mean(v) if v else None)
            for b, v in enumerate(acc)}


def _fact(rng: random.Random, town: str):
    winner = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    crop = rng.choice(_CROPS)
    sentence = (f" At the {town} county fair this autumn the first prize for the finest "
                f"{crop} was awarded to Mr. {winner}, who received a silver cup.")
    return winner, sentence


def retrieval(engine: Engine, filler_texts: list[str], trials_per_cell: int = 10,
              seed: int = 0, gate=None) -> dict:
    """``gate`` is applied to every composed prompt and option (pass firewall admission)."""
    rng = random.Random(seed)
    head_len = 1 if engine.prepend_eot else 0
    filler_ids = [engine.encode(t) for t in filler_texts]
    filler_ids = [f for f in filler_ids if len(f) > TOTALS[-1]]
    if not filler_ids:
        raise ValueError("need filler texts longer than 4,096 tokens")
    rows = []
    reqs = []
    for total in TOTALS:
        for depth in DEPTHS:
            for k in range(trials_per_cell):
                town, other_town = rng.sample(_TOWNS, 2)
                winner, needle = _fact(rng, town)
                decoy_winner, distractor = _fact(rng, other_town)
                while decoy_winner == winner:
                    decoy_winner, distractor = _fact(rng, other_town)
                names = {winner, decoy_winner}
                while len(names) < 4:
                    names.add(f"{rng.choice(_FIRST)} {rng.choice(_LAST)}")
                options = sorted(names)
                question = (f"\n\nQuestion. To whom was the first prize awarded at the "
                            f"{town} county fair?\nAnswer. It was awarded to Mr.")
                n_need = len(engine.encode(needle)) + len(engine.encode(distractor))
                n_q = len(engine.encode(question))
                budget = total - head_len - n_need - n_q - 24  # room for answer tokens and join drift
                src = rng.choice(filler_ids)
                start = rng.randrange(0, len(src) - budget)
                fill = src[start: start + budget]
                cut = _safe_cut(engine.enc, fill, int(len(fill) * depth))
                cut2 = _safe_cut(engine.enc, fill, int(len(fill) * rng.uniform(0.05, 0.95)))
                # Rebuild text so encoding stays clean at the joins.
                parts = [(cut, needle), (cut2, distractor)]
                parts.sort()
                text, prev = "", 0
                for at, sent in parts:
                    text += engine.enc.decode(fill[prev:at]) + sent
                    prev = at
                text += engine.enc.decode(fill[prev:]) + question
                conts = [f" {o}." for o in options]
                if gate is not None:
                    text = gate(text)
                    conts = [gate(c) for c in conts]
                reqs.append((text, conts))
                rows.append({"total_tokens": total, "depth": depth, "trial": k,
                             "options": options, "answer": options.index(winner),
                             "decoy_in_context": options.index(decoy_winner)})
    t0 = time.time()
    scores = engine.score_requests(reqs)
    elapsed = time.time() - t0
    for row, sc, (prefix, _) in zip(rows, scores, reqs):
        lps = [s.logprob for s in sc]
        row["pred"] = max(range(4), key=lambda i: lps[i])
        row["correct"] = row["pred"] == row["answer"]
        row["chose_distractor"] = row["pred"] == row["decoy_in_context"]
        row["n_tokens"] = head_len + len(engine.encode(prefix)) + max(s.n_tokens for s in sc)
    cells = {}
    for total in TOTALS:
        sub = [r for r in rows if r["total_tokens"] == total]
        cells[total] = {"accuracy": sum(r["correct"] for r in sub) / len(sub),
                        "chose_distractor": sum(r["chose_distractor"] for r in sub) / len(sub),
                        "n": len(sub),
                        "by_depth": {d: sum(r["correct"] for r in sub if r["depth"] == d)
                                     / max(1, sum(1 for r in sub if r["depth"] == d))
                                     for d in DEPTHS}}
    return {"by_total": cells, "chance": 0.25, "seconds": round(elapsed, 1), "rows": rows}
