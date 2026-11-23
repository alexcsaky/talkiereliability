"""How much do scores depend on batching? (stage 0, decided 23 Nov 2026)

Stage 0 found that right-padded batched scoring and one-at-a-time scoring
differ by up to 0.12 nats on a single token (about one bf16 step). Rule 6 needs
identical evidence across arms, and arms batch different prompts together, so
this measures the typical difference, not only the worst case, and what
one-at-a-time scoring would cost. Three comparisons on the same sequences:

* ``single``: production batching against one sequence per batch;
* ``neighbours``: production batching of all sequences against production
  batching of a random half alone (same prompt, different batch neighbours,
  as happens between arms);
* ``batch_8192`` and ``batch_32768``: other batch-token budgets.

Each reports the absolute difference in a continuation's summed log-prob and in
each request's log-odds (statement against negation, the quantity analysed),
plus the share of sequences scored bit-identically. Strings arrive already
admitted by the firewall (built from ``controls/known_truth`` requests).
"""

from __future__ import annotations

import random
import statistics
import time
from typing import Sequence

from serve.scoring import Engine, logodds


def _stats(xs: list[float]) -> dict:
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    return {"mean": statistics.fmean(s), "median": q(0.5), "p95": q(0.95), "p99": q(0.99),
            "max": s[-1], "share_zero": sum(x == 0.0 for x in s) / len(s)}


def _compare(a: list[list[float]], b: list[list[float]], groups, idx: Sequence[int]) -> dict:
    """Compare token log-probs ``a`` and ``b`` (aligned to ``idx``) of the same sequences."""
    pos = {i: k for k, i in enumerate(idx)}
    seq = [abs(sum(a[k]) - sum(b[k])) for k in range(len(idx))]
    tok = max((abs(x - y) for k in range(len(idx)) for x, y in zip(a[k], b[k])), default=0.0)
    lo = []
    for members, n_yes in groups:
        if not all(m in pos for m in members):
            continue
        la = [sum(a[pos[m]]) for m in members]
        lb = [sum(b[pos[m]]) for m in members]
        lo.append(abs(logodds(la[:n_yes], la[n_yes:]) - logodds(lb[:n_yes], lb[n_yes:])))
    return {"n_sequences": len(idx), "n_requests": len(lo), "max_token_diff": tok,
            "seq_logprob_absdiff": _stats(seq), "logodds_absdiff": _stats(lo)}


def _score(engine: Engine, items, max_batch_tokens: int) -> tuple[list[list[float]], float]:
    old = engine.max_batch_tokens
    engine.max_batch_tokens = max_batch_tokens
    try:
        t0 = time.time()
        out = engine.score_ids(items)
        return out, time.time() - t0
    finally:
        engine.max_batch_tokens = old


def measure(engine: Engine, requests: Sequence[tuple[str, Sequence[str]]],
            n_yes: Sequence[int], seed: int = 0, extra_budgets=(8192, 32768)) -> dict:
    items, groups = [], []
    for (prefix, conts), ny in zip(requests, n_yes):
        start_i = len(items)
        for c in conts:
            ids, start, _ = engine.build(prefix, c)
            items.append((ids, start))
        groups.append((list(range(start_i, len(items))), ny))
    every = list(range(len(items)))
    prod = engine.max_batch_tokens

    base, t_base = _score(engine, items, prod)
    single, t_single = _score(engine, items, 1)
    rng = random.Random(seed)
    # Halve by request so every kept request has all its continuations.
    kept = rng.sample(range(len(groups)), len(groups) // 2)
    half = sorted(i for g in kept for i in groups[g][0])
    half_scores, _ = _score(engine, [items[i] for i in half], prod)

    out = {
        "production_max_batch_tokens": prod,
        "n_sequences": len(items), "n_requests": len(groups),
        "mean_tokens_per_sequence": statistics.fmean(len(s) for s, _ in items),
        "seconds_batched": round(t_base, 2), "seconds_single": round(t_single, 2),
        "single_slowdown": round(t_single / max(t_base, 1e-9), 1),
        "single": _compare(base, single, groups, every),
        "neighbours": _compare([base[i] for i in half], half_scores, groups, half),
    }
    for b in extra_budgets:
        other, t = _score(engine, items, b)
        out[f"batch_{b}"] = {**_compare(base, other, groups, every), "seconds": round(t, 2)}
    return out
