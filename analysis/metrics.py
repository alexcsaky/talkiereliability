"""Scoring rules and discrimination. Only analysis/ reads outcomes."""

from __future__ import annotations

import math
import random
from typing import Sequence


def brier(p: Sequence[float], y: Sequence[int]) -> float:
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)


def log_score(p: Sequence[float], y: Sequence[int], eps: float = 1e-6) -> float:
    """Mean log-likelihood (higher is better)."""
    return sum(math.log(min(max(pi if yi else 1 - pi, eps), 1)) for pi, yi in zip(p, y)) / len(p)


def auroc(score: Sequence[float], y: Sequence[int]) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    pos = [s for s, t in zip(score, y) if t]
    neg = [s for s, t in zip(score, y) if not t]
    if not pos or not neg:
        return float("nan")
    ranked = sorted((s, i) for i, s in enumerate(score))
    ranks = [0.0] * len(score)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = sum(r for r, t in zip(ranks, y) if t)
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cluster_bootstrap(stat, items: Sequence[dict], cluster_key: str, n: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile interval, resampling whole clusters."""
    clusters: dict[str, list[dict]] = {}
    for it in items:
        clusters.setdefault(it[cluster_key], []).append(it)
    keys = sorted(clusters)
    rng = random.Random(seed)
    est = stat(list(items))
    draws = []
    for _ in range(n):
        sample = [it for k in (rng.choice(keys) for _ in keys) for it in clusters[k]]
        v = stat(sample)
        if not math.isnan(v):
            draws.append(v)
    draws.sort()
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return est, lo, hi


def softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]
