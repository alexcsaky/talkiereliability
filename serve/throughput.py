"""Prompt tokens per second for scoring (unknown 3 in the build brief)."""

from __future__ import annotations

import random
import time

import torch

from serve.scoring import Engine
from serve.tokenizer import EOT_ID


def measure(engine: Engine, text_ids: list[int], seq_lens=(256, 512, 1024, 1800),
            batch_tokens=(8192, 16384, 32768), n_seqs: int = 64, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for L in seq_lens:
        items = []
        for _ in range(n_seqs):
            s = rng.randrange(0, len(text_ids) - L)
            seq = [EOT_ID] + text_ids[s: s + L - 1]
            items.append((seq, len(seq) - 16))  # score the last 16 tokens, like a continuation
        for bt in batch_tokens:
            engine.max_batch_tokens = bt
            try:
                engine.score_ids(items[:4])  # warm up
                torch.cuda.synchronize()
                t0 = time.time()
                engine.score_ids(items)
                torch.cuda.synchronize()
                dt = time.time() - t0
                rows.append({"seq_len": L, "max_batch_tokens": bt, "seqs": n_seqs,
                             "tokens_per_sec": round(n_seqs * L / dt), "seconds": round(dt, 2),
                             "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({"seq_len": L, "max_batch_tokens": bt, "oom": True})
            torch.cuda.reset_peak_memory_stats()
    return rows


def chars_per_token(engine: Engine, texts: list[str]) -> float:
    """Tokens per 1,000 characters (unknown 5), on whatever texts are given."""
    n_chars = sum(len(t) for t in texts)
    n_tok = sum(len(engine.encode(t)) for t in texts)
    return 1000 * n_tok / n_chars
