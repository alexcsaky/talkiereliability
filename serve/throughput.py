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
    orig_bt = engine.max_batch_tokens
    on_cuda = engine.adapter.device.type == "cuda"
    head = [EOT_ID] if engine.prepend_eot else []
    rows = []
    try:
        for L in seq_lens:
            need = L - len(head)
            if len(text_ids) <= need:
                raise ValueError(f"need more than {need} tokens for seq_len={L}, got {len(text_ids)}")
            items = []
            for _ in range(n_seqs):
                s = rng.randrange(0, len(text_ids) - need)
                seq = head + text_ids[s: s + need]
                items.append((seq, len(seq) - 16))  # score the last 16 tokens, like a continuation
            for bt in batch_tokens:
                engine.max_batch_tokens = bt
                try:
                    engine.score_ids(items[:4])  # warm up
                    if on_cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    engine.score_ids(items)
                    if on_cuda:
                        torch.cuda.synchronize()
                    dt = time.time() - t0
                    peak = round(torch.cuda.max_memory_allocated() / 2**30, 1) if on_cuda else 0.0
                    rows.append({"seq_len": L, "max_batch_tokens": bt, "seqs": n_seqs,
                                 "tokens_per_sec": round(n_seqs * L / max(dt, 1e-9)),
                                 "seconds": round(dt, 2), "peak_mem_gib": peak})
                except torch.cuda.OutOfMemoryError:
                    if on_cuda:
                        torch.cuda.empty_cache()
                    rows.append({"seq_len": L, "max_batch_tokens": bt, "oom": True})
    finally:
        engine.max_batch_tokens = orig_bt
    return rows


def chars_per_token(engine: Engine, texts: list[str]) -> float:
    """Tokens per 1,000 characters (unknown 5), on whatever texts are given."""
    n_chars = sum(len(t) for t in texts)
    if n_chars == 0:
        raise ValueError("empty texts")
    n_tok = sum(len(engine.encode(t)) for t in texts)
    return 1000 * n_tok / n_chars
