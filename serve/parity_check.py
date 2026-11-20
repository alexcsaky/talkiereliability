"""Parity: does the transformers port give the official model's log-probabilities?

Runs in one GPU container. Both 13B models do not fit a 48 GB card together,
so the official model runs first, its outputs are kept on the CPU, it is
freed, and then the port runs on identical token ids.

Compared per prompt:

* per-token log-probabilities of the actual next tokens, every position;
* full-vocabulary logits at a few sampled positions (max abs difference, KL);
* top-1 agreement and top-10 overlap at every position.

The official forward returns last-position logits only; ``OfficialAdapter``
reproduces it at every position, and the job first checks that the adapter's
last position equals the official forward on each prompt.

Pass criterion (build brief, stage 0): within the port's stated tolerance, a
maximum absolute logit difference of about 0.03 (the port's model card
reports 0.03125). Reported as observed; the decision is Alex's.
"""

from __future__ import annotations

import gc
import random
import time

import torch

from serve.scoring import Engine, OfficialAdapter, PortAdapter

N_SAMPLED_POSITIONS = 4


def _run(engine: Engine, prompts_ids: list[list[int]], sampled: list[list[int]]) -> list[dict]:
    out = []
    for ids, pos in zip(prompts_ids, sampled):
        logits = engine.logits_at(ids)  # [T, V] float32 on CPU
        lp = torch.log_softmax(logits, -1)
        tok_lp = lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
        out.append({"tok_lp": tok_lp, "top10": logits.topk(10, dim=-1).indices,
                    "sampled": logits[pos].clone(), "sampled_lp": lp[pos].clone()})
        del logits, lp
    return out


def run_parity(load_official, load_port, prompts: list[str], seed: int = 0) -> dict:
    """``load_*`` return (model, enc). ``prompts`` must already be admitted."""
    t0 = time.time()
    off_model, enc = load_official()
    off = Engine(OfficialAdapter(off_model), enc)
    prompts_ids = [[65535] + off.encode(p) for p in prompts]
    noeot_ids = [off.encode(p) for p in prompts[:8] if len(off.encode(p)) >= 2]
    rng = random.Random(seed)
    sampled = [sorted(rng.sample(range(len(ids)), min(N_SAMPLED_POSITIONS, len(ids))))
               for ids in prompts_ids]
    noeot_sampled = [sorted(rng.sample(range(len(ids)), min(N_SAMPLED_POSITIONS, len(ids))))
                     for ids in noeot_ids]

    # 1. The adapter reproduces the official forward at the last position.
    adapter_err = 0.0
    with torch.no_grad():
        for ids in prompts_ids[:8]:
            t = torch.tensor([ids], device=off.adapter.device)
            ref = off.adapter.last_logits_reference(t)[0].float().cpu()
            mine = off.logits_at(ids, [len(ids) - 1])[0]
            adapter_err = max(adapter_err, (ref - mine).abs().max().item())
    a = _run(off, prompts_ids, sampled)
    a_noeot = _run(off, noeot_ids, noeot_sampled)
    t_off = time.time() - t0
    del off, off_model
    gc.collect()
    torch.cuda.empty_cache()

    t1 = time.time()
    port_model, enc2 = load_port()
    port = Engine(PortAdapter(port_model), enc2)
    assert [[65535] + port.encode(p) for p in prompts] == prompts_ids, "tokenizers differ"
    b = _run(port, prompts_ids, sampled)
    b_noeot = _run(port, noeot_ids, noeot_sampled)

    # 2. Check batched right-padded scoring invariance on the real port model.
    batch_items = [(ids, max(1, len(ids) - 16)) for ids in prompts_ids[: min(16, len(prompts_ids))]]
    batched_lps = port.score_ids(batch_items)
    port_single = Engine(port.adapter, enc2, max_batch_tokens=1)
    single_lps = port_single.score_ids(batch_items)
    batch_max_diff = max(
        (abs(x - y) for xs, ys in zip(batched_lps, single_lps) for x, y in zip(xs, ys)),
        default=0.0,
    )
    t_port = time.time() - t1

    per_prompt = []
    for i, (x, y) in enumerate(zip(a, b)):
        d_lp = (x["tok_lp"] - y["tok_lp"]).abs()
        d_logit = (x["sampled"] - y["sampled"]).abs().max().item()
        kl = (x["sampled_lp"].exp() * (x["sampled_lp"] - y["sampled_lp"])).sum(-1).max().item()
        top1 = (x["top10"][:, 0] == y["top10"][:, 0]).float().mean().item()
        overlap = sum(len(set(p.tolist()) & set(q.tolist()))
                      for p, q in zip(x["top10"], y["top10"])) / (10 * len(x["top10"]))
        per_prompt.append({
            "prompt": i, "n_tokens": len(prompts_ids[i]),
            "max_abs_logit_diff_sampled": d_logit,
            "max_abs_tok_logprob_diff": d_lp.max().item(),
            "mean_abs_tok_logprob_diff": d_lp.mean().item(),
            "sum_logprob_official": x["tok_lp"].sum().item(),
            "sum_logprob_port": y["tok_lp"].sum().item(),
            "max_kl_sampled": kl, "top1_agree": top1, "top10_overlap": overlap,
        })
    noeot_max_logit = max(
        ((x["sampled"] - y["sampled"]).abs().max().item() for x, y in zip(a_noeot, b_noeot)),
        default=0.0,
    )
    worst = lambda k: max(p[k] for p in per_prompt)  # noqa: E731
    return {
        "n_prompts": len(prompts),
        "adapter_vs_official_forward_max_abs": adapter_err,
        "max_abs_logit_diff": worst("max_abs_logit_diff_sampled"),
        "max_abs_logit_diff_noeot": noeot_max_logit,
        "batch_right_pad_max_abs_tok_logprob_diff": batch_max_diff,
        "max_abs_tok_logprob_diff": worst("max_abs_tok_logprob_diff"),
        "max_abs_seq_logprob_diff": max(abs(p["sum_logprob_official"] - p["sum_logprob_port"])
                                        for p in per_prompt),
        "max_kl": worst("max_kl_sampled"),
        "min_top1_agree": min(p["top1_agree"] for p in per_prompt),
        "mean_top10_overlap": sum(p["top10_overlap"] for p in per_prompt) / len(per_prompt),
        "seconds_official": round(t_off, 1), "seconds_port": round(t_port, 1),
        "per_prompt": per_prompt,
    }
