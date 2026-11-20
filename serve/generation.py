"""Text generation for the librarian mode and the demo (port model only).

Uses the port's KV cache through transformers ``generate``. ``temperature=0``
means greedy decoding; the official code divides by the temperature and so
fails at zero (verified, talkie/model.py at 35317ba), which is why the brief
says to use ``top_k=1`` there.
"""

from __future__ import annotations

import torch

from serve.scoring import check_text
from serve.tokenizer import EOT_ID, chat_ids, encode_text


@torch.no_grad()
def generate(model, enc, *, prompt: str | None = None, messages: list[dict] | None = None,
             style: str = "base", max_tokens: int = 256, temperature: float = 0.7,
             top_k: int | None = None, top_p: float | None = None, seed: int = 0,
             max_seq_len: int = 4096) -> dict:
    if (prompt is None) == (messages is None):
        raise ValueError("give exactly one of prompt or messages")
    if messages is not None:
        if style != "it":
            raise ValueError("messages need the chat model")
        for m in messages:
            check_text(m["content"])
        ids = chat_ids(enc, messages)
    else:
        check_text(prompt)
        ids = [EOT_ID] + encode_text(enc, prompt)
    if len(ids) + max_tokens > max_seq_len:
        raise ValueError(f"prompt {len(ids)} + max_tokens {max_tokens} > {max_seq_len}")

    stop = [EOT_ID]
    if style == "it":
        stop.append(enc.encode_single_token("<|end|>"))
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    greedy = temperature == 0 or top_k == 1
    kwargs = dict(max_new_tokens=max_tokens, eos_token_id=stop, pad_token_id=EOT_ID,
                  do_sample=not greedy)
    if not greedy:
        kwargs.update(temperature=temperature, top_k=top_k or 0, top_p=top_p or 1.0)
    out = model.generate(torch.tensor([ids], device=device), **kwargs)[0, len(ids):].tolist()
    finish = "length"
    if out and out[-1] in stop:
        out, finish = out[:-1], "stop"
    return {"text": enc.decode(out), "token_ids": out, "n_prompt_tokens": len(ids),
            "finish_reason": finish}
