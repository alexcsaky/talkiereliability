"""Log-probability scoring for Talkie, independent of how the model is loaded.

The official Talkie package returns last-position logits only and has no
log-probability call (verified, talkie/model.py at 35317ba). This module
computes per-token log-probabilities of continuations given a prefix, for
either the transformers port or the official model, through a small adapter:

* ``adapter.hidden(ids)`` returns final hidden states ``[B, T, D]``
  (after the last RMS norm), and
* ``adapter.head(h)`` returns float32 logits for any ``[..., D]``.

Conventions (see docs/DECISIONS.md, 20 Nov 2026):

* Text is encoded with special tokens disabled; a control marker in any
  string is an error (strings should have been through ``firewall.admit``).
* By default every sequence starts with ``<|endoftext|>`` (id 65535), the
  document separator, so a prefix reads as the start of a document. The
  official code prepends nothing; ``prepend_eot=False`` reproduces that.
* Prefix and continuation are encoded separately and concatenated, so every
  continuation is scored on the same prefix tokens. ``boundary_ok`` reports
  whether that equals encoding the joined string; continuations should start
  with a space or newline so that it does.
* Batches are right-padded. The port takes positions from ``arange`` and
  ignores the attention mask for that, so left padding would shift positions;
  with causal attention, trailing padding cannot affect real tokens.
* Nothing is truncated: a sequence longer than ``max_seq_len`` raises.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F

from serve.tokenizer import EOT_ID, encode_text

HEAD_CHUNK = 2048  # rows of hidden state per lm-head call, to bound memory


class ControlMarkerError(ValueError):
    pass


class SequenceTooLong(ValueError):
    pass


class Adapter(Protocol):
    device: torch.device

    def hidden(self, ids: torch.Tensor) -> torch.Tensor: ...

    def head(self, h: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class PortAdapter:
    """For ``serve.vendor.talkie_port.TalkieForCausalLM``."""

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        # The causal-LM class subclasses the bare model; call the bare forward
        # to get hidden states without computing logits at every position.
        out = super(type(self.model), self.model).forward(
            input_ids=ids, use_cache=False, return_dict=True)
        return out.last_hidden_state

    def head(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(h).float()
        scale = getattr(self.model.config, "logit_scale", 1.0)
        return logits * scale if scale != 1.0 else logits


class OfficialAdapter:
    """For ``talkie.model.TalkieModel`` (official code, commit 35317ba).

    Mirrors ``TalkieModel.forward`` line for line but keeps every position,
    and runs under bf16 autocast on CUDA as ``talkie.generate`` does.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        m = self.model
        with self._autocast():
            _, seq_len = ids.shape
            cos_sin = m.cos[:, :seq_len], m.sin[:, :seq_len]
            x = m.embed(ids)
            x = F.rms_norm(x, (x.shape[-1],))
            e_x = x
            for block in m.blocks:
                x = block(e_x, x, cos_sin)
            return F.rms_norm(x, (x.shape[-1],))

    def head(self, h: torch.Tensor) -> torch.Tensor:
        m = self.model
        with self._autocast():
            return F.linear(h, m.lm_head_gain(m.lm_head)).float()

    def last_logits_reference(self, ids: torch.Tensor) -> torch.Tensor:
        """The official forward itself, to check this adapter reproduces it."""
        with self._autocast():
            return self.model(ids)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class ContinuationScore:
    logprob: float
    n_tokens: int
    token_logprobs: list[float]
    token_ids: list[int]
    boundary_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def check_text(text: str) -> None:
    if "<|" in text or "|>" in text:
        raise ControlMarkerError(
            "control-marker characters in text bound for the model; "
            "pass strings through firewall.admit first")


class Engine:
    def __init__(self, adapter: Adapter, enc, *, max_seq_len: int = 4096,
                 prepend_eot: bool = True, max_batch_tokens: int = 16384):
        self.adapter = adapter
        self.enc = enc
        self.max_seq_len = max_seq_len
        self.prepend_eot = prepend_eot
        self.max_batch_tokens = max_batch_tokens

    # -- tokenisation ------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        check_text(text)
        return encode_text(self.enc, text)

    def build(self, prefix: str, continuation: str) -> tuple[list[int], int, bool]:
        """Returns (ids, index of first continuation token, boundary_ok)."""
        p = self.encode(prefix)
        c = self.encode(continuation)
        if not c:
            raise ValueError("empty continuation")
        boundary_ok = (p + c) == self.encode(prefix + continuation)
        head = [EOT_ID] if self.prepend_eot else []
        ids = head + p + c
        start = len(head) + len(p)
        if start == 0:
            raise ValueError("empty prefix with prepend_eot=False: first token has no context")
        if len(ids) > self.max_seq_len:
            raise SequenceTooLong(f"{len(ids)} tokens > max_seq_len {self.max_seq_len}")
        return ids, start, boundary_ok

    # -- scoring -----------------------------------------------------------

    def score(self, prefix: str, continuations: Sequence[str]) -> list[ContinuationScore]:
        return self.score_requests([(prefix, list(continuations))])[0]

    def score_requests(self, requests: Sequence[tuple[str, Sequence[str]]]
                       ) -> list[list[ContinuationScore]]:
        items, owners = [], []
        for ri, (prefix, conts) in enumerate(requests):
            for ci, cont in enumerate(conts):
                ids, start, ok = self.build(prefix, cont)
                items.append((ids, start, ok))
                owners.append((ri, ci))
        lps = self.score_ids([(ids, start) for ids, start, _ in items])
        out: list[list[ContinuationScore | None]] = [[None] * len(c) for _, c in requests]
        for (ri, ci), (ids, start, ok), tok_lp in zip(owners, items, lps):
            out[ri][ci] = ContinuationScore(
                logprob=float(sum(tok_lp)), n_tokens=len(tok_lp),
                token_logprobs=[float(x) for x in tok_lp],
                token_ids=ids[start:], boundary_ok=ok)
        return out  # type: ignore[return-value]

    @torch.no_grad()
    def score_ids(self, items: Sequence[tuple[list[int], int]]) -> list[list[float]]:
        """Log-probs of ids[start:] for each (ids, start). Batched by length."""
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        results: list[list[float] | None] = [None] * len(items)
        for batch in self._batches(order, items):
            seqs = [items[i][0] for i in batch]
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), EOT_ID, dtype=torch.long)
            for b, s in enumerate(seqs):
                ids[b, : len(s)] = torch.tensor(s, dtype=torch.long)
            ids = ids.to(self.adapter.device)
            h = self.adapter.hidden(ids)
            b_idx, t_idx, tgt, spans = [], [], [], []
            for b, i in enumerate(batch):
                s, start = items[i]
                n = len(s) - start
                spans.append((i, len(tgt), n))
                b_idx.extend([b] * n)
                t_idx.extend(range(start - 1, len(s) - 1))
                tgt.extend(s[start:])
            sel = h[torch.tensor(b_idx, device=h.device), torch.tensor(t_idx, device=h.device)]
            tgt_t = torch.tensor(tgt, device=h.device)
            lp = torch.empty(len(tgt), dtype=torch.float32, device=h.device)
            for a in range(0, len(tgt), HEAD_CHUNK):
                logits = self.adapter.head(sel[a: a + HEAD_CHUNK])
                lp[a: a + HEAD_CHUNK] = torch.log_softmax(logits, dim=-1).gather(
                    -1, tgt_t[a: a + HEAD_CHUNK, None])[:, 0]
            lp_list = lp.cpu().tolist()
            for i, off, n in spans:
                results[i] = lp_list[off: off + n]
        return results  # type: ignore[return-value]

    def _batches(self, order, items):
        batch: list[int] = []
        longest = 0
        for i in order:
            L = len(items[i][0])
            if batch and max(longest, L) * (len(batch) + 1) > self.max_batch_tokens:
                yield batch
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, L)
        if batch:
            yield batch

    @torch.no_grad()
    def logits_at(self, ids: list[int], positions: Sequence[int] | None = None) -> torch.Tensor:
        """Float32 logits ``[len(positions), V]`` predicting the token after each position."""
        t = torch.tensor([ids], dtype=torch.long, device=self.adapter.device)
        h = self.adapter.hidden(t)[0]
        if positions is not None:
            h = h[torch.tensor(list(positions), device=h.device)]
        return torch.cat([self.adapter.head(h[a: a + HEAD_CHUNK])
                          for a in range(0, h.shape[0], HEAD_CHUNK)]).cpu()

    def text_logprobs(self, text: str) -> list[float]:
        """Per-token log-probs of a whole text after <|endoftext|> (perplexity)."""
        ids = [EOT_ID] + self.encode(text)
        if len(ids) > self.max_seq_len:
            raise SequenceTooLong(f"{len(ids)} tokens > {self.max_seq_len}")
        return self.score_ids([(ids, 1)])[0]


def logodds(yes: Sequence[float], no: Sequence[float]) -> float:
    """log(sum exp yes) - log(sum exp no): probability summed over paraphrases."""
    return _lse(yes) - _lse(no)


def _lse(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))
