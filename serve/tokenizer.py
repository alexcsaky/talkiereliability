"""Talkie tokenizer, built the same way as the official code.

The regex, vocabulary filter and special-token ids are copied from
``talkie/tokenizer.py`` at commit 35317ba (verified by reading it). The one
deliberate difference is how text is encoded: ``encode_text`` uses
``encode_ordinary``, so a string such as ``<|user|>`` inside a passage stays
ordinary text and can never become a control token. The official code uses
``allowed_special="all"``.
"""

from __future__ import annotations

from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

_PAT_STR = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

BASE_VOCAB_SIZE = 65536
EOT_ID = BASE_VOCAB_SIZE - 1
_BASE_SPECIAL = {"<|endoftext|>": EOT_ID}
_IT_SPECIAL = {
    "<|endoftext|>": EOT_ID,
    "<|end|>": BASE_VOCAB_SIZE,
    "<|user|>": BASE_VOCAB_SIZE + 1,
    "<|assistant|>": BASE_VOCAB_SIZE + 2,
    "<|system|>": BASE_VOCAB_SIZE + 3,
}


def build_tokenizer(vocab_path: str | Path, style: str = "base") -> tiktoken.Encoding:
    ranks = load_tiktoken_bpe(str(vocab_path))
    ranks = {k: v for k, v in ranks.items() if v < BASE_VOCAB_SIZE - 1}
    special = dict(_IT_SPECIAL if style == "it" else _BASE_SPECIAL)
    return tiktoken.Encoding(name=f"talkie-{style}", pat_str=_PAT_STR,
                             mergeable_ranks=ranks, special_tokens=special)


def encode_text(enc: tiktoken.Encoding, text: str) -> list[int]:
    """Encode text with no special tokens, ever."""
    return enc.encode_ordinary(text)


_VALID_ROLES = frozenset(("system", "user", "assistant"))


def chat_ids(enc: tiktoken.Encoding, messages: list[dict]) -> list[int]:
    """Chat template of the official code, built from ids, not marker strings.

    ``<|system|>..<|end|><|user|>..<|end|>..<|assistant|>``; message text is
    encoded ordinarily, so markers inside it are inert.
    """
    sp = {k: enc.encode_single_token(k) for k in _IT_SPECIAL}
    ids: list[int] = []
    for m in messages:
        role = m.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"unknown chat role {role!r}; expected one of {sorted(_VALID_ROLES)}")
        ids.append(sp[f"<|{role}|>"])
        ids.extend(encode_text(enc, m["content"]))
        ids.append(sp["<|end|>"])
    ids.append(sp["<|assistant|>"])
    return ids
