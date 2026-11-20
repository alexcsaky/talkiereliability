# Vendored Talkie transformers port

Source: `xlr8harder/talkie-1930-13b-base-tf` at commit
`7cf020702ef7fd68ac16c85aaea389206ded519e` (Apache 2.0). The same two files are
byte-identical in `xlr8harder/talkie-1930-13b-it-tf` (`f858792d565ab60b38516890553f64960ebef8a7`)
and `xlr8harder/talkie-web-13b-base-tf` (`963f2bba0c3fceacb69f57f671015ca3eb162808`).

| File | SHA-256 |
| --- | --- |
| modeling_talkie.py | 005df4d724657ef6d3b9d39425d59893f61d22bcce0b05e948898d88451e6e34 |
| configuration_talkie.py | 49e41f9546961be5d9c51843e914a1ac6f7c0b6ee94a7a5f23d5ab021481564c |

Read in full by Claude on 20 Nov 2026 before first use: imports only torch and
transformers, no network, file, subprocess or pickle calls.

Behaviour that matters here (verified by reading the code):

- Positions come from `arange(seq_len)`, not from the attention mask, so
  **left padding shifts positions**. `serve/scoring.py` right-pads and reads
  logits only at real positions.
- The single `lm_head_gain` scalar of the official model is folded into
  `lm_head.weight` during conversion (bf16 rounding). The parity check
  measures the effect.
- The port's tokenizer (not vendored) encodes text with `allowed_special="all"`,
  like the official code. We use our own tokenizer in `serve/tokenizer.py`,
  which never turns text into control tokens.

The files are unmodified. Do not edit them; update by re-vendoring from a new
pinned commit and recording it in docs/DECISIONS.md.
