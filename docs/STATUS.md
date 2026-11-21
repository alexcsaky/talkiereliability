# Status

Last updated: 21 Nov 2026, after the weights download (before first GPU run).

**Stage 0 (model checks): code built, reviewed, hardened and tested on CPU. Modal credentials
work and the weights are on the Modal volume. No GPU job has run yet**; waiting for Alex's go-ahead.

## Current facts

Verified by reading code or the hub API or by measuring:

- Official Talkie code at commit `35317ba` (19 May 2026): `forward` returns
  last-position logits only; `temperature=0` divides by zero; prompts are
  encoded with `allowed_special="all"`; no token is prepended; the context
  default is 4,096; the RoPE base is 1,000,000.
- Official 53.1 GB FP32 checkpoint (`final.ckpt`): verified via HTTP Range
  inspection of the ZIP64 central directory and `data.pkl` to be uncompressed
  ZIP64 (`mmap=True`-compatible), using only five pickle globals whitelisted by
  PyTorch 2.8 `weights_only=True`, with all 443 `_orig_mod.`-stripped keys
  matching `TalkieModel(GPTConfig(vocab_size=65536))`.
- Model repos are pinned in `serve/models.py`. The base and chat checkpoints
  share one vocabulary (git blob `c3e0030`); the web twin's differs. The web
  twin's file is `base.ckpt`, not `final.ckpt`.
- The community port's code is safe: no network, file or exec calls. Verified
  byte-identical across all three port repos and vendored. Verified loading via
  `TalkieForCausalLM.from_pretrained(..., dtype=torch.bfloat16,
  use_safetensors=True)` under `transformers==5.8.0` and `accelerate` with
  `_attn_implementation="sdpa"`. Two quirks:
  - Left padding shifts positions.
  - It folds the lm-head gain into the weights in bf16.
- A tiny random model on CPU agrees with the port at every position (both with
  and without `<|endoftext|>` and under batched right-padding). The real 13B
  check runs in stage 0.
- Modal works from this session with `modal[api-proxy-support]` (21 Nov). The
  cloud environment holds the credentials as `MODAL_ID` and `MODAL_SECRET`, not
  `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, so map them per command:
  `export MODAL_TOKEN_ID="$MODAL_ID" MODAL_TOKEN_SECRET="$MODAL_SECRET"`.
  `modal token info` confirms token `talkieRel` in workspace `alexcsaky`.
- Weights download (21 Nov, `results/stage0/download.json`): all three ports
  match their pinned `vocab.txt` blob SHA-1 (`vocab_sha1_ok`), and the official
  `final.ckpt` matches its pinned SHA-256 (`sha256_ok`). All five Gutenberg
  context texts were fetched, admitted and cached on the volume. The firewall
  flagged (but admitted) post-1930 year strings in three of them: Keynes
  (1936, 2016), *Main Street* (2000) and *Babbitt* (1935). Not yet inspected;
  probably debt schedules, page or line numbers, or leftover boilerplate
  (hypothesis). The run took 46.5 min and settled at $0.17 (reserved $0.67).
- Modal prices (pricing page, 20 Nov 2026):
  - L40S: $0.000542/s
  - CPU: $0.0000131 per core per second
  - Memory: $0.00000222 per GiB per second
  - Volumes: 1 TiB a month free

  A 48 GiB L40S container is about $2.52/h; the 96 GiB parity container is
  about $2.91/h.
- Tests: 40 pass (re-run 21 Nov). They cover:
  - firewall: 22 of 22 planted post-1930 passages rejected, 10 genuine passages
    kept, `1930` allowed in headers, bound-volume date ranges rejected,
    pre-strip URL anachronism detection, post-1930 copyright rejection,
    `leak_probe` flag and `firewall.compose`;
  - budget guard: caps, `scaledown_s` accounting, unclamped overruns,
    `KeyboardInterrupt` retaining worst-case reservation, file locking;
  - scoring, FastAPI endpoints, generation, throughput `max_batch_tokens`
    restoration, batching invariance, guards;
  - parity (including right-padded batch invariance and `noeot`) and context
    code, run on tiny models;
  - all 312 known-truth questions passing the firewall, leak probes,
    confirmatory client refusal of `leak_probe=True`, and full non-truncating
    stage 0 report;
  - outcome isolation (including `answer_index`).

## Built (stage 0)

| Piece | File |
| --- | --- |
| Firewall and lexicon | `firewall.py`, `firewall_lexicon.yaml` |
| Spend guard (caps: $25 GPU, $50 API) | `budget.py`, `budget.yaml` |
| Scoring engine (official and port adapters) | `serve/scoring.py`, `serve/tokenizer.py` |
| FastAPI endpoints `/score`, `/score_batch`, `/generate`, `/health` | `serve/api.py`, `serve/generation.py` |
| Modal app: weights download, Gutenberg cache, server, stage 0 jobs | `serve/modal_app.py` |
| Parity, context and throughput tests | `serve/parity_check.py`, `serve/context_test.py`, `serve/throughput.py` |
| Reading a forecast (statement against negation, paraphrases, templates) | `forecaster/elicit.py`, `forecaster/client.py` |
| Known-truth set: 312 questions (75 census, 125 Nobel, 26 World Series, 86 negation) | `controls/build_known_truth.py`, `questions/known_truth.jsonl` |
| Leak probes: 21 post-1930 probes, 6 pre-1931 controls, 5 year probes (BPE-length-matched) | `controls/leak_probes.yaml`, `controls/leak_probes.py` |
| Stage 0 report | `analysis/stage0_report.py` |

## Stage 0 run plan and cost (hypothesis)

1. `modal run serve/modal_app.py::download`: about 131 GB to a Modal volume
   (three ports plus the official base checkpoint, verified against pinned
   SHA-256 and `vocab_sha1` before commit, plus cached pre-1931 Gutenberg
   context texts), CPU only (16 GiB RAM), about $0.10 to $0.20.
2. `modal run serve/modal_app.py::stage0 --what all`:
   - parity on 38 short and 3 long prompts (plus batched right-padding
     invariance and `prepend_eot=False`);
   - the context test at 1k to 4k;
   - throughput (tokenised remotely on the container; no local `vocab.txt` needed);
   - known truth (312 questions including statement-vs-negation), with and
     without the document-start token;
   - leak probes on base, chat (both raw completion and chat template) and web;
   - known truth on the web twin as a ceiling.

   The guard reserves at most about $14.50 if every job ran to its timeout
   (parity $4.37, base model $7.58, web and chat models $1.27 each); jobs settle
   at actual wall time.
   Expected actual spend is $3 to $5.
3. `python -m analysis.stage0_report`, then copy the numbers here and into
   `docs/LEAKAGE.md`, and stop at the gate.

## Gate for stage 0 (from the brief)

- Parity within the port's stated tolerance (max logit difference about 0.03).
- Known-truth AUROC clearly above chance. **No fixed threshold** (Alex, 20 Nov):
  the AUROC and its interval are reported and Alex decides.
- Leak-flagged share of pilot questions reported. The pilot questions are built
  in stage 2, so stage 0 reports the provisional probes, and the closed-book
  probe of the pilot questions runs at the start of stage 2.

## Settled by Alex on 20 Nov 2026

- The repo stays public; pushing the plan and brief is fine.
- Spend caps stay at $25 GPU and $50 API.
- Known-truth gate: report only, no fixed threshold.

## Next session

Done on 21 Nov: credentials checked, environment set up, 40 tests pass, and the
weights download completed and verified (see above). Next:

1. Look at the flagged year strings in the cached context texts (a CPU-only
   read of `/weights/context_texts.json`) and decide whether to strip or
   replace those texts before they reach Talkie in the context test.
2. With Alex's go-ahead, `modal run serve/modal_app.py::stage0 --what all`
   (GPU; reserves up to about $14.50, expected $3 to $5), then
   `python -m analysis.stage0_report` and stop at the gate.

## In progress

Nothing is running.

## Waiting on Alex

1. **Go-ahead for the stage 0 GPU run** (`stage0 --what all`). Optional: rename
   the environment variables to `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` so no
   mapping is needed, and set a workspace spend limit in Modal as a backstop.
2. **Review, for period language and fair decoys** (these are controls, not
   confirmatory prompts):
   - the known-truth headers in `questions/templates/known_truth.yaml`;
   - the leak probes in `controls/leak_probes.yaml`;
   - the firewall lexicon in `firewall_lexicon.yaml`.
3. Still open from the brief, needed by stage 1:
   - which modern model builds the ledger;
   - who audits the 200 verdicts;
   - whether a failed persistence test ends the experiment.
