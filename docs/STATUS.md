# Status

Last updated: 21 Nov 2026, after the stage 0 GPU run. **Stopped at the stage 0 gate.**

**Stage 0 (model checks): run on Modal on 21 Nov 2026 with Alex's go-ahead. All jobs completed;
the numbers are below and in `results/stage0/report.md`. Waiting for Alex's gate decision.**

## Stage 0 results (verified: measured 21 Nov 2026)

Full tables: `results/stage0/report.md` (`python -m analysis.stage0_report`).
Cost: $0.57 of GPU for the run (parity $0.07, base $0.37, web $0.08, chat $0.05);
$0.75 project total against the $25 cap.

1. **Parity: exact.** The official checkpoint and the vendored port give
   bit-identical logits on all 41 prompts (max difference 0.0, with and
   without `<|endoftext|>`; top-1 agreement 1.0). Both loaders read different
   files through different code, and the log-probabilities are finite and
   sensible (about 2.6 nats/token), so this looks genuine rather than a
   self-comparison, but it is tighter than the 0.03 tolerance anticipated.
   **Batched right-padded scoring differs from one-at-a-time scoring by up to
   0.12 nats on a single token** (about one bf16 step at logit sizes of 16 to
   32). Hypothesis: harmless if every arm is scored with identical batching,
   which rule 6 needs anyway; worth fixing `max_batch_tokens` per run.
2. **Context: usable to 4,096 tokens.** Loss on a fixed span falls from 2.34
   to 2.28 nats/token as the context grows from 1k to 4k; needle retrieval
   0.87 to 0.97 at every length (chance 0.25).
3. **Throughput:** about 5,600 tokens/s on one L40S at all batch sizes tried,
   26 to 29 GiB peak; 241 tokens per 1,000 characters of Gutenberg text.
4. **Known truth: close to chance, and so is the ceiling.** Base model AUROC
   (family-centred) 0.546, 95% interval 0.492 to 0.601; without the document
   start token 0.549 (0.496 to 0.602). By family: baseball 0.70, census 0.60,
   negation 0.55, Nobel 0.51. **The web twin, meant as a ceiling, scores 0.493
   (0.431 to 0.553)**, with census 0.36 and negation 0.38. Negation statements
   get log-odds of about +5 whether true or false. Hypothesis: the questions
   are too fine-grained (Nobel decoys are real laureates of neighbouring
   years; census pairs are close in size), so this set cannot show whether
   elicitation works. A failing ceiling points at the instrument, not the
   model.
5. **Leak probes: the base model knows some post-1930 facts.** It ranks the
   true answer first on 14 of 21 post-1930 probes (chance about 3.8) and on
   6 of 6 pre-1931 controls; the web twin gets 20 of 21. Strongest: the 1932
   election (0.99), Manchukuo (1.00), the New Deal (0.97), Gestapo (0.93),
   Hitler as Chancellor (0.86), Rhineland (0.75), the United Nations (0.70),
   war in 1939 (0.65), Hiroshima (0.61). Misses: Guernica, Molotov-Ribbentrop,
   Munich, West German capital, Spanish Republic month, gold standard month.
   Several hits have surface cues (Roosevelt and Hitler were well known by
   1930; "Gestapo" and "Manchukuo" are the most word-like options; "new deal"
   was an idiom), so some is plausibility rather than leakage (hypothesis),
   but the year probe for war in 1939, the United Nations and Hiroshima are
   hard to explain that way and match the leaks Talkie's authors report.
   Year probes: the base model puts 3% to 19% of its mass on 1931 or later
   when asked the present year (most likely year 1920 under every header).
   Chat model: 12 of 21 (raw), 10 of 21 (chat template).

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

## Known truth v2 and batching (verified: measured 23 Nov 2026)

Re-run with Alex's go-ahead: `stage0 --what known_truth,web,batching`, $0.62
($1.36 project total). Tables in `results/stage0/report.md`; the v1 results
are kept in `results/stage0/v1_2026-11-21/`.

1. **Known truth v2: the reading method works.** Base model AUROC
   (family-centred) **0.889** (95% interval 0.823 to 0.938, 65 clusters), up
   from 0.546 on v1; without the document-start token 0.892. By family:
   baseball 1.00, elections 1.00, events 0.997, offices 0.996, Nobel 0.986,
   negation 0.80 (offices 0.84, census top-10 0.79). The web twin scores 0.812.
2. **Except the census comparisons: 0.245, well below chance** (web twin
   0.138). Labels checked and correct. The model rates "New Haven had a larger
   population than Chicago" as true. Hypothesis: big cities are the usual
   yardstick in prose ("larger than Chicago"), so the false order, with the
   big city last, reads as more likely. The same facts asked as top-10
   membership score 0.79, so Talkie knows them; the wording is the problem.
   Without census, the base model scores 0.963. Proposed fix: keep the names
   in the same places and vary only the comparative ("X had a larger / a
   smaller population than Y").
3. **Batching: scores do not depend on batch neighbours or budget.** On 8,232
   sequences, production batching gives bit-identical scores to batching half
   the prompts alone, and to budgets of 8,192 and 32,768 tokens. Against one
   at a time, 99.7% of sequences are identical and the largest log-odds
   difference is 0.045; one at a time is 6.7 times slower. Recommendation:
   keep batched scoring at the production budget.

## Next session

23 Nov, later: census comparisons reworded (v2.1) and re-run; still below
chance (0.128), so the wording is not the cause; see `docs/DECISIONS.md`.
Headline 0.888. Batched scoring kept. Waiting for Alex on what to do with the
census comparisons; the known-truth gate passes once that is settled. Then:
write the leak-signature rule into `docs/EXPERIMENT_PLAN.md` and relabel the
web twin there, before the freeze.

## In progress

Nothing is running.

## Waiting on Alex

1. **Settled 23 Nov** (see `docs/DECISIONS.md`): known-truth set rebuilt
   with coarser questions; leaks flagged by signatures, not closed-book
   confidence; batching measured before choosing; web twin kept as a
   modern-data comparison, not a ceiling. **Open: spot-check the hand-entered
   office, election and event tables** (`controls/build_known_truth.py`), and
   go-ahead for the v2 GPU re-run. Optional: rename the
   environment variables to `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` and set a
   Modal spend limit.
2. **Review, for period language and fair decoys** (these are controls, not
   confirmatory prompts):
   - the known-truth headers in `questions/templates/known_truth.yaml`;
   - the leak probes in `controls/leak_probes.yaml`;
   - the firewall lexicon in `firewall_lexicon.yaml`.
3. Still open from the brief, needed by stage 1:
   - which modern model builds the ledger;
   - who audits the 200 verdicts;
   - whether a failed persistence test ends the experiment.
