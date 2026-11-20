# Status

Last updated: 20 Nov 2026, end of the first build session.

**Stage 0 (model checks): code built and tested on CPU. Not yet run on a GPU**,
because there are no Modal credentials in this environment.

## Current facts

Verified this session, by reading code or the hub API or by measuring:

- Official Talkie code at commit `35317ba` (19 May 2026): `forward` returns
  last-position logits only; `temperature=0` divides by zero; prompts are
  encoded with `allowed_special="all"`; no token is prepended; the context
  default is 4,096; the RoPE base is 1,000,000.
- Model repos are pinned in `serve/models.py`. The base and chat checkpoints
  share one vocabulary (git blob `c3e0030`); the web twin's differs. The web
  twin's file is `base.ckpt`, not `final.ckpt`.
- The community port's code is safe: no network, file or exec calls. It is
  identical across its three repos and vendored. Two quirks:
  - Left padding shifts positions.
  - It folds the lm-head gain into the weights in bf16.
- A tiny random model in fp32 on CPU agrees with the port to 1e-3 at every
  position. The real 13B check runs in stage 0.
- Modal is reachable from this session with `modal[api-proxy-support]`; only
  credentials are missing.
- Modal prices (pricing page, 20 Nov 2026):
  - L40S: $0.000542/s
  - CPU: $0.0000131 per core per second
  - Memory: $0.00000222 per GiB per second
  - Volumes: 1 TiB a month free

  A 48 GiB L40S container is about $2.52/h; the 96 GiB parity container is
  about $2.91/h.
- Tests: 29 pass. They cover:
  - firewall: 22 of 22 planted post-1930 passages rejected, 10 genuine passages kept;
  - budget guard;
  - scoring: batching invariance, manual check, guards;
  - parity and context code, run on tiny models;
  - known-truth and leak runs end to end;
  - outcome isolation.

## Built (stage 0)

| Piece | File |
| --- | --- |
| Firewall and lexicon | `firewall.py`, `firewall_lexicon.yaml` |
| Spend guard (placeholder caps: $25 GPU, $50 API) | `budget.py`, `budget.yaml` |
| Scoring engine (official and port adapters) | `serve/scoring.py`, `serve/tokenizer.py` |
| FastAPI endpoints `/score`, `/score_batch`, `/generate`, `/health` | `serve/api.py` |
| Modal app: weights download, server, stage 0 jobs | `serve/modal_app.py` |
| Parity, context and throughput tests | `serve/parity_check.py`, `serve/context_test.py`, `serve/throughput.py` |
| Reading a forecast (statement against negation, paraphrases, templates) | `forecaster/elicit.py`, `forecaster/client.py` |
| Known-truth set: 226 questions (75 census, 125 Nobel, 26 World Series) | `controls/build_known_truth.py`, `questions/known_truth.jsonl` |
| Leak probes: 23 post-1930 probes, 6 pre-1931 controls, 5 year probes | `controls/leak_probes.yaml`, `controls/leak_probes.py` |
| Stage 0 report | `analysis/stage0_report.py` |

## Stage 0 run plan and cost (hypothesis)

1. `modal run serve/modal_app.py::download`: about 131 GB to a Modal volume
   (three ports plus the official base checkpoint), CPU only, about $0.10.
2. `modal run serve/modal_app.py::stage0 --what all`:
   - parity on 38 short and 3 long prompts;
   - the context test at 1k to 4k;
   - throughput;
   - known truth, with and without the document-start token;
   - leak probes on base, chat and web;
   - known truth on the web twin as a ceiling.

   The guard reserves at most about $10. Expected actual spend is $3 to $5.
3. `python -m analysis.stage0_report`, then copy the numbers here and into
   `docs/LEAKAGE.md`, and stop at the gate.

## Gate for stage 0 (from the brief)

- Parity within the port's stated tolerance (max logit difference about 0.03).
- Known-truth AUROC clearly above chance, **at least 0.65 (proposal)**.
- Leak-flagged share of pilot questions reported. The pilot questions are built
  in stage 2, so stage 0 reports the provisional probes, and the closed-book
  probe of the pilot questions runs at the start of stage 2.

## In progress

Nothing is running.

## Waiting on Alex

1. **Repo visibility.** `alexcsaky/talkiereliability` is public; the brief says
   keep it private. Nothing is pushed until Alex makes it private or says
   public is fine.
2. **Modal credentials** as environment variables `MODAL_TOKEN_ID` and
   `MODAL_TOKEN_SECRET` in this cloud environment's settings (a new session
   picks them up). A dedicated, revocable Modal token and a workspace spend
   limit in Modal's settings are a good backstop. Alternatively, Alex runs the
   three commands above on his own machine.
3. **Spend caps** for GPU and API. Placeholders are $25 and $50.
4. **Gate threshold:** confirm known-truth AUROC of at least 0.65.
5. **Review, for period language and fair decoys** (these are controls, not
   confirmatory prompts):
   - the known-truth headers in `questions/templates/known_truth.yaml`;
   - the leak probes in `controls/leak_probes.yaml`;
   - the firewall lexicon in `firewall_lexicon.yaml`.
6. Still open from the brief, needed by stage 1:
   - which modern model builds the ledger;
   - who audits the 200 verdicts;
   - whether a failed persistence test ends the experiment.
