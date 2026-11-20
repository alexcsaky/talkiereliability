<!-- The build brief for the coding agent, written by Claude for Alex Csaky on 19 Nov 2026, copied verbatim from the first build session (20 Nov 2026). Where it and docs/EXPERIMENT_PLAN.md differ, the plan wins on method and this brief wins on engineering. Changes after that date are recorded in docs/DECISIONS.md, not here. -->

# Talkie reliability backtest: build brief for the coding agent

Written by Claude for Alex Csaky, 19 Nov 2026. Companion to the experiment plan:
https://claude.ai/artifact/AYwvpJn8FgJfB7oH7dWTCL (export it to `docs/EXPERIMENT_PLAN.md` in the repo; where the two differ, the plan wins on method and this brief wins on engineering).

Labels used here: **(verified)** a research agent read it in a primary source on 19 Nov 2026; **(reported)** secondary source; **(hypothesis)** an estimate or proposal. Nothing below has been run. Treat every technical fact as something to confirm in code before relying on it.

## 1. What you are building

A harness that tests one claim: a forecaster told the track record of its sources forecasts better than one that reads the same sources without it.

- **Forecaster:** Talkie, a 13B language model trained only on pre-1931 English text.
- **Task:** resolved questions about 1931 to 1940, answered through token probabilities.
- **Evidence:** passages from a pre-1931 archive, found by lexical search.
- **Treatment:** a reliability ledger giving each source a track record from forecasts that resolved by the end of 1930.
- **Main result:** paired Brier difference between real and shuffled track records on identical passages.

Build in the five stages of section 8. Each stage ends in a gate. Stop at each gate, write the result to `docs/STATUS.md`, and wait for Alex before starting the next stage.

## 2. Rules that override everything else

These protect the experiment's validity. If a task seems to need breaking one, stop and ask.

1. **Nothing dated after 31 Dec 1930 reaches Talkie.** Every string bound for the model passes through one module, `firewall.py`, which checks the publication date, strips modern matter and control markers, runs the anachronism check, and logs what it passed.
2. **No modern-model text reaches Talkie in the confirmatory mode**, apart from the frozen prompt templates and question statements, which are reviewed for anachronism and committed before any run.
3. **No web search anywhere in the ledger pipeline.** Verdicts rest only on passages in the frozen archive.
4. **No learned retriever or modern embedding model.** Search is BM25 only.
5. **No quantised weights.** bf16 only.
6. **Identical evidence across the reliability arms.** Passages are retrieved once per question, cached, and reused by arms 1 to 5.
7. **The confirmatory run uses the base model.** The chat model appears only in the librarian mode and the demo.
8. **Outcomes are held apart.** Question outcomes live in a separate file that the forecasting and librarian code cannot import. Only `analysis/` reads it.
9. **Freeze before the full run.** Templates, question set, panel, ledger, exclusion rules and analysis code are tagged in git before stage 3. No changes after the tag without a dated entry in `docs/DECISIONS.md`.
10. **Report everything.** No arm, mode or family is dropped from the results because it is null or negative.

## 3. Facts about Talkie

- Official repos, Apache 2.0 **(verified)**: `talkie-lm/talkie-1930-13b-base` (`final.ckpt`, 53.1 GB), `talkie-lm/talkie-1930-13b-it` (`rl-refined.pt`, 26.6 GB), `talkie-lm/talkie-web-13b-base` (53.1 GB, different vocabulary). Code: https://github.com/talkie-lm/talkie
- Custom architecture, not a transformers class: 40 layers, 40 heads, d=5120, RoPE, SwiGLU, RMSNorm **(verified)**. Weights are raw PyTorch pickles.
- Tokeniser: tiktoken BPE, 65,536 entries. The chat model adds `<|end|>`, `<|user|>`, `<|assistant|>`, `<|system|>` **(verified)**.
- The official API has `generate`, `chat`, `stream`, `batch_generate`. No log-probability call; `forward` returns last-position logits only; no KV cache **(verified)**.
- `temperature=0` divides by zero in the official code; use `top_k=1` **(verified)**.
- Prompts are encoded with `allowed_special="all"`, so the chat markers inside retrieved text become control tokens **(verified)**. The firewall must strip them.
- Context: default `max_seq_len` was 2,048 at release and changed to 4,096 on 19 May 2026 **(verified)**. One porter reports quality falling past about 2k **(reported)**.
- Community transformers ports, bf16 safetensors, `trust_remote_code` **(verified, model cards)**: `xlr8harder/talkie-1930-13b-base-tf`, `xlr8harder/talkie-1930-13b-it-tf`, `xlr8harder/talkie-web-13b-base-tf`. The card reports a maximum logit difference of 0.03 against the reference. A fork with a KV cache: `a9lim/talkie-1930-13b-it-hf-cached`.
- At least 28 GB of GPU memory in bf16 **(verified, README)**.
- **Leakage (verified, authors' post):** the 13B model knows some details of the Second World War, the United Nations and the division of Germany. The chat model was post-trained with Claude Sonnet 4.6 as judge and on chats with Claude Opus 4.6.
- Community leak probe set: `trumancai/talkie-1930-knowledge-bench`, 55,519 probes **(reported)**.

Security: `trust_remote_code` and `torch.load` both execute code from the hub. Pin every model repo to a commit hash, read the remote code before first use, and load only inside the Modal container.

## 4. Infrastructure

- **Compute (hypothesis, recommended):** Modal, one L40S (48 GB, $1.95 an hour, per-second billing, scale to zero). Weights, corpus and index on Modal volumes. The cloud coding session has no GPU and limited disk, so all heavy work runs as Modal functions.
- **Serving:** a small FastAPI app on plain transformers, not vLLM, for scoring. Endpoints:
  - `POST /score` with `{model, prefix, continuations: [str]}` returns, for each continuation, the summed log-probability, token count and per-token log-probabilities.
  - `POST /score_batch` for lists of the above, length-sorted internally.
  - `POST /generate` with `{model, prompt, max_tokens, top_k, top_p, temperature, seed}` for the librarian mode and demo.
  - `GET /health` returning model commit hashes and dtype.
- **Search:** SQLite FTS5 (or Tantivy) BM25 over passages, on CPU.
- **Modern model:** used only for ledger extraction and judging, and for the librarian. Model choice is Alex's (section 10).
- **Secrets as environment variables:** `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`, the modern-model API key. Never commit them.
- **Spend guards:** every Modal function has a timeout and scales to zero; a `budget.py` ledger records GPU seconds and API spend and refuses to start a job that would cross the cap. Caps are Alex's to set; until he does, use $25 of GPU and $50 of API spend in total as placeholders **(hypothesis)** and stop when reached.

Estimated cost **(hypothesis, arithmetic only)**: stage 0 and the pilot $2 to $10; full run $30 to $180 for 150M to 400M prompt tokens. Measure real tokens per second in stage 0 and update `docs/STATUS.md`.

## 5. Repo layout

```
talkie-backtest/
  AGENTS.md            shared brief for coding agents (start from this file)
  CLAUDE.md            first line: @AGENTS.md
  README.md
  docs/
    EXPERIMENT_PLAN.md   export of the plan
    STATUS.md            Current facts / In progress / Waiting on Alex
    DECISIONS.md         dated decisions and reasons
    LEAKAGE.md           every leak check, its result, and known leaks
    PREREGISTRATION.md   written at the end of stage 2
  firewall.py
  budget.py
  serve/         modal_app.py, parity_check.py, context_test.py
  archive/       ingest/, filters.py, index.py, search.py
  ledger/        panel.py, extract.py, judge.py, verify_quotes.py,
                 canaries.py, score.py, persistence.py
  questions/     frames/, templates/, generate.py, resolve.py,
                 shortcut_check.py, headline.yaml, canaries.yaml, mirror/
  forecaster/    prompts/, elicit.py, arms.py, pooling.py, calibrate.py
  controls/      known_truth.py, leak_probes.py, synthetic_task.py,
                 manipulation_check.py
  librarian/     librarian.py, guard.py, session.py
  analysis/      metrics.py, bootstrap.py, mixed_model.py, report.py
  data/          gitignored; fetch scripts only
  outcomes/      question outcomes; importable by analysis/ only
  results/       JSONL run outputs and reports
  tests/
```

Keep the repo private. Update and commit `docs/STATUS.md` at the end of every session.

## 6. Data records (JSONL)

- **passage:** `id, corpus, source_key, source_name, byline, pub_date, text, ocr_score, dedupe_cluster, firewall_pass`
- **source:** `source_key, name, type (newspaper | wire | columnist | service | speaker), in_panel`
- **claim:** `id, source_key, passage_id, quote, claim_date, topic, resolvable_by`
- **verdict:** `claim_id, judge, verdict (borne_out | not_borne_out | mixed | unresolved_by_1930), evidence_passage_id, evidence_quote, quote_verified, source_masked`
- **track_record:** `source_key, topic, n, k, shrunk_score, level (three bands), cutoff_claim_year, cutoff_outcome_year`
- **question:** `id, family, cluster, format (binary | mc | bins), statements (with paraphrases), resolution_date, stratum (rule | contemporaries | headline | canary | mirror), leak_flag`
- **forecast:** `question_id, arm, mode, template_id, passage_id, logodds_raw, p_raw, p_calibrated, model_commit, run_id`

## 7. Components

### 7.1 Archive
- First corpora **(access unverified: no data host was reachable during research)**: `dell-research-harvard/AmericanStories` (per-year load; uses a loading script), `dell-research-harvard/newswire`, historic Hansard XML, Stanford's parsed Congressional Record (`hein-bound.zip`), Miller Center speeches, then the Commercial and Financial Chronicle on FRASER and Literary Digest issues on the Internet Archive.
- Pilot slice: American Stories 1925 to 1930, Newswire 1900 to 1930, Hansard 1919 to 1930, Congressional Record 56th to 71st Congresses. Rough size 50 to 80 GB **(hypothesis)**.
- Filters, in order: issue-level date before 1931-01-01; drop derived modern fields (topic tags, Wikidata links) and boilerplate; anachronism lexicon and n-gram check; dedupe wire copy to the earliest dated appearance; chunk to about 120 words.
- Do not use: Institutional Books or HathiTrust (gated, no redistribution), Trove in bulk, modern re-OCR.
- Tests: a planted set of post-1930 passages must be rejected by the firewall at 100 per cent.

### 7.2 Reliability ledger
- **Panel:** 20 sources for the stage 1 pilot, 60 to 150 for stage 3, chosen by a coverage rule committed before extraction (minimum forward-looking statements before 1928 and minimum passages 1928 to 1930).
- **Extract:** forecast-like sentences from a deterministic sample of each source's passages (fixed seed, no selection by content beyond a forecast classifier).
- **Judge:** source identity masked; verdict must quote an archive passage; `verify_quotes.py` string-matches the quote and rejects the verdict if absent; "unresolved by 1930" allowed; claims resolving after 1930 excluded.
- **Canaries:** claims that resolved only after 1930, mixed in. Any verdict other than "unresolved" counts towards the leak rate reported in `docs/LEAKAGE.md`.
- **Second judge:** Talkie's base model on a sample, by paired-statement scoring. Export a 200-verdict audit sheet for a person.
- **Score:** beta-binomial shrinkage towards the source-type mean; topic scores only with about 10 or more claims; three bands for the in-prompt cue.
- **Persistence test (`persistence.py`):** score claims made up to 1920 on outcomes up to 1925; test whether that predicts each source's accuracy on claims resolving 1926 to 1930 (rank correlation with a bootstrap interval, and a comparison against shuffled sources). This is the stage 1 gate.
- **Seeds:** Hamilton's WSJ editorials (https://viking.som.yale.edu/will/research.papers/doweditorials.pdf, contents not opened), the appendix of Dominguez, Fair and Shapiro 1988, and Cox's 1929 appraisal of six forecasting services (https://www.nber.org/chapters/c1086.pdf, not yet read). Later studies point to pre-1931 text; they are never verdict evidence.

### 7.3 Questions
- Generate by rule: entity lists as of 1930 crossed with templates and horizons; thresholds from pre-1931 data only; exhaustive families or matched opposites; yes rate 40 to 60 per cent; no cluster above about 10 items.
- Families and counts are in the plan. Pilot 150, full about 1,500, plus an unscored 1921 to 1930 mirror of about 300.
- Period language only. Maintain a banned-vocabulary list (for example "World War II", "New Deal", "United Nations", "Nazi Germany" as a state) and test for it.
- `shortcut_check.py`: a bag-of-words classifier from question text to outcome, cross-validated. If it beats chance, rewrite the family.
- Several ground-truth datasets forbid redistribution (Correlates of War, Polity, JST). Commit fetch scripts and derived questions, not the datasets.

### 7.4 Forecaster
- **Reading a forecast:** each binary question is a statement A and its negation, three or four paraphrases each, scored as continuations of a period header (for example a dated almanac chronicle or a dispatch line). Sum probability over paraphrases per outcome, take log-odds, average over four to six templates. No PMI normalisation.
- **Evidence:** one passage per forward pass; pool per-passage log-odds. A concatenated variant with random order is a secondary check.
- **Arms:** 0 closed book; 1 archive; 2 real records; 3 shuffled; 4 inverted; 5 anonymous sources (arms 1 and 2 with names masked); ceiling: web twin closed book.
- **Modes for arms 2 to 4:** A in-prompt cue in period phrasing, three levels; B re-rank or filter passages by record; C weight per-passage log-odds by shrunk record in code (reuses arm 1 passes). Also an oracle run of mode C weighted by actual 1931 to 1940 accuracy, as a ceiling.
- **Calibration:** one temperature-and-bias map, cross-fitted over question families, fitted on all arms pooled.
- Keep prompts under about 1,800 tokens until `context_test.py` shows more is safe.

### 7.5 Controls
- `known_truth.py`: 1900 to 1929 questions closed book; the reading method must separate true from false.
- `leak_probes.py`: every target question closed book; tell-tale strings against matched decoys; the canary family; "what year is it" probes.
- `synthetic_task.py`: invented entities, two conflicting sources, only the record identifies the answer.
- `manipulation_check.py`: swap records on fixed passages; record the change in log-odds.

### 7.6 Librarian (exploratory)
- Talkie's chat model writes plain-language requests; a modern model turns them into `search(query)` or `ledger_lookup(source_key)` calls.
- `guard.py` enforces: the librarian sees only Talkie's request; query terms must be a subset of the request's words; ranking is the fixed BM25; results are verbatim passages at fixed length; a fixed refusal line for anything about events after 1930 or requests for its own view; full logging; divergence from a scripted parse flagged.
- Tests: canary requests must produce no added post-1930 term; a scripted run of the same sessions with no librarian for comparison.

### 7.7 Analysis
- Metrics: Brier, log score, calibration curve, AUROC.
- Primary: real against shuffled, per mode, Holm-corrected across modes. Secondary: real against none, inverted against none, dose check.
- Tests: paired differences with a bootstrap clustered by question family; mixed model with random effects for question and family.
- Exclusions fixed in advance: leak-flagged questions out of the primary analysis and reported separately; headline and canary strata never pooled.
- `report.py` writes one report with every arm, mode and family.

## 8. Stages and gates

Thresholds marked **(proposal)** are for Alex to confirm before the stage starts.

| Stage | Build | Gate |
| --- | --- | --- |
| 0. Model checks | Serving, parity check, context test, throughput, leak probes, known-truth check | Parity within the port's stated tolerance; known-truth AUROC clearly above chance, at least 0.65 **(proposal)**; leak-flagged share of pilot questions reported |
| 1. Archive and ledger pilot | Pilot corpus, index, firewall tests, 20-source ledger, canaries, persistence test | Firewall rejects all planted post-1930 passages; ledger canary leak rate reported; persistence result reported with its interval |
| 2. Pilot | 150 questions, all arms and modes, synthetic task, manipulation check | Mode A moves output on the synthetic task; variance and clustering estimated; `PREREGISTRATION.md` drafted for Alex |
| 3. Full run | Full panel, ledger, about 1,500 questions, analysis, report | None; report everything |
| 4. Exploratory | Librarian sessions, demo, replication on `typewriter-ai/typewriter-1913-7B-base-v2` | None |

## 9. Unknowns to settle by running code

1. Usable context: perplexity and retrieval accuracy at 1k, 2k, 3k and 4k tokens.
2. Whether full-sequence logits from the port match the official code on a few dozen prompts.
3. Real prompt tokens per second on the chosen GPU, with batching.
4. Leak rate on 1931 to 1940 for the base and chat models with no documents supplied.
5. Tokens per 1,000 characters on OCR-noisy period text.
6. Whether the American Stories loading script still works and what the 1925 to 1930 slice weighs.
7. FRASER API key terms and limits.
8. Ledger cost per source with the chosen modern model.

## 10. Ask Alex before proceeding

- Spend caps for GPU and for the modern-model API.
- Which modern model builds the ledger and acts as librarian.
- Gate thresholds marked as proposals.
- Who audits the 200 verdicts.
- Repo name and owner; accounts for Modal and Hugging Face.
- Whether the persistence test may end the experiment if it fails.
