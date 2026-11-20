# AGENTS.md: shared brief for coding agents

This repo is a backtest of one claim: a forecaster told the track record of
its sources forecasts better than one that reads the same sources without it.
The forecaster is Talkie (a 13B model trained only on pre-1931 text), asked
about 1931 to 1940.

Read, in order:

1. `docs/STATUS.md`: where things stand and what is waiting on Alex.
2. `docs/BUILD_BRIEF.md`: the engineering brief (rules, components, stages).
3. `docs/EXPERIMENT_PLAN.md`: the method. On method the plan wins; on
   engineering the brief wins.
4. `docs/DECISIONS.md`: dated decisions that refine or depart from either.

## Rules that override everything (brief section 2, short form)

1. Nothing dated after 31 Dec 1930 reaches Talkie. Every string bound for the
   model goes through `firewall.admit()`; `forecaster/client.py` refuses
   anything else.
2. No modern-model text reaches Talkie in the confirmatory mode, except frozen,
   reviewed templates and question statements.
3. No web search in the ledger pipeline. 4. BM25 only, no learned retriever.
5. bf16 only, no quantisation. 6. Identical cached evidence across arms 1 to 5.
7. Confirmatory run uses the base model. 8. Outcomes live in `outcomes/` and
   only `analysis/` reads them (`tests/test_outcome_isolation.py` enforces it).
9. Freeze (git tag) before stage 3. 10. Report everything.

If a task seems to need breaking a rule, stop and ask Alex.

## Working conventions

- Work in stages; stop at each gate, write the result to `docs/STATUS.md`,
  and wait for Alex. Gate thresholds marked "proposal" are Alex's to confirm.
- Every paid job runs inside `budget.gpu_job(...)` or `budget.api_job(...)`.
  Caps are in `budget.yaml`; the ledger is `results/budget_ledger.jsonl`
  (committed). Never raise a cap without Alex.
- All GPU work runs on Modal (`serve/modal_app.py`). The coding session has no
  GPU. Models are pinned by commit in `serve/models.py`; the port's code is
  vendored and reviewed (`serve/vendor/talkie_port/`), so no hub code runs.
- Label facts in docs: **verified** (read in a primary source or measured),
  **reported** (secondary source), **hypothesis** (estimate or proposal).
- Use British English in prose.
- Update and commit `docs/STATUS.md` at the end of every session.

## Layout

```
firewall.py, firewall_lexicon.yaml   the gate for every string bound for Talkie
budget.py, budget.yaml               spend guard and caps
serve/      scoring engine, tokenizer, loaders, FastAPI app, Modal app,
            parity_check.py, context_test.py, throughput.py, vendor/
forecaster/ client.py (only route to the model), elicit.py (reading a forecast)
controls/   known_truth.py, build_known_truth.py, leak_probes.py(.yaml)
questions/  question sets and templates (no outcomes)
outcomes/   outcomes, read only by analysis/
analysis/   metrics.py, stage0_report.py
archive/, ledger/, librarian/        stages 1 to 4 (empty so far)
data/       gitignored; fetch scripts only
results/    run outputs, firewall and budget logs
tests/      pytest; CPU only, tiny random models
```

## Running things

```bash
uv venv .venv --python 3.12 && . .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e ".[model,dev]" "talkie @ git+https://github.com/talkie-lm/talkie@35317ba3a84861a84c84065bd73faf88ad19329c"
./data/fetch_vocab.sh
python -m pytest
```

Modal (needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`; in this cloud
environment the client needs `modal[api-proxy-support]`, already in
`pyproject.toml`):

```bash
modal run serve/modal_app.py::download
modal run serve/modal_app.py::stage0 --what all
python -m analysis.stage0_report
```
