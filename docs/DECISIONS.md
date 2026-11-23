# Decisions

Dated decisions and their reasons. Newest at the bottom of each date. "Alex"
marks decisions Alex made; everything else is the coding agent's engineering
call, open to Alex's review.

## 20 Nov 2026 (first build session, stage 0 code)

**Repo (Alex).** The work lives in `alexcsaky/talkiereliability` (the brief's
layout calls it `talkie-backtest/`; the layout is used at the repo root). The
repo is public; the brief said private, and Alex chose to keep it public,
plan and brief included.

**Spend caps (Alex).** $25 GPU and $50 API, the brief's placeholders, confirmed.

**Known-truth gate (Alex).** No fixed AUROC threshold; the report gives the
AUROC with its clustered bootstrap interval and Alex decides. Replaces the
brief's proposal of at least 0.65.

**Port code vendored instead of `trust_remote_code`.** The two model files of
`xlr8harder/talkie-1930-13b-base-tf` (commit 7cf0207) were read in full and
copied unmodified into `serve/vendor/talkie_port/` with their SHA-256. They are
byte-identical across the three port repos. Loading our reviewed copy means no
hub code runs in the containers, which is stronger than pinning alone.

**Official checkpoints loaded with `torch.load(weights_only=True, mmap=True)`.**
No pickled code can execute, and the 53 GB FP32 file streams from the volume
instead of filling memory. The model is built directly in bf16 on the GPU;
copying FP32 into bf16 rounds the same way as the official `.to(bfloat16)`.
The official checkpoint's SHA-256 is checked after download.

**Our own tokenizer, encoding with special tokens disabled.** `serve/tokenizer.py`
copies the official regex, vocabulary filter and special ids, but encodes text
with `encode_ordinary`, so `<|user|>` in a passage stays text. The firewall
also strips markers, and the scoring engine raises on any `<|` or `|>`: three
independent layers against brief section 3's control-token problem.

**Right padding in batches.** The port takes positions from `arange`, not the
attention mask, so left padding would shift positions (verified by reading the
code). With causal attention, trailing pads cannot affect real tokens.

**`<|endoftext|>` at document start, provisionally.** The official code
prepends nothing. A document separator before a period header makes the prompt
read as the start of a document, as in pretraining (hypothesis). Stage 0 runs
the known-truth check both ways (`known_truth_base.json` and
`known_truth_base_noeot.json`); keep whichever separates true from false
better, and record it here before stage 2.

**Continuation joins.** Prefix and continuation are tokenised separately and
concatenated so every option is scored on identical prefix tokens; after a
header ending in a line break the statement starts the line, otherwise it
follows a space. `boundary_ok` records whether this equals tokenising the
joined text.

**Firewall: what it rejects and what it flags.** The lexicon
(`firewall_lexicon.yaml`) lists only terms that did not exist before 1931
(e.g. "World War II", "Manchukuo", "New Deal" capitalised); words that existed
earlier (Hitler, Nazi, television, Anschluss) are left to the leak probes.
Years after 1930 are flagged, not rejected, in passages (a 1929 editorial that
names 1935 is a forecast) and in question statements (which name the target
decade); they reject templates and headers. A data-driven n-gram list
(n-grams frequent in American Stories after 1935 and absent before 1931) is
deferred to stage 1, when the corpus exists.

**One firewall exemption: leak probes.** Leak probes must score post-1930 words
such as "Manchukuo". Scoring is stateless, so a probe cannot inform any
forecast. `admit(..., leak_probe=True)` skips the lexicon and year checks, is
allowed only for kinds "probe" and "continuation", still strips control
markers, and is logged each time.

**Known-truth design.** Matched pairs from three rule-generated families with
authoritative data: census rank pairs (Census Bureau tables for 1900 to 1920),
Nobel laureate against a laureate of the same prize from a nearby year (Nobel
API), and world's series winner against loser (Lahman database via a
third-party GitHub mirror, `cbwinslow/baseballdatabank`, because the Chadwick
repo's paths returned 404; spot-checked against the record). A seeded coin
decides which side is A. Headers are retrospective (a reference work dated the
year after the event). 226 questions.

**Outcomes separated even for controls.** Known-truth outcomes and the true
option of each leak probe live in `outcomes/`; options in the probe file are
shuffled. A test fails if code outside `analysis/` mentions `outcomes/`.

**Budget guard reserves the worst case.** A Modal job reserves timeout x
(GPU + CPU + memory price) before it starts and settles at wall time x price
when it ends. Modal bills CPU and memory on top of the GPU (verified on the
pricing page, 20 Nov 2026), which adds about $0.5 an hour for a 64 GiB
container. Region is not pinned (pinning costs 1.15 to 1.75 times more,
reported).

**Modal from this cloud session.** The Modal client reached Modal's API through
the session's proxy once `modal[api-proxy-support]` was installed (a dummy
token got "Token not found" back, verified). Only real credentials are missing.

**Stage 0 gate item that cannot run yet.** "Leak-flagged share of pilot
questions" needs the pilot questions, which stage 2 builds. Stage 0 reports the
provisional probes (27 choice probes, 6 of them pre-1931 controls; 5 year
probes) and the community probe set; `leak_probes.probe_questions` runs on the
pilot questions at the start of stage 2.

## 20 Nov 2026 (stage 0 review fixes, before first GPU run)

**Firewall hardened (`firewall.py`, `firewall_lexicon.yaml`).** Fixed
`_FUTURE_YEAR_RE` so `1930` is no longer matched as a year after 1930 (which
previously rejected all six 1929 known-truth items and would have rejected
1930-dated forecasting headers). Made `parse_date` strict full-string matching
so bound-volume ranges (`1930-1935`, `1930/1940`) and malformed dates are
rejected rather than truncated to `1930`. Anachronism checks now run before URL
and boilerplate stripping so terms inside modern URLs or footnotes are caught,
and post-1930 copyright/reprint notices reject passages outright. Added
`Admitted.leak_probe` and `firewall.compose` so confirmatory `TalkieClient`
instances refuse leak-probe-exempt strings (`allow_leak_probes=False` by
default) and `elicit.build_requests` requires an already-admitted `passage`
without re-subjecting pre-1931 forecasts to `header`'s future-year ban. Tightened
`firewall_lexicon.yaml` (added `United Nations`, `Great Depression`, `WW1`/`WW2`,
`Wehrmacht`, `Luftwaffe`, `Vichy France`, `Chancellor Hitler`, `President
Franklin Roosevelt`; removed pre-1931 false positives `Superman` and bare
`Holocaust`).

**Known-truth set expanded with a statement-vs-negation family (`controls/build_known_truth.py`).**
Added 86 rule-generated statement-vs-negation questions from the same Census and
World Series sources (total 312 questions across 36 clusters of at most 10 items),
so stage 0 tests negation sensitivity alongside symmetric entity swaps. Fixed
duplicate-city keying in `_census_labels`, pinned the Lahman mirror to commit
`8a908d51833e`, fixed missing `"the"` before Nobel organisation names, used
`dateAwarded` for reserved Nobel prizes, changed `kt-handbook` to December of
`pub_year`, and removed the anachronistic `"Sir"` on Austen Chamberlain before
1925.

**Leak probes balanced for BPE token length and Rule 8 (`controls/leak_probes.*`).**
Revised `controls/leak_probes.yaml` and `outcomes/leak_probes.jsonl` to match
Talkie BPE token counts across options, replace pre-1931-predictable and
1931-edge recency-biased items, and remove the `"MONDAY MORNING"` calendar cue
from `yr-newspaper` (Jan 5 was a Monday in 1931). Removed `answer_index` scoring
from `controls/leak_probes.py::knowledge_bench` (moved to
`analysis/stage0_report.py` per Rule 8) and enforced the firewall lexicon on
`knowledge_bench` stems (`leak_probe=False`). Stage 0 now also probes
`talkie-1930-13b-it` in its chat template (`leak_it_chat.json`).

**Modal, throughput, parity and reporting fixes (`serve/`, `budget.py`, `analysis/`).**
Restored `engine.max_batch_tokens` after `throughput.measure` and moved
`reset_peak_memory_stats` before timing; tokenised Gutenberg texts remotely on
`TalkieServer.throughput` (removing the local `vocab.txt` dependency); cached
Gutenberg texts on the Modal volume during `download` and fetched once on CPU
before booting any GPU container; raised `download_weights` memory to 16 GiB,
removed dead `hf_transfer`, and refused to commit on SHA-256 or `vocab_sha1`
mismatch; lowered `scaledown_window` to 15 s and included it in `budget.gpu_job`
without clamping wall-time overruns, leaving reservations open on
`KeyboardInterrupt`, with `fcntl.flock` locking. Extended `parity_check` with
batched right-padding invariance and `prepend_eot=False` checks. Replaced
truncating JSON dumps in `analysis/stage0_report.py` with full Markdown tables
for download, health, parity, context and throughput, plus log-space and
length-normalised leak probe reporting.


## 21 Nov 2026 (review of the Stage 0 fixes)

Commit 6e1ee5f was reviewed against its handoff report: the reported fixes are
present, the 38 tests pass, and the six bugs confirmed beforehand (1930
treated as after 1930, ranges and malformed dates accepted, oversized
clusters, "Superman" false positive, batch size leaking from the throughput
test, answers read in `controls/`) are fixed. Four follow-ups:

**Headline known-truth AUROC centres log-odds within each family.** The new
negation family has a constant offset unrelated to truth (negated statements
are longer, so every item leans towards A). Pooling raw log-odds across
families mixes those offsets into the ranking and dilutes the number the gate
decision rests on. The pooled row is kept, labelled "uncentred". Centring uses
only the model's log-odds, not outcomes.

**The chat-template instruction for the chat-model leak probe goes through the
firewall.** It reaches the model like any other string; the backends now
accept only an `Admitted` value.

**The base-model Stage 0 container reserves its true worst case**: every
sequential call can run to `METHOD_TIMEOUT`, so the reservation is
calls x 1,800 s, not a 3,600 s ceiling. Total worst case for Stage 0 is
about $14.50 against the $25 cap.

**`budget.open_reservations()`** lists reservations left open by an
interrupted job; close each with `budget.settle(job_id, kind, billed_usd)`
using Modal's bill. `budget.summary()` shows the count.

Noted, not changed: the chat-probe instruction reads "Continue the following
sentence…" while the sentence itself sits in the assistant turn, not after
the instruction. Wording is for Alex to judge alongside the other probes.

## 21 Nov 2026 (weights download and stage 0 run)

**Modal credentials are mapped per command.** The cloud environment stores
them as `MODAL_ID` and `MODAL_SECRET`; each `modal` command runs with
`MODAL_TOKEN_ID="$MODAL_ID" MODAL_TOKEN_SECRET="$MODAL_SECRET"`.

**Firewall tightened after the context texts were read in full.** The flagged
post-1930 years were all genuine period text: Keynes projecting reparations to
1936 and "2016 [kilometres] had been repaired"; *Main Street* imagining "the
year 2000"; a *Babbitt* banner, "1,000,000 in 1935". The firewall did miss two
modern insertions that carry no year:
- the Gutenberg *Age of Innocence* ends with "A Note on the Text" citing the
  Library of America edition (founded 1979);
- *Main Street* opens with a "Produced by Charles Keller" transcriber credit.

Both could reach Talkie: the context test reads from each book's start, and
the retrieval filler is drawn from anywhere in the book. Changes:
- "Library of America" is added to the lexicon, so any string containing it
  is rejected;
- a strip pattern removes "Produced/Prepared/Transcribed by ..." credit
  paragraphs up to the next blank line;
- a planted reject passage and a stripping test cover both (41 tests pass).

The cache was rebuilt with the new `refresh_context` entrypoint (CPU, under
the budget guard). *The Age of Innocence* is now rejected, so stage 0 used
four texts (the retrieval and parity passages need three). Lesson for stage 1
(hypothesis): modern editorial matter without a date (introductions, notes on
the text, transcriber credits) is a leak path the date and year checks cannot
see; Gutenberg and archive sources need a front- and back-matter check.

**Stage 0 was run in one go with Alex's approval** ("check the flagged texts
first, then run stage 0"). It was launched detached (`setsid nohup`) so the
orchestrating client was not bound to the session's two-hour command limit.

## 23 Nov 2026 (Alex's stage 0 gate decisions)

Asked after the stage 0 run; options and recommendations are in the session
record. Settled:

1. **Known-truth set: rebuild with coarser questions.** Facts a 1930 corpus
   discusses often and with clear margins (for example a city twice the size,
   a laureate against a non-laureate), so that a pass or a fail says whether
   the reading method works. The current set came out at AUROC 0.546 because
   its decoys are near-misses (Nobel laureates of neighbouring years, cities
   of similar size).
2. **Leak rule: flag by leak signatures, not by closed-book confidence.** This
   replaces the plan's rule that every question the closed-book probe flags
   leaves the primary analysis. That rule cannot tell memory from a good
   forecast, so it would remove the predictable questions. A question is
   flagged only on evidence a forecast cannot explain: knowing later coinages
   or exact unforecastable details, confidence well above contemporary
   forecasters, or family-level skill close to the 1921 to 1930 mirror. The
   primary analysis keeps all questions, and results without the flagged ones
   are reported alongside. Method change to `docs/EXPERIMENT_PLAN.md`, to be
   written up in full before the freeze (rule 9).
3. **Batching: measure, then decide.** The next GPU run measures the typical
   batched-against-single difference (not only the 0.12-nat worst case) and
   the cost of one-at-a-time scoring; the choice is made with those numbers.

Open: the role of the web twin (Alex asked for more explanation).
