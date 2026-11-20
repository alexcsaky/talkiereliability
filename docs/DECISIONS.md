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
provisional probes (29 choice probes, 6 of them pre-1931 controls; 5 year
probes) and the community probe set; `leak_probes.probe_questions` runs on the
pilot questions at the start of stage 2.
