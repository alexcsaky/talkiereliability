# Leakage

Every leak check, its result, and known leaks. Results are added as they are
measured; nothing below has been run on the real model yet.

## Known leaks (before any measurement)

- **Talkie's authors:** the 13B model knows some details of the Second World
  War, the United Nations and the division of Germany, from misdated documents
  and later editorial insertions (verified, Talkie post, per the brief).
- **Chat model:** post-trained with Claude Sonnet 4.6 as judge and on chats with
  Claude Opus 4.6 (verified, per the brief). It is used only in the librarian
  mode and the demo.
- **Users** report it giving the year as 1939 (reported, repo issues, per the
  plan).
- **The authors' own anachronism filter** let through text about the New Deal
  and the war (verified, Talkie post, per the plan).

## Checks built

| Check | Where | Stage | Status |
| --- | --- | --- | --- |
| Firewall: date, modern matter, control markers, lexicon | `firewall.py` | all | Built; 22 of 22 planted post-1930 passages rejected, 10 of 10 genuine period passages passed (`tests/test_firewall.py`) |
| Choice probes: post-1930 facts against decoys, with pre-1931 controls | `controls/leak_probes.yaml` | 0 | Built; not yet run on Talkie |
| Year probes: "what year is it" under five period headers | `controls/leak_probes.yaml` | 0 | Built; not yet run |
| Community probe set (1930s multiple choice, 1920s retention reference) | `controls/leak_probes.py` | 0, optional | Built; not yet run |
| Closed-book probe of every target question | `leak_probes.probe_questions` | 2 | Built; needs the pilot questions |
| Ledger canary claims | `ledger/canaries.py` | 1 | Not built |
| Librarian canary requests | `librarian/guard.py` | 4 | Not built |
| 1921 to 1930 mirror (out-of-sample skill close to in-sample is suspicious) | `questions/mirror/` | 2 | Not built |

## Results

None yet. Stage 0 writes `results/stage0/leak_base.json`, `leak_it.json`,
`leak_it_chat.json` and `leak_web.json`; `python -m analysis.stage0_report`
summarises them (with both summed and per-token length-normalised ranks), and
the summary is copied here.
