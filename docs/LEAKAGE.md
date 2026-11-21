# Leakage

Every leak check, its result, and known leaks. Results are added as they are
measured. Stage 0 probes were run on 21 Nov 2026.

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
| Firewall: date, modern matter, control markers, lexicon | `firewall.py` | all | Built; 23 of 23 planted post-1930 passages rejected, 10 of 10 genuine period passages passed (`tests/test_firewall.py`). 21 Nov: "Library of America" added and transcriber credits stripped after both were found in cached Gutenberg texts |
| Choice probes: post-1930 facts against decoys, with pre-1931 controls | `controls/leak_probes.yaml` | 0 | Run 21 Nov (results below) |
| Year probes: "what year is it" under five period headers | `controls/leak_probes.yaml` | 0 | Run 21 Nov (results below) |
| Community probe set (1930s multiple choice, 1920s retention reference) | `controls/leak_probes.py` | 0, optional | Built; not yet run |
| Closed-book probe of every target question | `leak_probes.probe_questions` | 2 | Built; needs the pilot questions |
| Ledger canary claims | `ledger/canaries.py` | 1 | Not built |
| Librarian canary requests | `librarian/guard.py` | 4 | Not built |
| 1921 to 1930 mirror (out-of-sample skill close to in-sample is suspicious) | `questions/mirror/` | 2 | Not built |

## Results

### Stage 0, 21 Nov 2026 (verified: measured)

Full tables in `results/stage0/report.md`. Choice probes give the model's
share of probability on the true option; chance for "ranked first" is about
3.8 of 21.

| Model | Controls first | Post-1930 probes first | Mean log(share / uniform) | Mass on 1931+ as "present year" |
| --- | --- | --- | --- | --- |
| talkie-1930-13b-base | 6 of 6 | 14 of 21 | +0.48 | 0.03 to 0.19 (mode 1920) |
| talkie-1930-13b-it, raw | 6 of 6 | 12 of 21 | -0.07 | 0.03 to 0.27 |
| talkie-1930-13b-it, chat template | 5 of 6 | 10 of 21 | -0.75 | 0.01 to 0.14 |
| talkie-web-13b-base (ceiling) | 6 of 6 | 20 of 21 | +1.56 | 0.54 to 0.77 (mode 1933 to 1944) |

Base-model hits, with share of probability on the truth: 1932 election 0.99,
Manchukuo 1.00, New Deal 0.97, Gestapo 0.93, Hitler as Chancellor 0.86, Roosevelt's
third term 0.79, Rhineland 0.75, United Nations 0.70, Olympics in Berlin 0.65,
war in 1939 0.65, Hiroshima 0.61, coronation 1937 0.55, war's end in 1945
0.30, repeal of Prohibition 0.17. Misses: Guernica, Molotov-Ribbentrop, Munich,
the West German capital, the month of the Spanish Republic, and the month
Britain left gold.

Reading (hypothesis): some hits come from plausibility, not knowledge.
Roosevelt and Hitler were prominent by 1930, "Gestapo" and "Manchukuo" are the
most word-like options, and "new deal" was an idiom. Without decoys of matched
plausibility the probes cannot separate the two. The year of the war (1939
over seven other years), the United Nations and Hiroshima are harder to
explain that way. They agree with the leaks the authors report (Second World
War, United Nations). The stage 2 closed-book probe of every target question is
therefore essential, not optional.
