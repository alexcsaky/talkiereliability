# Talkie reliability backtest

Does a forecaster forecast better when told the track record of its sources?
This repo tests that claim on Talkie, a 13B language model trained only on
English text from before 1931, asked about resolved questions from 1931 to
1940, with evidence from a pre-1931 archive and a reliability ledger built
from forecasts that resolved by the end of 1930.

- Method: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)
- Engineering brief: [docs/BUILD_BRIEF.md](docs/BUILD_BRIEF.md)
- Where things stand: [docs/STATUS.md](docs/STATUS.md)
- For coding agents: [AGENTS.md](AGENTS.md)

Status: stage 0 (model checks) code built and tested on CPU; not yet run on a GPU.
