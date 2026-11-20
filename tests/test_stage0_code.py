"""Run the stage 0 GPU jobs' code on tiny CPU models, to catch bugs cheaply."""

import pytest

from serve import context_test, parity_check
from serve.scoring import Engine, OfficialAdapter, PortAdapter

FILLER = ("The farmers of the valley gathered at the market on Saturday, and the talk "
          "was all of the weather and the price of wheat. ") * 200


def test_retrieval_and_loss_run(models, enc):
    _, port = models
    eng = Engine(PortAdapter(port), enc)
    res = context_test.retrieval(eng, [FILLER], trials_per_cell=1, gate=lambda t: t)
    assert set(res["by_total"]) == set(context_test.TOTALS)
    assert all(r["n_tokens"] <= 4096 for r in res["rows"])
    lv = context_test.loss_vs_context(eng, [FILLER])
    assert len(lv["by_total"]) == 4
    pos = context_test.loss_by_position(eng, [FILLER])
    assert len(pos) == 8


def test_parity_runs_and_agrees_on_tiny_models(models, enc):
    off, port = models
    prompts = ["The King opened Parliament today.", FILLER[:3000]]
    res = parity_check.run_parity(lambda: (off, enc), lambda: (port, enc), prompts)
    assert res["adapter_vs_official_forward_max_abs"] < 1e-4
    assert res["max_abs_logit_diff"] < 1e-2
    assert res["max_abs_logit_diff_noeot"] < 1e-2
    assert res["batch_right_pad_max_abs_tok_logprob_diff"] < 1e-4
    assert res["min_top1_agree"] > 0.99
