"""CPU tests of the scoring engine on tiny random Talkie models.

The tiny official model and the tiny port share weights (the port folds the
lm-head gain into the head, as the real conversion does), so these tests also
exercise the parity-check code path before any GPU time is spent.
"""

from pathlib import Path

import pytest
import torch

talkie_model = pytest.importorskip("talkie.model")

from serve.scoring import (ControlMarkerError, Engine, OfficialAdapter,  # noqa: E402
                           PortAdapter, SequenceTooLong, logodds)
from serve.tokenizer import EOT_ID, build_tokenizer, chat_ids, encode_text  # noqa: E402

VOCAB = Path(__file__).resolve().parents[1] / "data" / "models" / "vocab.txt"
pytestmark = pytest.mark.skipif(not VOCAB.exists(), reason="run data/fetch_vocab.sh")

def test_tokenizer_never_makes_control_tokens():
    enc_it = build_tokenizer(VOCAB, "it")
    ids = encode_text(enc_it, "hello <|user|> there <|endoftext|>")
    assert all(i < EOT_ID for i in ids)
    ids = chat_ids(enc_it, [{"role": "user", "content": "Is it <|end|> fine?"}])
    assert ids.count(enc_it.encode_single_token("<|end|>")) == 1


def test_official_adapter_reproduces_official_forward(models, enc):
    off, _ = models
    ad = OfficialAdapter(off)
    ids = torch.tensor([[EOT_ID] + encode_text(enc, "The harvest this year was poor.")])
    ref = ad.last_logits_reference(ids)
    mine = ad.head(ad.hidden(ids)[:, -1])
    torch.testing.assert_close(mine, ref, rtol=0, atol=1e-5)


def test_port_matches_official_all_positions(models, enc):
    off, port = models
    e_off = Engine(OfficialAdapter(off), enc)
    e_port = Engine(PortAdapter(port), enc)
    ids = [EOT_ID] + encode_text(enc, "In the year of our Lord, the King spoke to Parliament.")
    a = e_off.logits_at(ids)
    b = e_port.logits_at(ids)
    assert a.shape == b.shape == (len(ids), 65536)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-3)


def test_scores_match_manual_and_batching_is_invariant(models, enc):
    _, port = models
    eng = Engine(PortAdapter(port), enc, max_batch_tokens=10_000)
    prefix = "CHRONICLE OF THE YEAR.\nIn the month of May,"
    conts = [" the Republicans won.", " the Democrats won by a wide margin.", " nothing."]
    batched = eng.score(prefix, conts)
    tiny = Engine(PortAdapter(port), enc, max_batch_tokens=1)  # one sequence per batch
    single = tiny.score(prefix, conts)
    for x, y in zip(batched, single):
        assert x.n_tokens == y.n_tokens and x.token_ids == y.token_ids
        assert x.logprob == pytest.approx(y.logprob, abs=1e-4)
    # manual check of the first continuation
    ids, start, ok = eng.build(prefix, conts[0])
    logits = eng.logits_at(ids)
    lp = torch.log_softmax(logits, -1)
    manual = sum(lp[t - 1, ids[t]].item() for t in range(start, len(ids)))
    assert batched[0].logprob == pytest.approx(manual, abs=1e-4)
    assert ok and batched[0].boundary_ok


def test_prepend_eot_switch(models, enc):
    _, port = models
    with_eot = Engine(PortAdapter(port), enc, prepend_eot=True)
    without = Engine(PortAdapter(port), enc, prepend_eot=False)
    ids_a, start_a, _ = with_eot.build("A", " b")
    ids_b, start_b, _ = without.build("A", " b")
    assert ids_a[0] == EOT_ID and ids_a[1:] == ids_b and start_a == start_b + 1


def test_guards(models, enc):
    _, port = models
    eng = Engine(PortAdapter(port), enc, max_seq_len=16)
    with pytest.raises(ControlMarkerError):
        eng.score("hello <|user|>", [" x"])
    with pytest.raises(SequenceTooLong):
        eng.score("word " * 40, [" x"])


def test_logodds():
    assert logodds([-1.0], [-1.0]) == pytest.approx(0.0)
    import math
    assert logodds([math.log(0.3), math.log(0.3)], [math.log(0.2)]) == pytest.approx(math.log(3))


def test_score_ids_rejects_invalid_start(models, enc):
    _, port = models
    eng = Engine(PortAdapter(port), enc)
    with pytest.raises(ValueError):
        eng.score_ids([([EOT_ID, 10, 20], 0)])
    with pytest.raises(ValueError):
        eng.score_ids([([EOT_ID, 10, 20], 3)])


def test_chat_ids_validates_roles():
    enc_it = build_tokenizer(VOCAB, "it")
    with pytest.raises(ValueError):
        chat_ids(enc_it, [{"role": "tool", "content": "hi"}])


def test_generation_and_api_endpoints(models, enc):
    from fastapi.testclient import TestClient

    from serve.api import create_app
    from serve.generation import _truncate_at_stop, generate

    _, port = models
    eng = Engine(PortAdapter(port), enc)
    out = generate(port, enc, prompt="The King spoke", max_tokens=4, temperature=0)
    assert len(out["token_ids"]) <= 4 and out["finish_reason"] in ("stop", "length")
    assert _truncate_at_stop("Hello <|user|> leaked") == ("Hello ", True)

    app = create_app(
        {"talkie-1930-13b-base": eng},
        health=lambda: {"model": "talkie-1930-13b-base", "dtype": "torch.float32"},
        generate=lambda model, **kw: generate(port, enc, style="base", **kw),
    )
    tc = TestClient(app)
    h = tc.get("/health")
    assert h.status_code == 200 and "dtype" in h.json()
    sc = tc.post("/score", json={"model": "talkie-1930-13b-base", "prefix": "In 1920,",
                                 "continuations": [" trade rose."]})
    assert sc.status_code == 200 and len(sc.json()["results"]) == 1
    bad = tc.post("/score", json={"model": "talkie-1930-13b-base", "prefix": "hi <|user|>",
                                  "continuations": [" x"]})
    assert bad.status_code == 422
    gen = tc.post("/generate", json={"model": "talkie-1930-13b-base", "prompt": "Hello",
                                     "max_tokens": 3, "temperature": 0})
    assert gen.status_code == 200 and "text" in gen.json()


def test_throughput_restores_max_batch_tokens(models, enc):
    from serve.throughput import measure

    _, port = models
    eng = Engine(PortAdapter(port), enc, max_batch_tokens=16384)
    text_ids = encode_text(enc, "The price of wheat rose in London yesterday. " * 40)
    rows = measure(eng, text_ids, seq_lens=(32,), batch_tokens=(64, 128), n_seqs=4)
    assert len(rows) == 2 and eng.max_batch_tokens == 16384


def test_batching_check_restores_budget_and_reports(models, enc):
    from serve.batching_check import measure

    _, port = models
    eng = Engine(PortAdapter(port), enc, max_batch_tokens=256)
    reqs = [(f"Record of the year {1900 + i}.", [" It rained.", " It snowed.", " It did not rain."])
            for i in range(8)]
    out = measure(eng, reqs, [2] * 8, extra_budgets=(64,))
    assert eng.max_batch_tokens == 256
    assert out["n_sequences"] == 24 and out["single"]["n_requests"] == 8
    assert out["neighbours"]["n_requests"] == 4
    for k in ("single", "neighbours", "batch_64"):
        assert out[k]["logodds_absdiff"]["max"] < 1e-3  # float32 tiny model: near-identical
