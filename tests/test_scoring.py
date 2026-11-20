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
