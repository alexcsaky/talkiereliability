"""End to end on tiny CPU models: firewall -> client -> engine -> report."""

import json

import pytest

import firewall
from controls import known_truth, leak_probes
from forecaster import elicit
from forecaster.client import EngineBackend, NotAdmitted, TalkieClient
from serve.scoring import Engine, PortAdapter


@pytest.fixture
def client(models, enc, tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_FIREWALL_LOG", str(tmp_path / "fw.jsonl"))
    _, port = models
    return TalkieClient(EngineBackend({"talkie-1930-13b-base": Engine(PortAdapter(port), enc)}))


@pytest.fixture
def probe_client(models, enc, tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_FIREWALL_LOG", str(tmp_path / "fw.jsonl"))
    _, port = models
    return TalkieClient(EngineBackend({"talkie-1930-13b-base": Engine(PortAdapter(port), enc)}),
                        allow_leak_probes=True)


def test_client_refuses_unadmitted_and_confirmatory_refuses_leak_probe(client):
    with pytest.raises(NotAdmitted):
        client.score([("raw prefix", [" raw continuation"])])
    pre = firewall.admit("In 1932 Japan set up", kind="probe", origin="lp", leak_probe=True)
    opt = firewall.admit(" Manchukuo", kind="continuation", origin="lp:0", leak_probe=True)
    with pytest.raises(NotAdmitted):
        client.score([(pre, [opt])])


def test_all_known_truth_questions_pass_firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_FIREWALL_LOG", str(tmp_path / "fw.jsonl"))
    qs, ts = known_truth.load()
    reqs, rejected = elicit.build_requests(qs, ts)
    assert rejected == []
    assert len(reqs) == len(qs) * len(ts)
    assert {q.family for q in qs} == {"census", "nobel", "baseball", "negation"}


def test_build_requests_passage_rules_and_atomic_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_FIREWALL_LOG", str(tmp_path / "fw.jsonl"))
    q = elicit.BinaryQuestion(id="q1", family="test", yes=("A happened.",), no=("A did not happen.",),
                              fields={"year": 1930})
    t_ok = elicit.Template("t0", "DISPATCH, {year}.\n\n")
    t_bad = elicit.Template("t1", "DISPATCH, 1935.\n\n")
    # Raw passage string must be refused (Rule 1).
    with pytest.raises(TypeError):
        elicit.build_requests([q], [t_ok], passage="raw text")  # type: ignore[arg-type]
    # Admitted pre-1931 forecast passage mentioning 1935 must succeed!
    p = firewall.admit("Bankers forecast recovery by 1935.", kind="passage",
                       origin="p-1929", pub_date="1929-11-01")
    reqs, rej = elicit.build_requests([q], [t_ok], passage=p)
    assert len(reqs) == 1 and rej == [] and "1935" in reqs[0].prefix.text
    # Partial template failure must drop the entire question (no orphan t0 row).
    reqs2, rej2 = elicit.build_requests([q], [t_ok, t_bad])
    assert reqs2 == [] and len(rej2) == 1


def test_known_truth_runs(client):
    res = known_truth.run(client, limit=6)
    assert res["n_questions"] == 6 and not res["rejected_by_firewall"]
    assert all(r["boundary_ok"] for r in res["per_template"])
    assert len(res["per_template"]) == 6 * 4


def test_leak_probes_run_and_chat_template_supported(probe_client, models, enc, tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_FIREWALL_LOG", str(tmp_path / "fw.jsonl"))
    res = leak_probes.choice_and_year(probe_client)
    assert not res["rejected_by_firewall"]
    assert all(r["boundary_ok"] for r in res["rows"])
    kinds = {r["kind"] for r in res["rows"]}
    assert kinds == {"choice", "year"}


def test_stage0_report_builds_all_sections_without_truncation(client, probe_client, tmp_path, monkeypatch):
    from analysis import stage0_report
    monkeypatch.setattr(stage0_report, "RES", tmp_path)
    (tmp_path / "download.json").write_text(json.dumps({
        "talkie-1930-13b-base": {"vocab_sha1_ok": True},
        "official:talkie-1930-13b-base": {"sha256_ok": True},
        "context_texts": [{"ebook": 543, "ok": True}],
    }))
    (tmp_path / "health_base.json").write_text(json.dumps({
        "model": "talkie-1930-13b-base", "repo": "xlr8harder/talkie-1930-13b-base-tf",
        "revision": "7cf020702ef7", "dtype": "torch.bfloat16", "device": "L40S",
        "load_seconds": 12.3, "max_seq_len": 4096,
    }))
    (tmp_path / "throughput.json").write_text(json.dumps({
        "rows": [{"seq_len": 256, "max_batch_tokens": 8192, "seqs": 64,
                  "tokens_per_sec": 4321, "seconds": 3.8, "peak_mem_gib": 28.1}],
        "tokens_per_1000_chars_gutenberg": 234.5,
    }))
    (tmp_path / "context.json").write_text(json.dumps({
        "loss_vs_context": {
            "by_total": {"1024": {"mean_loss": 2.5, "ppl": 12.18, "n_texts": 5},
                         "4096": {"mean_loss": 2.4, "ppl": 11.02, "n_texts": 5}},
            "rows": [{"text": i, "total_tokens": 1024, "context_tokens": 767, "loss": 2.5} for i in range(20)],
        },
        "loss_by_position": {"0-512": 2.6, "3584-4096": 2.4},
        "retrieval": {
            "by_total": {"4096": {"accuracy": 0.9, "chose_distractor": 0.05, "n": 30,
                                  "by_depth": {"0.1": 0.9, "0.5": 0.9, "0.9": 0.9}}},
            "chance": 0.25, "seconds": 10.0,
            "rows": [{"total_tokens": 4096, "trial": i} for i in range(120)],
        },
    }))
    (tmp_path / "known_truth_base.json").write_text(json.dumps(known_truth.run(client, limit=40)))
    lk = leak_probes.choice_and_year(probe_client)
    # Inject an extreme logprob gap to verify log_softmax avoids math domain error.
    lk["rows"][0]["logprobs"][0] = -1000.0
    (tmp_path / "leak_base.json").write_text(json.dumps(lk))
    stage0_report.main()
    text = (tmp_path / "report.md").read_text()
    assert "Known-truth check" in text and "Leak probes" in text
    assert "4321" in text  # throughput rows preserved!
    assert "Needle retrieval" in text and "0.900" in text  # context retrieval not truncated!
    assert "sha256_ok=True" in text and "torch.bfloat16" in text
