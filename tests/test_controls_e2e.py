"""End to end on tiny CPU models: firewall -> client -> engine -> report."""

import json

import pytest

from controls import known_truth, leak_probes
from forecaster.client import EngineBackend, NotAdmitted, TalkieClient
from serve.scoring import Engine, PortAdapter


@pytest.fixture
def client(models, enc, tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_FIREWALL_LOG", str(tmp_path / "fw.jsonl"))
    _, port = models
    return TalkieClient(EngineBackend({"talkie-1930-13b-base": Engine(PortAdapter(port), enc)}))


def test_client_refuses_unadmitted(client):
    with pytest.raises(NotAdmitted):
        client.score([("raw prefix", [" raw continuation"])])


def test_known_truth_runs(client):
    res = known_truth.run(client, limit=6)
    assert res["n_questions"] == 6 and not res["rejected_by_firewall"]
    assert all(r["boundary_ok"] for r in res["per_template"])
    assert len(res["per_template"]) == 6 * 4


def test_leak_probes_run(client):
    res = leak_probes.choice_and_year(client)
    assert not res["rejected_by_firewall"]
    kinds = {r["kind"] for r in res["rows"]}
    assert kinds == {"choice", "year"}


def test_stage0_report_builds(client, tmp_path, monkeypatch):
    from analysis import stage0_report
    monkeypatch.setattr(stage0_report, "RES", tmp_path)
    (tmp_path / "known_truth_base.json").write_text(json.dumps(known_truth.run(client, limit=40)))
    (tmp_path / "leak_base.json").write_text(json.dumps(leak_probes.choice_and_year(client)))
    stage0_report.main()
    text = (tmp_path / "report.md").read_text()
    assert "Known-truth check" in text and "Leak probes" in text
