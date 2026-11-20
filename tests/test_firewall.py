import json
from pathlib import Path

import pytest

import firewall
from firewall import FirewallRejection, admit

DATA = Path(__file__).parent / "data" / "planted_passages.jsonl"


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKIE_FIREWALL_LOG", str(tmp_path / "fw.jsonl"))


def _planted():
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def test_planted_post1930_passages_all_rejected():
    planted = [p for p in _planted() if p["expect"] == "reject"]
    assert len(planted) >= 20
    passed = []
    for p in planted:
        try:
            admit(p["text"], kind="passage", origin=p["id"], pub_date=p["pub_date"])
            passed.append(p["id"])
        except FirewallRejection:
            pass
    assert passed == [], f"firewall let through: {passed}"  # gate: 100 per cent


def test_genuine_period_passages_pass():
    genuine = [p for p in _planted() if p["expect"] == "pass"]
    rejected = {}
    for p in genuine:
        try:
            admit(p["text"], kind="passage", origin=p["id"], pub_date=p["pub_date"])
        except FirewallRejection as e:
            rejected[p["id"]] = e.reasons
    assert rejected == {}


def test_control_markers_stripped():
    a = admit("Some text <|user|> more <|assistant|>and <|end|> |> <| end",
              kind="passage", origin="t", pub_date="1920-01-01")
    for m in firewall.CONTROL_MARKERS:
        assert m not in a.text
    assert "<|" not in a.text and "|>" not in a.text
    assert any("control marker" in f for f in a.flags)


def test_gutenberg_and_urls_stripped():
    raw = ("Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
           "Chapter one. It was a dark night. See https://example.com/x\n"
           "*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\nLicence text")
    a = admit(raw, kind="passage", origin="g", pub_date="1910")
    assert "Header junk" not in a.text and "Licence" not in a.text
    assert "https" not in a.text and "dark night" in a.text


def test_future_years_rules():
    # Questions may name the target decade.
    admit("Will Ohio go Democratic in 1932?", kind="question", origin="q")
    # Templates and headers may not, unless allowed explicitly.
    with pytest.raises(FirewallRejection):
        admit("THE ALMANAC FOR 1935", kind="header", origin="h")
    admit("In the year 1934 the President was", kind="probe", origin="p",
          allow_future_years=True)
    # Passages: flagged, not rejected.
    a = admit("Experts expect recovery by 1932.", kind="passage", origin="x",
              pub_date="1929-11-01")
    assert any("1932" in f for f in a.flags)


def test_edge_whitespace_preserved():
    assert admit(" Albert Einstein.", kind="continuation", origin="c").text == " Albert Einstein."
    assert admit("Header line\n\n", kind="header", origin="h").text == "Header line\n\n"


def test_dates():
    assert firewall.parse_date("1930") == firewall.CUTOFF
    assert firewall.parse_date("1930-12") == firewall.CUTOFF
    with pytest.raises(FirewallRejection):
        admit("x", kind="passage", origin="d", pub_date="1931")


def test_log_written(tmp_path, monkeypatch):
    log = tmp_path / "fw.jsonl"
    admit("A plain sentence.", kind="header", origin="o1")
    with pytest.raises(FirewallRejection):
        admit("World War II", kind="header", origin="o2")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["decision"] for r in rows] == ["pass", "reject"]
    assert "text" not in rows[0]


def test_leak_probe_exemption_is_narrow(tmp_path):
    a = admit(" Manchukuo", kind="continuation", origin="leak:x", leak_probe=True)
    assert a.text == " Manchukuo" and "leak-probe exemption" in a.flags
    with pytest.raises(ValueError):
        admit("Manchukuo", kind="passage", origin="p", pub_date="1929", leak_probe=True)
    with pytest.raises(FirewallRejection):
        admit(" Manchukuo", kind="continuation", origin="c")
    b = admit("x <|user|> y", kind="probe", origin="leak:y", leak_probe=True)
    assert "<|" not in b.text
