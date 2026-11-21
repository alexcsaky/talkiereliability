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


def test_transcriber_credit_stripped():
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK MAIN STREET ***\r\n"
           "Produced by Charles Keller and the Online\r\n"
           "Distributed Proofreading Team.\r\n\r\nMAIN STREET\r\n\r\n"
           "By Sinclair Lewis\r\n\r\nThe play was produced by Mr. Belasco.")
    a = admit(raw, kind="passage", origin="credit", pub_date="1920")
    assert "Keller" not in a.text and "Proofreading" not in a.text
    assert a.text.startswith("MAIN STREET") and "produced by Mr. Belasco" in a.text


def test_future_years_rules():
    # 1930 is on or before the cutoff and must NOT be treated as a future year.
    assert firewall.future_years("Published in 1930.") == []
    h1930 = admit("NEW YORK, DECEMBER 31, 1930\n\n", kind="header", origin="h1930")
    assert not h1930.flags
    # Questions may name the target decade.
    admit("Will Ohio go Democratic in 1932?", kind="question", origin="q")
    # Templates and headers may not name 1931+, unless allowed explicitly.
    with pytest.raises(FirewallRejection):
        admit("THE ALMANAC FOR 1935", kind="header", origin="h")
    with pytest.raises(FirewallRejection):
        admit("THE ALMANAC FOR 1931", kind="header", origin="h1931")
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
    assert firewall.parse_date("1930-12-31T23:59:59Z") == firewall.CUTOFF
    with pytest.raises(FirewallRejection):
        admit("x", kind="passage", origin="d", pub_date="1931")
    # Bound-volume date ranges and malformed dates must be rejected, never truncated.
    for bad_date in ("1930-1935", "1930/1940", "1930-12-31 / 1931-01-01", "1930-13-45"):
        with pytest.raises(ValueError):
            firewall.parse_date(bad_date)
        with pytest.raises(FirewallRejection):
            admit("General summary.", kind="passage", origin="range", pub_date=bad_date)


def test_anachronisms_caught_before_stripping_and_modern_copyright_rejected():
    # Anachronism inside a URL must reject rather than being silently stripped away.
    with pytest.raises(FirewallRejection):
        admit("See https://en.wikipedia.org/wiki/World_War_II for notes.",
              kind="passage", origin="url-leak", pub_date="1925")
    # Post-1930 copyright / reprint notices must reject passages.
    with pytest.raises(FirewallRejection):
        admit("Copyright 1998. Reprinted from an earlier edition.",
              kind="passage", origin="copy-leak", pub_date="1925")
    # Lexicon gap closures and removed false positives.
    for leak_txt in ("The United Nations met in New York.", "During Great Depression, trade fell.",
                     "The Wehrmacht crossed the border.", "Chancellor Hitler addressed the assembly."):
        with pytest.raises(FirewallRejection):
            admit(leak_txt, kind="passage", origin="lex-gap", pub_date="1928")
    # Genuine pre-1931 usages of Superman (Shaw/Nietzsche) and Holocaust (fires/massacres).
    admit("Mr. Bernard Shaw's play Man and Superman was revived at the Royal Court.",
          kind="passage", origin="shaw", pub_date="1921")
    admit("Holocaust in New York factory claims one hundred and forty lives.",
          kind="passage", origin="fire", pub_date="1911")


def test_log_written(tmp_path, monkeypatch):
    log = tmp_path / "fw.jsonl"
    admit("A plain sentence.", kind="header", origin="o1")
    with pytest.raises(FirewallRejection):
        admit("World War II", kind="header", origin="o2")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["decision"] for r in rows] == ["pass", "reject"]
    assert "text" not in rows[0]


def test_leak_probe_exemption_is_narrow_and_compose_preserves_flags(tmp_path):
    a = admit(" Manchukuo", kind="continuation", origin="leak:x", leak_probe=True)
    assert a.text == " Manchukuo" and a.leak_probe is True and "leak-probe exemption" in a.flags
    with pytest.raises(ValueError):
        admit("Manchukuo", kind="passage", origin="p", pub_date="1929", leak_probe=True)
    with pytest.raises(FirewallRejection):
        admit(" Manchukuo", kind="continuation", origin="c")
    b = admit("x <|user|> y", kind="probe", origin="leak:y", leak_probe=True)
    assert "<|" not in b.text and b.leak_probe is True

    p = admit("Bankers expect recovery by 1934.", kind="passage", origin="p1", pub_date="1929-10-30")
    h = admit("NEW YORK, DECEMBER 31, 1930\n\n", kind="header", origin="h1")
    comp = firewall.compose([p, h], kind="header", origin="comp1", sep="\n\n")
    assert comp.leak_probe is False and "1934" in comp.text
    with pytest.raises(TypeError):
        firewall.compose([p, "raw"], kind="header", origin="bad")  # type: ignore[list-item]
