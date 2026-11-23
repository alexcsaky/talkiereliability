"""Known-truth set v2 (23 Nov 2026): structure checks that need no network."""

import json
import random
import re
from pathlib import Path

import yaml

from controls import build_known_truth as b
from controls import known_truth

REPO = Path(__file__).resolve().parents[1]


def _questions():
    return [json.loads(x) for x in (REPO / "questions" / "known_truth.jsonl").read_text().splitlines()]


def test_one_office_holder_per_year():
    for table in (b.US_PRESIDENT, b.UK_PRIME_MINISTER):
        for year in range(1900, 1930):
            assert sum(year in ys for ys in table.values()) == 1, year


def test_office_decoys_are_far_off():
    rng = random.Random(0)
    for table in (b.US_PRESIDENT, b.UK_PRIME_MINISTER):
        for year in range(1900, 1930):
            d = b._office_decoy(rng, table, year)
            assert min(abs(y - year) for y in table[d]) >= 6


def test_event_decoy_years_in_range():
    rng = random.Random(0)
    for year, _ in b.EVENTS:
        d = b._decoy_year(rng, year)
        assert 1900 <= d <= 1929 and 5 <= abs(d - year) <= 10


def test_retrospective_headers_name_no_event_year():
    y = yaml.safe_load((REPO / "questions" / "templates" / "known_truth.yaml").read_text())
    for t in y["retrospective_templates"]:
        years = {int(x) for x in re.findall(r"\b(1[89]\d\d)\b", t["header"])}
        assert years <= {1900, 1930}, t["id"]
        assert "{" not in t["header"], t["id"]


def test_events_use_retrospective_headers_and_others_dated():
    _, _, which = known_truth.load_sets()
    for q in _questions():
        want = "retrospective" if q["family"] == "events" else "dated"
        assert which[q["id"]] == want, q["id"]


def test_ids_and_clusters_do_not_reveal_answers():
    """questions/ holds no outcomes: no id or cluster may name a year or person
    that appears in only one of the two statements."""
    for q in _questions():
        yes, no = " ".join(q["yes"]), " ".join(q["no"])
        for tag in (q["id"], q["cluster"]):
            for tok in re.findall(r"[A-Za-z]{4,}|\d{4}", tag):
                assert (tok in yes) == (tok in no), (q["id"], tok)


def test_pairs_differ_and_are_balanced():
    qs = _questions()
    assert all(q["yes"] != q["no"] for q in qs)
    assert len({q["id"] for q in qs}) == len(qs)
    outs = [json.loads(x)["outcome"] for x in (REPO / "outcomes" / "known_truth.jsonl").read_text().splitlines()]
    assert len(outs) == len(qs) and 0.4 < sum(outs) / len(outs) < 0.6
