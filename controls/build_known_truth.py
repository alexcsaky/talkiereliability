"""Build the known-truth question set: outcomes from 1900 to 1929, by rule.

Families (each item is a matched pair, so A and not-A differ only in which
name comes where):

* ``census``  pairs of the 40 largest US places at the 1900, 1910 and 1920
  censuses (U.S. Census Bureau, POP-twps0027, tables 13 to 15).
* ``nobel``   the laureate of each Nobel prize 1901 to 1929 against a laureate
  of the same prize from a nearby year (Nobel Prize API v2.1).
* ``baseball`` world's series 1903 to 1929, winner against loser (Lahman
  database via a GitHub mirror; checked by hand against the record for a
  sample of years).

Which side is A is decided by a seeded coin, so about half of the items have
a true A. Writes:

* ``questions/known_truth.jsonl``   statements only, no outcomes
* ``outcomes/known_truth.jsonl``    whether A is true (analysis/ reads this)
* ``questions/known_truth_sources.json`` URLs and SHA-256 of the raw downloads

Run: ``python -m controls.build_known_truth``. Needs network.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "known_truth"
SEED = 19301231

CENSUS = {1900: "tab13", 1910: "tab14", 1920: "tab15"}
CENSUS_URL = "https://www2.census.gov/library/working-papers/1998/demographics/pop-twps0027/{}.txt"
NOBEL_URL = "https://api.nobelprize.org/2.1/nobelPrizes?nobelPrizeYear=1901&yearTo=1929&limit=500"
LAHMAN = "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/{}.csv"

PRIZE = {"phy": "the Nobel Prize for Physics", "che": "the Nobel Prize for Chemistry",
         "med": "the Nobel Prize for Medicine", "lit": "the Nobel Prize for Literature",
         "pea": "the Nobel Peace Prize"}

_sources: dict[str, dict] = {}


def fetch(url: str, name: str) -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    data = path.read_bytes()
    _sources[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return data


# ---------------------------------------------------------------------------


def _census_rows(year: int) -> list[tuple[int, str, str]]:
    text = fetch(CENSUS_URL.format(CENSUS[year]), f"census_{year}.txt").decode("latin-1")
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+(.+?)\s(?:city|town|borough|village)?,?\s*([A-Z]{2})\b", line)
        m = re.match(r"\s*(\d+)\s+(.+?),\s*([A-Z]{2})\b", line)
        if not m:
            continue
        rank, place, state = int(m.group(1)), m.group(2), m.group(3)
        place = re.sub(r"\s+(city|town|borough|village|township)\b.*$", "", place).strip(" .*")
        rows.append((rank, place, state))
    rows.sort()
    return rows


_STATE = {"OR": "Oregon", "ME": "Maine", "MO": "Missouri", "KS": "Kansas", "NJ": "New Jersey",
          "NY": "New York", "MA": "Massachusetts", "PA": "Pennsylvania", "OH": "Ohio"}


def census_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    for year in CENSUS:
        rows = _census_rows(year)[:40]
        assert len(rows) == 40 and rows[0][0] == 1, f"census {year} parse failed"
        names = [p for _, p, _ in rows]
        label = {}
        for _, p, st in rows:
            label[p] = p if names.count(p) == 1 else f"{p}, {_STATE.get(st, st)}"
        pairs, used = [], set()
        while len(pairs) < 25:
            gap = rng.choice([1, 2, 3, 4, 6, 10])
            i = rng.randrange(0, 40 - gap)
            j = i + gap
            if (i, j) in used:
                continue
            used.add((i, j))
            pairs.append((rows[i][1], rows[j][1]))
        for k, (big, small) in enumerate(pairs):
            x, y = label[big], label[small]
            a_true = rng.random() < 0.5
            A, B = (x, y) if a_true else (y, x)
            stmt = lambda P, Q: [  # noqa: E731
                f"At the census of {year}, {P} had a larger population than {Q}.",
                f"The census of {year} showed {P} to be more populous than {Q}.",
                f"By the census of {year}, {P} stood above {Q} in population."]
            items.append(({"id": f"kt-census-{year}-{k:02d}", "family": "census",
                           "cluster": f"census-{year}", "yes": stmt(A, B), "no": stmt(B, A),
                           "fields": {"year": year, "pub_year": year + 1}}, a_true))
    return items


def nobel_items(rng: random.Random) -> list[tuple[dict, bool]]:
    data = json.loads(fetch(NOBEL_URL, "nobel_1901_1929.json"))
    by_cat: dict[str, dict[int, list[str]]] = {}
    for p in data["nobelPrizes"]:
        cat = p["category"]["en"][:3].lower()
        cat = {"phy": "phy", "che": "che", "phys": "phy", "lit": "lit", "pea": "pea"}.get(cat, cat)
        if p["category"]["en"].startswith("Physiology"):
            cat = "med"
        year = int(p["awardYear"])
        names = []
        for lau in p.get("laureates", []):
            n = (lau.get("knownName") or lau.get("orgName") or {}).get("en")
            if n:
                names.append(n)
        if names and year <= 1929:
            by_cat.setdefault(cat, {})[year] = names
    items = []
    for cat, years in sorted(by_cat.items()):
        if cat not in PRIZE:
            continue
        for year, names in sorted(years.items()):
            near = [y for y in years if y != year and abs(y - year) <= 4]
            pool = [n for y in near for n in years[y] if n not in names]
            if not pool:
                continue
            true_name, decoy = names[0], rng.choice(sorted(set(pool)))
            a_true = rng.random() < 0.5
            A, B = (true_name, decoy) if a_true else (decoy, true_name)
            prize = PRIZE[cat]
            stmt = lambda X: [  # noqa: E731
                f"{prize[0].upper() + prize[1:]} for the year {year} was awarded to {X}.",
                f"{prize[0].upper() + prize[1:]} for {year} went to {X}.",
                f"{X} received {prize} for the year {year}."]
            items.append(({"id": f"kt-nobel-{cat}-{year}", "family": "nobel",
                           "cluster": f"nobel-{cat}", "yes": stmt(A), "no": stmt(B),
                           "fields": {"year": year, "pub_year": year + 1}}, a_true))
    return items


def baseball_items(rng: random.Random) -> list[tuple[dict, bool]]:
    series = list(csv.DictReader(io.StringIO(fetch(LAHMAN.format("SeriesPost"),
                                                   "lahman_SeriesPost.csv").decode())))
    teams = list(csv.DictReader(io.StringIO(fetch(LAHMAN.format("Teams"),
                                                  "lahman_Teams.csv").decode())))
    name = {(t["yearID"], t["teamID"]): t["name"] for t in teams}
    items = []
    for s in series:
        year = int(s["yearID"])
        if s["round"] != "WS" or not 1903 <= year <= 1929:
            continue
        w, l = name[(s["yearID"], s["teamIDwinner"])], name[(s["yearID"], s["teamIDloser"])]
        a_true = rng.random() < 0.5
        A, B = (w, l) if a_true else (l, w)
        stmt = lambda X, Y: [  # noqa: E731
            f"The world's series of {year} was won by the {X}, who defeated the {Y}.",
            f"In the world's series of {year} the {X} beat the {Y}.",
            f"The {X} won the baseball championship of the world in {year}, defeating the {Y}."]
        items.append(({"id": f"kt-baseball-{year}", "family": "baseball", "cluster": "baseball",
                       "yes": stmt(A, B), "no": stmt(B, A),
                       "fields": {"year": year, "pub_year": year + 1}}, a_true))
    return items


def main() -> None:
    rng = random.Random(SEED)
    items = census_items(rng) + nobel_items(rng) + baseball_items(rng)
    qpath = REPO / "questions" / "known_truth.jsonl"
    opath = REPO / "outcomes" / "known_truth.jsonl"
    opath.parent.mkdir(exist_ok=True)
    with qpath.open("w") as fq, opath.open("w") as fo:
        for q, a_true in items:
            fq.write(json.dumps(q, ensure_ascii=False) + "\n")
            fo.write(json.dumps({"question_id": q["id"], "outcome": int(a_true)}) + "\n")
    (REPO / "questions" / "known_truth_sources.json").write_text(
        json.dumps({"seed": SEED, "sources": _sources}, indent=2) + "\n")
    fams = {}
    for q, a in items:
        fams.setdefault(q["family"], []).append(a)
    for f, v in fams.items():
        print(f"{f}: {len(v)} items, A true in {sum(v)}")


if __name__ == "__main__":
    main()
