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
LAHMAN_COMMIT = "8a908d51833ee99b56fce165c93dc99ee8886d6f"
LAHMAN = f"https://raw.githubusercontent.com/cbwinslow/baseballdatabank/{LAHMAN_COMMIT}/core/{{}}.csv"

PRIZE = {"phy": "the Nobel Prize for Physics", "che": "the Nobel Prize for Chemistry",
         "med": "the Nobel Prize for Medicine", "lit": "the Nobel Prize for Literature",
         "pea": "the Nobel Peace Prize"}

_ORG_PREFIXES = ("Institute ", "Permanent ", "International ", "Office ", "Friends ")
_KNIGHTED_YEAR = {
    "Sir William Ramsay": 1902,
    "Sir Austen Chamberlain": 1925,
    "Sir Frederick Hopkins": 1925,
}

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


def _census_labels(rows: list[tuple[int, str, str]]) -> dict[tuple[str, str], str]:
    names = [p for _, p, _ in rows]
    return {
        (p, st): p if names.count(p) == 1 else f"{p}, {_STATE.get(st, st)}"
        for _, p, st in rows
    }


def census_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    for year in CENSUS:
        rows = _census_rows(year)[:40]
        assert len(rows) == 40 and rows[0][0] == 1, f"census {year} parse failed"
        label = _census_labels(rows)
        pairs, used = [], set()
        while len(pairs) < 25:
            gap = rng.choice([1, 2, 3, 4, 6, 10])
            i = rng.randrange(0, 40 - gap)
            j = i + gap
            if (i, j) in used:
                continue
            used.add((i, j))
            pairs.append(((rows[i][1], rows[i][2]), (rows[j][1], rows[j][2])))
        for k, (big, small) in enumerate(pairs):
            x, y = label[big], label[small]
            a_true = rng.random() < 0.5
            A, B = (x, y) if a_true else (y, x)
            stmt = lambda P, Q: [  # noqa: E731
                f"At the census of {year}, {P} had a larger population than {Q}.",
                f"The census of {year} showed {P} to be more populous than {Q}.",
                f"By the census of {year}, {P} stood above {Q} in population."]
            items.append(({"id": f"kt-census-{year}-{k:02d}", "family": "census",
                           "cluster": f"census-{year}-{k // 9}", "yes": stmt(A, B), "no": stmt(B, A),
                           "fields": {"year": year, "pub_year": year + 1}}, a_true))
    return items


def _format_laureate(name: str, pub_year: int) -> tuple[str, str]:
    """Return (mid-sentence form, sentence-start form), fixing articles and knighthood dates."""
    if name in _KNIGHTED_YEAR and pub_year < _KNIGHTED_YEAR[name]:
        name = name.removeprefix("Sir ")
    if name.startswith(_ORG_PREFIXES):
        return f"the {name}", f"The {name}"
    return name, name


def _load_nobel() -> tuple[dict[str, dict[int, list[str]]], dict[tuple[str, int], int]]:
    data = json.loads(fetch(NOBEL_URL, "nobel_1901_1929.json"))
    by_cat: dict[str, dict[int, list[str]]] = {}
    award_year: dict[tuple[str, int], int] = {}
    for p in data["nobelPrizes"]:
        cat = p["category"]["en"][:3].lower()
        cat = {"phy": "phy", "che": "che", "phys": "phy", "lit": "lit", "pea": "pea"}.get(cat, cat)
        if p["category"]["en"].startswith("Physiology"):
            cat = "med"
        year = int(p["awardYear"])
        da = p.get("dateAwarded") or ""
        actual_yr = int(da[:4]) if len(da) >= 4 and da[:4].isdigit() else year
        names = []
        for lau in p.get("laureates", []):
            n = (lau.get("knownName") or lau.get("orgName") or {}).get("en")
            if n:
                names.append(n)
        if names and year <= 1929:
            by_cat.setdefault(cat, {})[year] = names
            award_year[(cat, year)] = actual_yr
    return by_cat, award_year


def nobel_items(rng: random.Random) -> list[tuple[dict, bool]]:
    by_cat, award_year = _load_nobel()
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
            pub_year = min(1930, max(year + 1, award_year[(cat, year)]))
            prize = PRIZE[cat]

            def stmt(raw_name: str) -> list[str]:
                mid, cap = _format_laureate(raw_name, pub_year)
                return [
                    f"{prize[0].upper() + prize[1:]} for the year {year} was awarded to {mid}.",
                    f"{prize[0].upper() + prize[1:]} for {year} went to {mid}.",
                    f"{cap} received {prize} for the year {year}.",
                ]

            decade = (year // 10) * 10
            items.append(({"id": f"kt-nobel-{cat}-{year}", "family": "nobel",
                           "cluster": f"nobel-{cat}-{decade}s", "yes": stmt(A), "no": stmt(B),
                           "fields": {"year": year, "pub_year": pub_year}}, a_true))
    return items


def _load_baseball() -> list[tuple[int, str, str]]:
    series = list(csv.DictReader(io.StringIO(fetch(LAHMAN.format("SeriesPost"),
                                                   "lahman_SeriesPost.csv").decode())))
    teams = list(csv.DictReader(io.StringIO(fetch(LAHMAN.format("Teams"),
                                                  "lahman_Teams.csv").decode())))
    name = {(t["yearID"], t["teamID"]): t["name"] for t in teams}
    out = []
    for s in series:
        year = int(s["yearID"])
        if s["round"] != "WS" or not 1903 <= year <= 1929:
            continue
        out.append((year, name[(s["yearID"], s["teamIDwinner"])], name[(s["yearID"], s["teamIDloser"])]))
    return out


def baseball_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    for year, w, l in _load_baseball():
        a_true = rng.random() < 0.5
        A, B = (w, l) if a_true else (l, w)
        stmt = lambda X, Y: [  # noqa: E731
            f"The world's series of {year} was won by the {X}, who defeated the {Y}.",
            f"In the world's series of {year} the {X} beat the {Y}.",
            f"The {X} won the baseball championship of the world in {year}, defeating the {Y}."]
        decade = (year // 10) * 10
        items.append(({"id": f"kt-baseball-{year}", "family": "baseball",
                       "cluster": f"baseball-{decade}s",
                       "yes": stmt(A, B), "no": stmt(B, A),
                       "fields": {"year": year, "pub_year": year + 1}}, a_true))
    return items


def negation_items(rng: random.Random) -> list[tuple[dict, bool]]:
    """Rule-generated statement-vs-negation items from the same authoritative datasets.

    Validates that the reading method distinguishes affirmative statements from
    their explicit negations (the structure used by the 1931 to 1940 forecast
    families), not only symmetric entity swaps.
    """
    items: list[tuple[dict, bool]] = []

    # 1. Census top-10 membership (10 true + 10 false per census year = 60 items).
    for year in CENSUS:
        rows = _census_rows(year)[:40]
        label = _census_labels(rows)
        top10 = [(rows[i][1], rows[i][2]) for i in range(10)]
        below = rng.sample([(rows[i][1], rows[i][2]) for i in range(15, 40)], 10)
        chosen = [(c, True) for c in top10] + [(c, False) for c in below]
        rng.shuffle(chosen)
        for k, (key, is_top10) in enumerate(chosen):
            city = label[key]
            yes = [
                f"At the census of {year}, {city} stood among the ten most populous cities of the United States.",
                f"The census of {year} placed {city} within the ten largest American cities.",
                f"By the census of {year}, {city} ranked in the first ten cities of the United States in population.",
            ]
            no = [
                f"At the census of {year}, {city} did not stand among the ten most populous cities of the United States.",
                f"The census of {year} did not place {city} within the ten largest American cities.",
                f"By the census of {year}, {city} did not rank in the first ten cities of the United States in population.",
            ]
            items.append(({"id": f"kt-neg-census-{year}-{k:02d}", "family": "negation",
                           "cluster": f"neg-census-{year}-{k // 10}",
                           "yes": yes, "no": no,
                           "fields": {"year": year, "pub_year": year + 1}}, is_top10))

    # 2. World Series champion statement vs negation (26 items).
    for year, w, l in _load_baseball():
        a_true = rng.random() < 0.5
        team = w if a_true else l
        yes = [
            f"The {team} won the world's series of {year}.",
            f"In {year}, the {team} captured the baseball championship of the world.",
            f"The {team} were victorious in the world's series of {year}.",
        ]
        no = [
            f"The {team} did not win the world's series of {year}.",
            f"In {year}, the {team} failed to capture the baseball championship of the world.",
            f"The {team} were not victorious in the world's series of {year}.",
        ]
        decade = (year // 10) * 10
        items.append(({"id": f"kt-neg-baseball-{year}", "family": "negation",
                       "cluster": f"neg-baseball-{decade}s",
                       "yes": yes, "no": no,
                       "fields": {"year": year, "pub_year": year + 1}}, a_true))

    return items


def main() -> None:
    rng = random.Random(SEED)
    items = census_items(rng) + nobel_items(rng) + baseball_items(rng) + negation_items(rng)
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
