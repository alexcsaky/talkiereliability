"""Build the known-truth question set (version 2): outcomes from 1900 to 1929.

Version 1 (``controls/archive/build_known_truth_v1.py``) came out near chance
in stage 0 (AUROC 0.546, and the web twin 0.493) because its decoys were near
misses: Nobel laureates of neighbouring years, cities of similar size. Alex
decided on 23 Nov 2026 to rebuild it with coarser questions: facts a 1930
corpus discusses often, with clear margins, so that a pass or a fail says
whether the reading method works. Families (each a matched pair: A and not-A
differ only in the asserted fact):

* ``census``    a top-10 city against a city ranked 30 to 40, at least 2.5
  times smaller; "larger" against "smaller" with the names fixed in place
  (v2.1, 23 Nov) (U.S. Census Bureau, POP-twps0027, tables 13 to 15).
* ``nobel``     a prize's laureate against a laureate of a distant prize
  (sciences against Literature or Peace) within four years who never won
  this one, e.g. a physicist named for Literature (Nobel Prize API v2.1).
* ``baseball``  the world's series winner against the last-placed team of the
  winner's league that year (Lahman database via a GitHub mirror).
* ``office``    the President of the United States, and the Prime Minister of
  Great Britain, on 1 July of each year 1900 to 1929, against a holder whose
  nearest year in office is at least six years away.
* ``election``  each US presidential election 1900 to 1928, winner against
  runner-up.
* ``events``    the year of a well-known event against a year five to ten years
  off. These use retrospective headers (``template_set: retrospective``), so
  that no header names the year being asserted.
* ``negation``  statement against explicit negation, on office holders and on
  clear-cut census top-10 membership (ranks 1 to 5 against 30 to 40).

The office, election and event tables are entered by hand from the standard
record (``HAND_ENTERED`` below) and should be spot-checked by Alex. Which side
is A is decided by a seeded coin. Writes:

* ``questions/known_truth.jsonl``          statements only, no outcomes
* ``outcomes/known_truth.jsonl``           whether A is true (analysis/ reads it)
* ``questions/known_truth_sources.json``   URLs and SHA-256 of the raw downloads

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
QUESTION_SET = "v2.1"

CENSUS = {1900: "tab13", 1910: "tab14", 1920: "tab15"}
CENSUS_URL = "https://www2.census.gov/library/working-papers/1998/demographics/pop-twps0027/{}.txt"
NOBEL_URL = "https://api.nobelprize.org/2.1/nobelPrizes?nobelPrizeYear=1901&yearTo=1929&limit=500"
LAHMAN_COMMIT = "8a908d51833ee99b56fce165c93dc99ee8886d6f"
LAHMAN = f"https://raw.githubusercontent.com/cbwinslow/baseballdatabank/{LAHMAN_COMMIT}/core/{{}}.csv"

PRIZE = {"phy": "the Nobel Prize for Physics", "che": "the Nobel Prize for Chemistry",
         "med": "the Nobel Prize for Medicine", "lit": "the Nobel Prize for Literature",
         "pea": "the Nobel Peace Prize"}
N_NOBEL = 60
# Decoys come from a distant field: a scientist for Literature or Peace, a writer
# or peace laureate for a science prize.
FIELD = {"phy": "science", "che": "science", "med": "science", "lit": "letters", "pea": "peace"}
CENSUS_MIN_RATIO = 2.5

_ORG_PREFIXES = ("Institute ", "Permanent ", "International ", "Office ", "Friends ")
_KNIGHTED_YEAR = {
    "Sir William Ramsay": 1902,
    "Sir Austen Chamberlain": 1925,
    "Sir Frederick Hopkins": 1925,
}

# ---------------------------------------------------------------------------
# Hand-entered tables (standard record; Alex to spot-check)
# ---------------------------------------------------------------------------

# Holder on 1 July of each year.
US_PRESIDENT = {
    "William McKinley": range(1900, 1902),      # died 14 Sept 1901
    "Theodore Roosevelt": range(1902, 1909),    # to 4 March 1909
    "William H. Taft": range(1909, 1913),
    "Woodrow Wilson": range(1913, 1921),
    "Warren G. Harding": range(1921, 1924),     # died 2 Aug 1923
    "Calvin Coolidge": range(1924, 1929),       # to 4 March 1929
    "Herbert Hoover": range(1929, 1930),
}
UK_PRIME_MINISTER = {
    "Lord Salisbury": [1900, 1901, 1902],               # resigned 11 July 1902
    "A. J. Balfour": [1903, 1904, 1905],                # to 5 Dec 1905
    "Sir Henry Campbell-Bannerman": [1906, 1907],       # to 5 April 1908
    "H. H. Asquith": list(range(1908, 1917)),           # to 5 Dec 1916
    "David Lloyd George": list(range(1917, 1923)),      # to 19 Oct 1922
    "Stanley Baldwin": [1923, 1925, 1926, 1927, 1928],  # from 22 May 1923; Nov 1924 to 4 June 1929
    "J. Ramsay MacDonald": [1924, 1929],                # Jan to Nov 1924; from 5 June 1929
}
# (year, winner, runner-up in the electoral college)
US_ELECTIONS = [
    (1900, "William McKinley", "William J. Bryan"),
    (1904, "Theodore Roosevelt", "Alton B. Parker"),
    (1908, "William H. Taft", "William J. Bryan"),
    (1912, "Woodrow Wilson", "Theodore Roosevelt"),
    (1916, "Woodrow Wilson", "Charles E. Hughes"),
    (1920, "Warren G. Harding", "James M. Cox"),
    (1924, "Calvin Coolidge", "John W. Davis"),
    (1928, "Herbert Hoover", "Alfred E. Smith"),
]
# (year, clause). Clauses are period phrasing and read after "In {year}, ".
EVENTS = [
    (1900, "the foreign legations at Peking were besieged by the Boxers"),
    (1900, "the town of Mafeking was relieved"),
    (1901, "Queen Victoria died"),
    (1901, "Marconi first sent a wireless signal across the Atlantic"),
    (1903, "the Wright brothers made the first flight in a power-driven aeroplane"),
    (1903, "Panama declared its independence of Colombia"),
    (1904, "war broke out between Russia and Japan"),
    (1905, "the Japanese fleet destroyed the Russian fleet at Tsushima"),
    (1906, "an earthquake and fire destroyed much of San Francisco"),
    (1908, "the Olympic Games were held in London"),
    (1908, "the Ford Motor Company brought out its Model T car"),
    (1909, "Commander Peary reached the North Pole"),
    (1910, "King Edward VII died"),
    (1910, "the Union of South Africa was formed"),
    (1911, "King George V was crowned at Westminster"),
    (1911, "Captain Amundsen reached the South Pole"),
    (1912, "the steamship Titanic struck an iceberg and sank"),
    (1912, "the Chinese Republic was proclaimed"),
    (1914, "the Archduke Francis Ferdinand was assassinated at Sarajevo"),
    (1914, "the Panama Canal was opened to traffic"),
    (1915, "the Lusitania was sunk by a German submarine"),
    (1916, "the rising took place in Dublin at Easter"),
    (1917, "the Tsar of Russia abdicated"),
    (1917, "the United States declared war upon Germany"),
    (1918, "the armistice brought the fighting of the Great War to an end"),
    (1919, "the Treaty of Versailles was signed"),
    (1919, "Alcock and Brown made the first flight across the Atlantic without a stop"),
    (1920, "the Eighteenth Amendment, which brought national prohibition, took effect"),
    (1920, "the first Assembly of the League of Nations met at Geneva"),
    (1922, "the Fascists marched on Rome and Mussolini became Prime Minister of Italy"),
    (1922, "the tomb of Tutankhamen was discovered in the Valley of the Kings"),
    (1922, "the Irish Free State was established"),
    (1923, "an earthquake destroyed much of Tokyo and Yokohama"),
    (1924, "Lenin died"),
    (1925, "John T. Scopes was tried at Dayton, Tennessee, for teaching evolution"),
    (1926, "a general strike took place in Great Britain"),
    (1927, "Charles A. Lindbergh flew alone from New York to Paris"),
    (1928, "the Kellogg Pact for the renunciation of war was signed at Paris"),
    (1929, "prices on the New York Stock Exchange collapsed"),
]
HAND_ENTERED = {"US_PRESIDENT": US_PRESIDENT, "UK_PRIME_MINISTER": UK_PRIME_MINISTER,
                "US_ELECTIONS": US_ELECTIONS, "EVENTS": EVENTS}

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


def _item(id_: str, family: str, cluster: str, yes: list[str], no: list[str], fields: dict,
          a_true: bool, template_set: str = "dated") -> tuple[dict, bool]:
    q = {"id": id_, "family": family, "cluster": cluster, "yes": yes, "no": no, "fields": fields}
    if template_set != "dated":
        q["template_set"] = template_set
    return q, a_true


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------

_STATE = {"OR": "Oregon", "ME": "Maine", "MO": "Missouri", "KS": "Kansas", "NJ": "New Jersey",
          "NY": "New York", "MA": "Massachusetts", "PA": "Pennsylvania", "OH": "Ohio"}


def _census_rows(year: int) -> list[tuple[int, str, str, int]]:
    """(rank, place, state, population) for the 40 largest places."""
    text = fetch(CENSUS_URL.format(CENSUS[year]), f"census_{year}.txt").decode("latin-1")
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+(.+?),?\s+([A-Z]{2})\b[ .*]*?([\d,]{5,})", line)
        if not m:
            continue
        place = re.sub(r"\s+(city|town|borough|village|township)\b.*$", "", m.group(2)).strip(" .*")
        rows.append((int(m.group(1)), place, m.group(3), int(m.group(4).replace(",", ""))))
    rows = sorted(r for r in rows if r[0] <= 40)
    assert [r[0] for r in rows] == list(range(1, 41)), f"census {year} parse failed"
    return rows


def _census_labels(rows) -> dict[tuple[str, str], str]:
    names = [r[1] for r in rows]
    return {(p, st): p if names.count(p) == 1 else f"{p}, {_STATE.get(st, st)}"
            for _, p, st, _ in rows}


def census_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    for year in CENSUS:
        rows = _census_rows(year)
        label = _census_labels(rows)
        big, small = rows[:10], rows[29:40]
        pairs = [(b, s) for b in big for s in small if b[3] >= CENSUS_MIN_RATIO * s[3]]
        pairs = rng.sample(pairs, 20)
        for k, (b, s) in enumerate(pairs):
            x, y = label[(b[1], b[2])], label[(s[1], s[2])]
            a_true = rng.random() < 0.5
            A, B = (x, y) if a_true else (y, x)
            # Both statements keep the names in the same places and differ only
            # in the comparative. v2 swapped the names instead and scored 0.245
            # (big cities are the usual yardstick, so "than Chicago" read as
            # likely whatever the truth); see docs/DECISIONS.md, 23 Nov.
            more = [f"At the census of {year}, {A} had a larger population than {B}.",
                    f"The census of {year} showed {A} to be more populous than {B}.",
                    f"By the census of {year}, {A} stood above {B} in population."]
            less = [f"At the census of {year}, {A} had a smaller population than {B}.",
                    f"The census of {year} showed {A} to be less populous than {B}.",
                    f"By the census of {year}, {A} stood below {B} in population."]
            items.append(_item(f"kt2-census-{year}-{k:02d}", "census", f"census-{year}-{k // 5}",
                               more, less, {"year": year, "pub_year": year + 1}, a_true))
    return items


# ---------------------------------------------------------------------------
# Nobel
# ---------------------------------------------------------------------------


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
        en = p["category"]["en"]
        cat = "med" if en.startswith("Physiology") else en[:3].lower()
        year = int(p["awardYear"])
        da = p.get("dateAwarded") or ""
        actual_yr = int(da[:4]) if len(da) >= 4 and da[:4].isdigit() else year
        names = [n for lau in p.get("laureates", [])
                 if (n := (lau.get("knownName") or lau.get("orgName") or {}).get("en"))]
        if names and year <= 1929:
            by_cat.setdefault(cat, {})[year] = names
            award_year[(cat, year)] = actual_yr
    return by_cat, award_year


def nobel_items(rng: random.Random) -> list[tuple[dict, bool]]:
    by_cat, award_year = _load_nobel()
    ever = {cat: {n for ns in years.values() for n in ns} for cat, years in by_cat.items()}
    items = []
    for cat, years in sorted(by_cat.items()):
        if cat not in PRIZE:
            continue
        for year, names in sorted(years.items()):
            pool = sorted({n for other, oy in by_cat.items() if other in PRIZE and FIELD[other] != FIELD[cat]
                           for y, ns in oy.items() if abs(y - year) <= 4
                           for n in ns if n not in ever[cat]})
            if not pool:
                continue
            true_name, decoy = names[0], rng.choice(pool)
            a_true = rng.random() < 0.5
            A, B = (true_name, decoy) if a_true else (decoy, true_name)
            pub_year = min(1930, max(year + 1, award_year[(cat, year)]))
            prize = PRIZE[cat]
            Prize = prize[0].upper() + prize[1:]

            def stmt(raw_name: str) -> list[str]:
                mid, cap = _format_laureate(raw_name, pub_year)
                return [f"{Prize} for the year {year} was awarded to {mid}.",
                        f"{Prize} for {year} went to {mid}.",
                        f"{cap} received {prize} for the year {year}."]

            items.append(_item(f"kt2-nobel-{cat}-{year}", "nobel", f"nobel-{cat}-{(year // 10) * 10}s",
                               stmt(A), stmt(B), {"year": year, "pub_year": pub_year}, a_true))
    keep = set(rng.sample(range(len(items)), min(N_NOBEL, len(items))))
    return [it for i, it in enumerate(items) if i in keep]


# ---------------------------------------------------------------------------
# Baseball
# ---------------------------------------------------------------------------


def _load_baseball() -> list[tuple[int, str, str]]:
    """(year, winner, last-placed team of the winner's league)."""
    series = list(csv.DictReader(io.StringIO(fetch(LAHMAN.format("SeriesPost"),
                                                   "lahman_SeriesPost.csv").decode())))
    teams = list(csv.DictReader(io.StringIO(fetch(LAHMAN.format("Teams"),
                                                  "lahman_Teams.csv").decode())))
    by_key = {(t["yearID"], t["teamID"]): t for t in teams}
    out = []
    for s in series:
        year = int(s["yearID"])
        if s["round"] != "WS" or not 1903 <= year <= 1929:
            continue
        win = by_key[(s["yearID"], s["teamIDwinner"])]
        league = [t for t in teams if t["yearID"] == s["yearID"] and t["lgID"] == win["lgID"]]
        last = max(league, key=lambda t: int(t["Rank"]))
        assert last["teamID"] not in (s["teamIDwinner"], s["teamIDloser"])
        out.append((year, win["name"], last["name"]))
    return out


def baseball_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    for year, w, last in _load_baseball():
        a_true = rng.random() < 0.5
        A, B = (w, last) if a_true else (last, w)
        stmt = lambda X: [  # noqa: E731
            f"The world's series of {year} was won by the {X}.",
            f"The {X} won the world's series of {year}.",
            f"In {year} the baseball championship of the world was won by the {X}."]
        items.append(_item(f"kt2-baseball-{year}", "baseball", f"baseball-{(year // 10) * 10}s",
                           stmt(A), stmt(B), {"year": year, "pub_year": year + 1}, a_true))
    return items


# ---------------------------------------------------------------------------
# Office holders and elections
# ---------------------------------------------------------------------------

OFFICES = {
    "us": ("the President of the United States", "President of the United States", US_PRESIDENT),
    "uk": ("the Prime Minister of Great Britain", "Prime Minister of Great Britain",
           UK_PRIME_MINISTER),
}


def _holder(table, year: int) -> str:
    (name,) = [n for n, ys in table.items() if year in ys]
    return name


def _office_decoy(rng: random.Random, table, year: int) -> str:
    far = sorted(n for n, ys in table.items() if min(abs(y - year) for y in ys) >= 6)
    return rng.choice(far)


def _office_stmt(office: str, bare: str, X: str, year: int, neg: bool = False) -> list[str]:
    was = "was not" if neg else "was"
    held = "did not hold" if neg else "held"
    return [f"On the first day of July, {year}, {office} {was} {X}.",
            f"In the middle of the year {year}, {X} {was} {bare}.",
            f"{X} {held} the office of {bare} in July, {year}."]


def office_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    for key, (office, bare, table) in OFFICES.items():
        for year in range(1900, 1930):
            true, decoy = _holder(table, year), _office_decoy(rng, table, year)
            a_true = rng.random() < 0.5
            A, B = (true, decoy) if a_true else (decoy, true)
            items.append(_item(f"kt2-office-{key}-{year}", "office", f"office-{key}-{(year - 1900) // 5}",
                               _office_stmt(office, bare, A, year), _office_stmt(office, bare, B, year),
                               {"year": year, "pub_year": year + 1}, a_true))
    return items


def election_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    for year, win, runner in US_ELECTIONS:
        a_true = rng.random() < 0.5
        A, B = (win, runner) if a_true else (runner, win)
        stmt = lambda X: [  # noqa: E731
            f"At the presidential election of {year}, {X} was elected President of the United States.",
            f"{X} was chosen President of the United States at the election of {year}.",
            f"The presidential election of {year} was won by {X}."]
        items.append(_item(f"kt2-election-{year}", "election", f"election-{(year - 1900) // 12}",
                           stmt(A), stmt(B), {"year": year, "pub_year": year + 1}, a_true))
    return items


# ---------------------------------------------------------------------------
# Events (retrospective headers: no header names the asserted year)
# ---------------------------------------------------------------------------


def _decoy_year(rng: random.Random, year: int) -> int:
    opts = [y for y in range(1900, 1930) if 5 <= abs(y - year) <= 10]
    return rng.choice(opts)


def event_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    events = list(EVENTS)
    rng.shuffle(events)  # ids and clusters must not follow the true years
    for k, (year, clause) in enumerate(events):
        decoy = _decoy_year(rng, year)
        a_true = rng.random() < 0.5
        A, B = (year, decoy) if a_true else (decoy, year)
        cap = clause[0].upper() + clause[1:]
        stmt = lambda y: [  # noqa: E731
            f"In {y}, {clause}.",
            f"It was in the year {y} that {clause}.",
            f"{cap} in the year {y}."]
        items.append(_item(f"kt2-event-{k:02d}", "events", f"events-{k // 6}",
                           stmt(A), stmt(B), {"year": 1929, "pub_year": 1930}, a_true,
                           template_set="retrospective"))
    return items


# ---------------------------------------------------------------------------
# Negation (statement against explicit negation)
# ---------------------------------------------------------------------------


def negation_items(rng: random.Random) -> list[tuple[dict, bool]]:
    items = []
    # 1. Office holders: true holder (A true) or a far-off holder (A false).
    for key, (office, bare, table) in OFFICES.items():
        for year in range(1900, 1930):
            a_true = rng.random() < 0.5
            X = _holder(table, year) if a_true else _office_decoy(rng, table, year)
            items.append(_item(f"kt2-neg-office-{key}-{year}", "negation",
                               f"neg-office-{key}-{(year - 1900) // 6}",
                               _office_stmt(office, bare, X, year),
                               _office_stmt(office, bare, X, year, neg=True),
                               {"year": year, "pub_year": year + 1}, a_true))
    # 2. Census top-10 membership: ranks 1 to 5 (true) against 30 to 40 (false).
    for year in CENSUS:
        rows = _census_rows(year)
        label = _census_labels(rows)
        chosen = [(r, True) for r in rows[:5]] + [(r, False) for r in rng.sample(rows[29:40], 5)]
        rng.shuffle(chosen)
        for k, (r, is_top10) in enumerate(chosen):
            city = label[(r[1], r[2])]
            yes = [f"At the census of {year}, {city} stood among the ten most populous cities of the United States.",
                   f"The census of {year} placed {city} within the ten largest American cities.",
                   f"By the census of {year}, {city} ranked in the first ten cities of the United States in population."]
            no = [f"At the census of {year}, {city} did not stand among the ten most populous cities of the United States.",
                  f"The census of {year} did not place {city} within the ten largest American cities.",
                  f"By the census of {year}, {city} did not rank in the first ten cities of the United States in population."]
            items.append(_item(f"kt2-neg-census-{year}-{k:02d}", "negation", f"neg-census-{year}",
                               yes, no, {"year": year, "pub_year": year + 1}, is_top10))
    return items


def build(rng: random.Random) -> list[tuple[dict, bool]]:
    return (census_items(rng) + nobel_items(rng) + baseball_items(rng) + office_items(rng)
            + election_items(rng) + event_items(rng) + negation_items(rng))


def main() -> None:
    items = build(random.Random(SEED))
    qpath = REPO / "questions" / "known_truth.jsonl"
    opath = REPO / "outcomes" / "known_truth.jsonl"
    opath.parent.mkdir(exist_ok=True)
    with qpath.open("w") as fq, opath.open("w") as fo:
        for q, a_true in items:
            fq.write(json.dumps(q, ensure_ascii=False) + "\n")
            fo.write(json.dumps({"question_id": q["id"], "outcome": int(a_true)}) + "\n")
    (REPO / "questions" / "known_truth_sources.json").write_text(
        json.dumps({"question_set": QUESTION_SET, "seed": SEED, "sources": _sources,
                    "hand_entered": {k: {n: list(v) for n, v in t.items()} if isinstance(t, dict)
                                     else t for k, t in HAND_ENTERED.items()}},
                   indent=2) + "\n")
    fams: dict[str, list[bool]] = {}
    for q, a in items:
        fams.setdefault(q["family"], []).append(a)
    for f, v in fams.items():
        print(f"{f}: {len(v)} items, A true in {sum(v)}, "
              f"{len({q['cluster'] for q, _ in items if q['family'] == f})} clusters")


if __name__ == "__main__":
    main()
