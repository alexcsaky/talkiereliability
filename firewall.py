"""The firewall: the single gate for every string bound for Talkie.

Rule 1 of the build brief: nothing dated after 31 Dec 1930 reaches Talkie.
Every string sent to the model (passages, prompt templates, question
statements, continuations, probes) goes through ``admit()`` here. The client
in ``forecaster/client.py`` refuses to send anything that has not been
admitted, and the server additionally encodes text with special tokens
disabled, so chat markers can never become control tokens.

What ``admit()`` does, in order:

1. For passages: require a publication date and reject anything after
   ``CUTOFF``.
2. Strip modern matter: Project Gutenberg wrappers, URLs, library
   boilerplate, Wikidata ids (``clean_text``).
3. Strip chat control markers such as ``<|user|>`` and any stray ``<|`` or
   ``|>`` (``strip_control_markers``).
4. Run the anachronism lexicon (``anachronism_hits``). Any hit rejects.
5. Note mentions of years after 1930. For passages these are flagged but
   allowed (a 1929 editorial that mentions 1935 is a forecast, which is what
   the ledger wants); for templates and probes they reject unless the caller
   says future years are expected (question statements about 1931 to 1940).
6. Log what was passed: a SHA-256 of the admitted text, its origin and the
   reasons, to ``results/firewall_log.jsonl``.

The lexicon lives in ``firewall_lexicon.yaml``. It is deliberately narrow: it
lists terms that did not exist before 1931, not words that merely became more
common later. Leak detection in the model itself is ``controls/leak_probes.py``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import yaml

CUTOFF = dt.date(1930, 12, 31)
REPO = Path(__file__).resolve().parent
LEXICON_PATH = REPO / "firewall_lexicon.yaml"
DEFAULT_LOG = REPO / "results" / "firewall_log.jsonl"

Kind = Literal["passage", "template", "question", "continuation", "probe", "header"]
KINDS: tuple[str, ...] = ("passage", "template", "question", "continuation", "probe", "header")

# Chat markers used by the Talkie chat model, plus the document separator.
CONTROL_MARKERS = ("<|endoftext|>", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>")
_MARKER_RE = re.compile(r"<\|[^<>|]{0,40}\|>")
_STRAY_RE = re.compile(r"<\||\|>")

# Four-digit years 1931 to 2099, not part of a longer number.
_FUTURE_YEAR_RE = re.compile(r"(?<![\d,.])(193[1-9]|19[4-9]\d|20\d\d)(?![\d])")

# Post-1930 copyright / modern edition notices inside body text.
_MODERN_COPYRIGHT_RE = re.compile(
    r"(?i)(?:\bcopyright\b|©|\(c\)|\breprinted\b)\s*(?:©|\(c\))?\s*(?:193[1-9]|19[4-9]\d|20\d\d)\b"
)

_DATE_FULL_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_DATE_MONTH_RE = re.compile(r"^(\d{4}-\d{2})$")
_DATE_YEAR_RE = re.compile(r"^(\d{4})$")

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)

_log_lock = threading.Lock()


class FirewallRejection(ValueError):
    """Raised when a string may not be sent to Talkie."""

    def __init__(self, reasons: list[str], origin: str):
        self.reasons = reasons
        self.origin = origin
        super().__init__(f"firewall rejected {origin!r}: {'; '.join(reasons)}")


@dataclass(frozen=True)
class Admitted:
    """A string that has passed the firewall. Only these may reach the model."""

    text: str
    kind: str
    origin: str
    sha256: str
    flags: tuple[str, ...] = field(default_factory=tuple)
    leak_probe: bool = False

    def __str__(self) -> str:  # so templates can be formatted naturally
        return self.text


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lexicon:
    patterns: tuple[tuple[str, re.Pattern], ...]
    strip: tuple[re.Pattern, ...]


@lru_cache(maxsize=4)
def load_lexicon(path: str | None = None) -> _Lexicon:
    data = yaml.safe_load(Path(path or LEXICON_PATH).read_text())
    pats = []
    for term in data.get("terms", []):
        text = " ".join(term["text"].split())
        flags = 0 if term.get("case", "insensitive") == "sensitive" else re.I
        body = r"\s+".join(re.escape(w) for w in text.split(" "))
        pats.append((text, re.compile(rf"(?<![\w-]){body}(?![\w-])", flags)))
    strip = tuple(re.compile(p) for p in data.get("strip_patterns", []))
    return _Lexicon(tuple(pats), strip)


# ---------------------------------------------------------------------------
# Pure checks (usable on their own, e.g. by archive/filters.py)
# ---------------------------------------------------------------------------


def strip_control_markers(text: str) -> tuple[str, int]:
    """Remove chat control markers and stray marker halves. Returns (text, count)."""
    n = 0
    text, k = _MARKER_RE.subn(" ", text)
    n += k
    text, k = _STRAY_RE.subn(" ", text)
    n += k
    return text, n


def strip_gutenberg(text: str) -> str:
    """Drop the Project Gutenberg header and licence, keeping the book text."""
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[: m.start()]
    return text


def clean_text(text: str, lexicon: _Lexicon | None = None) -> str:
    """Strip modern matter and normalise whitespace (keeps paragraph breaks)."""
    lex = lexicon or load_lexicon()
    text = strip_gutenberg(text)
    for pat in lex.strip:
        text = pat.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def anachronism_hits(text: str, lexicon: _Lexicon | None = None) -> list[str]:
    """Lexicon terms found in ``text``."""
    lex = lexicon or load_lexicon()
    flat = " ".join(text.split())
    return [term for term, pat in lex.patterns if pat.search(flat)]


def future_years(text: str) -> list[int]:
    """Years after 1930 mentioned in ``text``."""
    return sorted({int(y) for y in _FUTURE_YEAR_RE.findall(text)})


def parse_date(value: str | dt.date | dt.datetime | None) -> dt.date | None:
    """Parse a single publication date. Rejects ranges (e.g. bound volumes) and malformed values."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    m = _DATE_FULL_RE.match(s)
    if m:
        return dt.datetime.strptime(m.group(1), "%Y-%m-%d").date()
    m = _DATE_MONTH_RE.match(s)
    if m:
        d = dt.datetime.strptime(m.group(1), "%Y-%m").date()
        nxt = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        return nxt - dt.timedelta(days=1)
    m = _DATE_YEAR_RE.match(s)
    if m:
        d = dt.datetime.strptime(m.group(1), "%Y").date()
        return d.replace(month=12, day=31)
    raise ValueError(f"unparseable date {value!r}")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def admit(
    text: str,
    *,
    kind: Kind,
    origin: str,
    pub_date: str | dt.date | None = None,
    allow_future_years: bool | None = None,
    leak_probe: bool = False,
    log: bool | Path = True,
) -> Admitted:
    """Check and clean one string bound for Talkie, or raise FirewallRejection.

    ``origin`` names where the string came from (a passage id, a template id,
    a question id). ``pub_date`` is required for passages. Future years are
    allowed by default only for questions and continuations, which talk about
    1931 to 1940 by design; for passages they are flagged, not rejected (unless
    paired with a post-1930 copyright/reprint notice).

    ``leak_probe=True`` is the one exemption: leak probes measure how likely
    Talkie finds post-1930 words such as "Manchukuo", so those strings must
    reach the model. Scoring is stateless (nothing carries from one call to
    the next), so a probe cannot inform any forecast. The exemption skips the
    lexicon and future-year checks, still strips control markers, applies only
    to kinds "probe" and "continuation", marks ``Admitted.leak_probe=True``,
    and is logged on every use.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    if leak_probe and kind not in ("probe", "continuation"):
        raise ValueError("leak_probe exemption applies only to probes and their continuations")
    reasons: list[str] = []
    flags: list[str] = ["leak-probe exemption"] if leak_probe else []

    if kind == "passage":
        try:
            d = parse_date(pub_date)
        except ValueError as e:
            reasons.append(str(e))
            d = None
        else:
            if d is None:
                reasons.append("passage has no publication date")
            elif d > CUTOFF:
                reasons.append(f"published {d.isoformat()}, after {CUTOFF.isoformat()}")

    lex = load_lexicon()
    # Scan the Gutenberg-stripped body BEFORE stripping URLs/boilerplate so
    # anachronisms embedded in modern URLs or footnotes are caught, not hidden.
    pre_strip = strip_gutenberg(text)
    pre_strip_norm = pre_strip.replace("_", " ")
    cleaned = clean_text(text, lex)
    cleaned, n_markers = strip_control_markers(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned).strip()
    if cleaned:
        # Keep edge whitespace: it decides how a prefix and a continuation
        # join, so scoring depends on it. Normalised to one space or <=2 newlines.
        cleaned = _edge_ws(text, lead=True) + cleaned + _edge_ws(text, lead=False)
    if n_markers:
        flags.append(f"stripped {n_markers} control marker(s)")

    hits = sorted(set(anachronism_hits(pre_strip_norm, lex)) | set(anachronism_hits(cleaned, lex)))
    if hits and leak_probe:
        flags.append("anachronism allowed for leak probe: " + ", ".join(hits))
    elif hits:
        reasons.append("anachronism: " + ", ".join(hits))

    if not leak_probe and _MODERN_COPYRIGHT_RE.search(pre_strip):
        reasons.append("post-1930 copyright or reprint notice")

    years = future_years(cleaned)
    if years and leak_probe:
        flags.append("mentions years after 1930: " + ", ".join(map(str, years)))
    elif years:
        if allow_future_years is None:
            allow_future_years = kind in ("question", "continuation", "passage")
        if allow_future_years:
            flags.append("mentions years after 1930: " + ", ".join(map(str, years)))
        else:
            reasons.append("mentions years after 1930: " + ", ".join(map(str, years)))

    if not cleaned:
        reasons.append("empty after cleaning")

    if reasons:
        _write_log(log, {"decision": "reject", "kind": kind, "origin": origin,
                         "reasons": reasons, "flags": flags,
                         "sha256": _sha(text)})
        raise FirewallRejection(reasons, origin)

    adm = Admitted(text=cleaned, kind=kind, origin=origin, sha256=_sha(cleaned),
                   flags=tuple(flags), leak_probe=leak_probe)
    _write_log(log, {"decision": "pass", "kind": kind, "origin": origin,
                     "flags": flags, "leak_probe": leak_probe,
                     "sha256": adm.sha256, "chars": len(cleaned),
                     **({"text": cleaned} if os.environ.get("TALKIE_FIREWALL_LOG_TEXT") else {})})
    return adm


def compose(
    parts: Iterable[Admitted],
    *,
    kind: Kind,
    origin: str,
    sep: str = "",
    log: bool | Path = True,
) -> Admitted:
    """Safely join already-admitted strings (e.g. a forecast passage and a header).

    Preserves each part's original admission policy (so a pre-1931 passage that
    mentions 1935 is not rejected by ``header``'s stricter year rule), while
    refusing raw strings, rejecting any control markers formed across joins,
    and propagating ``leak_probe`` if any part used the exemption.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    seq = list(parts)
    if not seq:
        raise ValueError("compose requires at least one Admitted part")
    for p in seq:
        if not isinstance(p, Admitted):
            raise TypeError(f"compose accepts only Admitted parts, got {type(p).__name__}")
    joined = sep.join(p.text for p in seq)
    if "<|" in joined or "|>" in joined:
        raise FirewallRejection(["control marker formed across composed boundary"], origin)
    leak_probe = any(p.leak_probe for p in seq)
    flags = tuple(dict.fromkeys(f for p in seq for f in p.flags))
    adm = Admitted(text=joined, kind=kind, origin=origin, sha256=_sha(joined),
                   flags=flags, leak_probe=leak_probe)
    _write_log(log, {"decision": "pass", "kind": kind, "origin": origin,
                     "composed_from": [p.origin for p in seq], "flags": list(flags),
                     "leak_probe": leak_probe, "sha256": adm.sha256, "chars": len(joined)})
    return adm


def admit_many(items: Iterable[dict], **common) -> tuple[list[Admitted], list[FirewallRejection]]:
    """Admit a batch; returns (admitted, rejections) instead of raising."""
    ok, bad = [], []
    for item in items:
        try:
            ok.append(admit(**{**common, **item}))
        except FirewallRejection as e:
            bad.append(e)
    return ok, bad


def _edge_ws(text: str, *, lead: bool) -> str:
    m = re.match(r"^\s*", text) if lead else re.search(r"\s*$", text)
    ws = m.group(0) if m else ""
    if not ws:
        return ""
    n = ws.count("\n")
    return "\n" * min(n, 2) if n else " "


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_log(log: bool | Path, record: dict) -> None:
    if log is False:
        return
    path = Path(os.environ.get("TALKIE_FIREWALL_LOG", DEFAULT_LOG)) if log is True else Path(log)
    record = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
