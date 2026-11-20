"""Pinned model registry. Every model is loaded from an exact commit.

Commit hashes and file hashes read from the Hugging Face API on 20 Nov 2026
(verified). The official repos were last changed on 23 Apr 2026; the ports on
9 Jul 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PortSpec:
    name: str
    repo: str
    revision: str
    style: Literal["base", "it"]
    vocab_sha1: str  # git blob id of vocab.txt, as reported by the hub
    role: str


@dataclass(frozen=True)
class OfficialSpec:
    name: str
    repo: str
    revision: str
    checkpoint: str
    checkpoint_sha256: str
    style: Literal["base", "it"]


PORTS: dict[str, PortSpec] = {
    "talkie-1930-13b-base": PortSpec(
        "talkie-1930-13b-base", "xlr8harder/talkie-1930-13b-base-tf",
        "7cf020702ef7fd68ac16c85aaea389206ded519e", "base",
        "c3e00302c06ba088e28124c84a85c565ea31a3f5",
        "forecaster for every confirmatory arm"),
    "talkie-1930-13b-it": PortSpec(
        "talkie-1930-13b-it", "xlr8harder/talkie-1930-13b-it-tf",
        "f858792d565ab60b38516890553f64960ebef8a7", "it",
        "c3e00302c06ba088e28124c84a85c565ea31a3f5",
        "librarian mode and demo only (post-trained with modern Claude models)"),
    "talkie-web-13b-base": PortSpec(
        "talkie-web-13b-base", "xlr8harder/talkie-web-13b-base-tf",
        "963f2bba0c3fceacb69f57f671015ca3eb162808", "base",
        "880d814e96e90fe057798993433fde8a45f86b64",  # different vocabulary
        "ceiling: modern twin, closed book"),
}

OFFICIAL: dict[str, OfficialSpec] = {
    "talkie-1930-13b-base": OfficialSpec(
        "talkie-1930-13b-base", "talkie-lm/talkie-1930-13b-base",
        "b7c97680791f7fca4262c3c80b36ff7d666faab0", "final.ckpt",
        "caa3c1691dd8cdd2cbae9167437aaa6630144f7303f75883658698f2061b6f9f", "base"),
    "talkie-1930-13b-it": OfficialSpec(
        "talkie-1930-13b-it", "talkie-lm/talkie-1930-13b-it",
        "8033675be6360ae0127fa75f941c12d52064f1dc", "rl-refined.pt",
        "c6fd903b67ad312c7537095fa982b0e77735ccb689c8cfc774ca492694e70064", "it"),
    "talkie-web-13b-base": OfficialSpec(
        "talkie-web-13b-base", "talkie-lm/talkie-web-13b-base",
        "1e5b771c9d38d44f54d35e722c5c0d73da418dd8", "base.ckpt",
        "a082abde819ce2861b466e3dea628233b666c27c85e6ad00f4601a91473da3f9", "base"),
}

# Official inference code, for the parity check.
OFFICIAL_CODE_REPO = "https://github.com/talkie-lm/talkie"
OFFICIAL_CODE_COMMIT = "35317ba3a84861a84c84065bd73faf88ad19329c"  # 19 May 2026

MAX_SEQ_LEN = 4096
EOT_ID = 65535
