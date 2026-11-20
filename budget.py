"""Spend guard: record GPU and API spend, refuse jobs that would cross a cap.

Every paid job declares its worst case before it starts. For a Modal job the
worst case is its timeout times the hourly price of the container (GPU plus
CPU plus memory), so a job that hangs still cannot overspend. When the job
ends, the actual cost (wall time times the same price) replaces the
reservation.

The ledger is ``results/budget_ledger.jsonl`` and is committed, so spend is
tracked across sessions and machines. Caps come from ``budget.yaml``.

Modal prices were read from https://modal.com/pricing on 20 Nov 2026
(verified): L40S $0.000542/s, A100-40GB $0.000583/s, A100-80GB $0.000694/s,
H100 $0.001097/s, CPU $0.0000131 per physical core per second, memory
$0.00000222 per GiB per second. Region selection costs 1.15 to 1.75 times
more (reported), so jobs do not pin a region.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import yaml

REPO = Path(__file__).resolve().parent
CONFIG_PATH = REPO / "budget.yaml"
DEFAULT_LEDGER = REPO / "results" / "budget_ledger.jsonl"

Kind = Literal["gpu", "api"]

GPU_USD_PER_SEC = {
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    None: 0.0,
}
CPU_USD_PER_CORE_SEC = 0.0000131
MEM_USD_PER_GIB_SEC = 0.00000222


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Container:
    """What a Modal container costs per second."""

    gpu: str | None = "L40S"
    cpu_cores: float = 4.0
    memory_gib: float = 64.0

    @property
    def usd_per_sec(self) -> float:
        return (GPU_USD_PER_SEC[self.gpu]
                + self.cpu_cores * CPU_USD_PER_CORE_SEC
                + self.memory_gib * MEM_USD_PER_GIB_SEC)

    @property
    def usd_per_hour(self) -> float:
        return self.usd_per_sec * 3600


def load_caps(path: Path | None = None) -> dict[str, float]:
    data = yaml.safe_load(Path(path or CONFIG_PATH).read_text())
    return {"gpu": float(data["caps_usd"]["gpu"]), "api": float(data["caps_usd"]["api"])}


def _ledger_path() -> Path:
    return Path(os.environ.get("TALKIE_BUDGET_LEDGER", DEFAULT_LEDGER))


def _read() -> list[dict]:
    p = _ledger_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _append(rec: dict) -> None:
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **rec}
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def spent(kind: Kind) -> float:
    """Committed spend plus open reservations for ``kind``, in USD."""
    total = 0.0
    open_res: dict[str, float] = {}
    for r in _read():
        if r["kind"] != kind:
            continue
        if r["event"] == "reserve":
            open_res[r["job_id"]] = r["usd"]
        elif r["event"] == "settle":
            open_res.pop(r["job_id"], None)
            total += r["usd"]
        elif r["event"] == "manual":
            total += r["usd"]
    return total + sum(open_res.values())


def remaining(kind: Kind) -> float:
    return load_caps()[kind] - spent(kind)


def reserve(name: str, kind: Kind, worst_case_usd: float, note: str = "") -> str:
    """Reserve ``worst_case_usd``; raise BudgetExceeded if the cap would be crossed."""
    cap = load_caps()[kind]
    already = spent(kind)
    if already + worst_case_usd > cap + 1e-9:
        raise BudgetExceeded(
            f"{name}: worst case ${worst_case_usd:.2f} on top of ${already:.2f} "
            f"would cross the {kind} cap of ${cap:.2f}")
    job_id = uuid.uuid4().hex[:12]
    _append({"event": "reserve", "kind": kind, "job_id": job_id, "name": name,
             "usd": round(worst_case_usd, 6), "note": note})
    return job_id


def settle(job_id: str, kind: Kind, actual_usd: float, **detail) -> None:
    _append({"event": "settle", "kind": kind, "job_id": job_id,
             "usd": round(actual_usd, 6), **detail})


def record_manual(name: str, kind: Kind, usd: float, note: str) -> None:
    """Record spend that happened outside a guarded job (e.g. read off a bill)."""
    _append({"event": "manual", "kind": kind, "job_id": uuid.uuid4().hex[:12],
             "name": name, "usd": round(usd, 6), "note": note})


@contextlib.contextmanager
def gpu_job(name: str, container: Container, timeout_s: float,
            note: str = "") -> Iterator[dict]:
    """Guard a Modal job. Reserves timeout x price; settles wall time x price.

    Wall time on the client includes container start and image pull, which
    Modal also bills, so it is an upper bound on the true charge.
    """
    worst = container.usd_per_sec * timeout_s
    job_id = reserve(name, "gpu", worst, note=note)
    info = {"job_id": job_id, "worst_case_usd": worst}
    t0 = time.monotonic()
    try:
        yield info
    finally:
        secs = time.monotonic() - t0
        settle(job_id, "gpu", min(secs, timeout_s) * container.usd_per_sec,
               seconds=round(secs, 1), gpu=container.gpu,
               cpu_cores=container.cpu_cores, memory_gib=container.memory_gib)


@contextlib.contextmanager
def api_job(name: str, worst_case_usd: float, note: str = "") -> Iterator[dict]:
    """Guard modern-model API spend. The caller sets info['actual_usd']."""
    job_id = reserve(name, "api", worst_case_usd, note=note)
    info = {"job_id": job_id, "actual_usd": worst_case_usd}
    try:
        yield info
    finally:
        settle(job_id, "api", float(info["actual_usd"]))


def summary() -> dict:
    caps = load_caps()
    return {k: {"cap": caps[k], "spent_or_reserved": round(spent(k), 4),
                "remaining": round(caps[k] - spent(k), 4)} for k in caps}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
