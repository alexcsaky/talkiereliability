"""Modal app: weights volume, scoring server and the stage 0 jobs.

All heavy work runs here; the coding session has no GPU. Every entrypoint is
wrapped in ``budget.gpu_job``, which reserves the worst case (timeout x price)
before starting and refuses to start if a cap would be crossed.

Usage (needs MODAL_TOKEN_ID and MODAL_TOKEN_SECRET):

    modal run serve/modal_app.py::download            # CPU only, ~$0.10
    modal run serve/modal_app.py::stage0 --what all   # GPU jobs, writes results/stage0/
    modal deploy serve/modal_app.py                   # optional persistent server

The deployed web endpoint requires Modal proxy auth (``requires_proxy_auth``),
so it is not public.
"""


import json
import os
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from serve.models import OFFICIAL, OFFICIAL_CODE_COMMIT, PORTS  # noqa: E402

APP_NAME = "talkie-backtest"
GPU = "L40S"
SERVER_CPU, SERVER_MEM_GIB = 4.0, 48
PARITY_MEM_GIB = 96   # the 53 GB official checkpoint is streamed through the page cache
SCALEDOWN_WINDOW = 15  # short window for batch jobs so idle GPU burn stays minimal
METHOD_TIMEOUT = 1800  # per-RPC timeout on TalkieServer
VALID_JOBS = frozenset({"parity", "context", "throughput", "known_truth", "leak", "knowledge_bench", "web"})
IT_CHAT_PROMPT = "Continue the following sentence directly without extra commentary."

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("talkie-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0", "transformers==5.8.0", "accelerate", "safetensors",
        "tiktoken==0.12.0", "huggingface_hub>=1.0", "fastapi[standard]",
        "pydantic>=2", "pyyaml", "numpy",
        f"talkie @ git+https://github.com/talkie-lm/talkie@{OFFICIAL_CODE_COMMIT}",
    )
    .env({"TALKIE_FIREWALL_LOG": "/tmp/firewall_log.jsonl"})
    .add_local_file(REPO / "firewall_lexicon.yaml", "/root/firewall_lexicon.yaml")
    .add_local_python_source("serve", "firewall", "forecaster", "controls")
)

# Gutenberg texts for the context test, all published before 1931.
# (ebook id, title as it appears in the header, publication year)
CONTEXT_TEXTS = [
    (15776, "The Economic Consequences of the Peace", 1919),
    (543, "Main Street", 1920),
    (541, "The Age of Innocence", 1920),
    (1156, "Babbitt", 1922),
    (64317, "The Great Gatsby", 1925),
]
GUTENBERG_MIRRORS = ["https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
                     "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
                     "https://aleph.gutenberg.org/cache/epub/{id}/pg{id}.txt"]
CONTEXT_CACHE_PATH = Path("/weights/context_texts.json")


def _fetch_gutenberg_texts() -> list[dict]:
    """Download and firewall-admit the pre-1931 Gutenberg context texts."""
    import urllib.request

    import firewall

    out = []
    for ebook, title, year in CONTEXT_TEXTS:
        raw = None
        for m in GUTENBERG_MIRRORS:
            try:
                with urllib.request.urlopen(m.format(id=ebook), timeout=60) as r:
                    raw = r.read().decode("utf-8", "replace")
                break
            except Exception:  # noqa: BLE001 - try the next mirror
                continue
        if raw is None or title.lower() not in raw[:3000].lower():
            out.append({"ebook": ebook, "title": title, "ok": False})
            continue
        try:
            adm = firewall.admit(raw, kind="passage", origin=f"gutenberg:{ebook}",
                                 pub_date=str(year))
            out.append({"ebook": ebook, "title": title, "ok": True, "text": adm.text,
                        "flags": list(adm.flags)})
        except firewall.FirewallRejection as e:
            out.append({"ebook": ebook, "title": title, "ok": False, "reasons": e.reasons})
    return out


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@app.function(image=image, volumes={"/weights": weights}, cpu=2, memory=16384,
              timeout=3 * 3600)
def download_weights(ports: list[str], officials: list[str]) -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download

    from serve.loaders import git_blob_sha1, official_path, port_dir, sha256_file

    report = {}
    for name in ports:
        spec = PORTS[name]
        d = port_dir(name)
        snapshot_download(spec.repo, revision=spec.revision, local_dir=str(d),
                          allow_patterns=["*.safetensors", "*.json", "vocab.txt"])
        vocab_sha1 = git_blob_sha1(d / "vocab.txt")
        if vocab_sha1 != spec.vocab_sha1:
            raise RuntimeError(
                f"{name}: vocab.txt blob SHA-1 {vocab_sha1} != pinned {spec.vocab_sha1}"
            )
        report[name] = {"dir": str(d), "files": sorted(p.name for p in d.iterdir()),
                        "vocab_sha1": vocab_sha1, "vocab_sha1_ok": True}
        weights.commit()
    for name in officials:
        spec = OFFICIAL[name]
        path = official_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        for fn in (spec.checkpoint, "vocab.txt"):
            hf_hub_download(spec.repo, fn, revision=spec.revision, local_dir=str(path.parent))
        digest = sha256_file(path)
        if digest != spec.checkpoint_sha256:
            raise RuntimeError(
                f"official:{name}: SHA-256 {digest} != pinned {spec.checkpoint_sha256}; "
                "refusing to commit corrupted checkpoint"
            )
        report[f"official:{name}"] = {"path": str(path), "sha256": digest, "sha256_ok": True}
        weights.commit()

    # Cache the Gutenberg context texts onto the volume on CPU so GPU jobs never
    # wait on external mirrors or fail mid-run if Gutenberg rate-limits.
    texts = _fetch_gutenberg_texts()
    ok = [t for t in texts if t["ok"]]
    if len(ok) >= 3:
        CONTEXT_CACHE_PATH.write_text(json.dumps(texts, ensure_ascii=False))
        weights.commit()
    report["context_texts"] = [{k: v for k, v in t.items() if k != "text"} for t in texts]
    return report


# ---------------------------------------------------------------------------
# Scoring server
# ---------------------------------------------------------------------------


@app.cls(image=image, gpu=GPU, cpu=SERVER_CPU, memory=SERVER_MEM_GIB * 1024,
         volumes={"/weights": weights}, timeout=METHOD_TIMEOUT,
         scaledown_window=SCALEDOWN_WINDOW)
class TalkieServer:
    model: str = modal.parameter(default="talkie-1930-13b-base")

    @modal.enter()
    def load(self):
        import torch

        from serve.loaders import load_port
        from serve.scoring import Engine, PortAdapter

        t0 = time.time()
        self.torch = torch
        self.hf_model, self.enc = load_port(self.model)
        self.engine = Engine(PortAdapter(self.hf_model), self.enc, max_batch_tokens=16384)
        self.load_seconds = time.time() - t0

    @modal.method()
    def health(self) -> dict:
        spec = PORTS[self.model]
        return {"model": self.model, "repo": spec.repo, "revision": spec.revision,
                "dtype": str(next(self.hf_model.parameters()).dtype),
                "device": self.torch.cuda.get_device_name(0),
                "load_seconds": round(self.load_seconds, 1),
                "prepend_eot": self.engine.prepend_eot, "max_seq_len": self.engine.max_seq_len}

    @modal.method()
    def score_batch(self, payload: list[dict], prepend_eot: bool = True,
                    chat_user_prompt: str | None = None) -> list[list[dict]]:
        self.engine.prepend_eot = prepend_eot
        try:
            res = self.engine.score_requests(
                [(p["prefix"], p["continuations"]) for p in payload],
                chat_user_prompt=chat_user_prompt,
            )
        finally:
            self.engine.prepend_eot = True
        return [[c.to_dict() for c in r] for r in res]

    @modal.method()
    def generate(self, **kw) -> dict:
        from serve.generation import generate
        return generate(self.hf_model, self.enc, style=PORTS[self.model].style, **kw)

    @modal.method()
    def throughput(self, texts: list[str]) -> dict:
        from serve.throughput import chars_per_token, measure
        from serve.tokenizer import encode_text

        joined = "\n\n".join(texts)
        ids = encode_text(self.enc, joined)
        return {
            "rows": measure(self.engine, ids),
            "tokens_per_1000_chars_gutenberg": chars_per_token(self.engine, texts),
        }

    @modal.method()
    def context(self, texts: list[str]) -> dict:
        import firewall
        from serve import context_test

        gate = lambda t: firewall.admit(t, kind="probe", origin="context-retrieval",  # noqa: E731
                                        allow_future_years=True).text
        return {"loss_vs_context": context_test.loss_vs_context(self.engine, texts),
                "loss_by_position": context_test.loss_by_position(self.engine, texts),
                "retrieval": context_test.retrieval(self.engine, texts, gate=gate)}

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        from serve.api import create_app
        return create_app({self.model: self.engine},
                          health=lambda: self.health.local(),
                          generate=lambda model, **kw: self.generate.local(**kw))


@app.function(image=image, gpu=GPU, cpu=SERVER_CPU, memory=PARITY_MEM_GIB * 1024,
              volumes={"/weights": weights}, timeout=5400)
def parity(prompts: list[str], name: str = "talkie-1930-13b-base") -> dict:
    from serve.loaders import load_official, load_port
    from serve.parity_check import run_parity
    return run_parity(lambda: load_official(name), lambda: load_port(name), prompts)


@app.function(image=image, volumes={"/weights": weights}, cpu=1, memory=2048, timeout=600)
def fetch_context_texts(force_refresh: bool = False) -> list[dict]:
    """Return cached Gutenberg texts from the volume, or fetch and admit them on CPU."""
    if not force_refresh and CONTEXT_CACHE_PATH.exists():
        cached = json.loads(CONTEXT_CACHE_PATH.read_text())
        if len([t for t in cached if t.get("ok")]) >= 3:
            return cached
    out = _fetch_gutenberg_texts()
    if len([t for t in out if t["ok"]]) >= 3:
        CONTEXT_CACHE_PATH.write_text(json.dumps(out, ensure_ascii=False))
        weights.commit()
    return out


# ---------------------------------------------------------------------------
# Local entrypoints (run on the coding session; spend guarded)
# ---------------------------------------------------------------------------


def _save(name: str, data) -> None:
    out = REPO / "results" / "stage0"
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
    print(f"wrote results/stage0/{name}")


@app.local_entrypoint()
def download(ports: str = "talkie-1930-13b-base,talkie-1930-13b-it,talkie-web-13b-base",
             officials: str = "talkie-1930-13b-base"):
    import budget
    c = budget.Container(gpu=None, cpu_cores=2, memory_gib=16)
    with budget.gpu_job("download-weights", c, timeout_s=3 * 3600):
        rep = download_weights.remote([p for p in ports.split(",") if p],
                                      [o for o in officials.split(",") if o])
    _save("download.json", rep)


@app.local_entrypoint()
def refresh_context():
    """Re-fetch and re-admit the Gutenberg context texts (after a firewall change)."""
    import budget
    cpu_box = budget.Container(gpu=None, cpu_cores=1, memory_gib=2)
    with budget.gpu_job("refresh-context-texts", cpu_box, timeout_s=600):
        texts = fetch_context_texts.remote(force_refresh=True)
    _save("context_texts.json", [{k: v for k, v in t.items() if k != "text"}
                                 | {"chars": len(t.get("text", ""))} for t in texts])


def parity_prompts() -> list[str]:
    """A few dozen admitted prompts: question frames plus longer passages."""
    import firewall
    from controls.known_truth import load
    from forecaster.elicit import build_requests

    qs, ts = load()
    reqs, rejected = build_requests(qs[::8][:28], ts[:1])
    if rejected:
        raise RuntimeError(f"parity_prompts rejected by firewall: {rejected}")
    prompts = [r.prefix.text + r.yes[0].text for r in reqs]
    planted = REPO / "tests" / "data" / "planted_passages.jsonl"
    for line in planted.read_text().splitlines():
        p = json.loads(line)
        if p["expect"] == "pass":
            prompts.append(firewall.admit(p["text"], kind="passage", origin=p["id"],
                                          pub_date=p["pub_date"]).text)
    return prompts


def _require_context_texts() -> list[dict]:
    """Fetch (or load cached) Gutenberg texts on CPU before booting any GPU container."""
    import budget
    cpu_box = budget.Container(gpu=None, cpu_cores=1, memory_gib=2)
    with budget.gpu_job("fetch-context-texts", cpu_box, timeout_s=600):
        all_texts = fetch_context_texts.remote()
    ok = [t for t in all_texts if t.get("ok")]
    if len(ok) < 3:
        raise RuntimeError(
            f"only {len(ok)} of {len(CONTEXT_TEXTS)} Gutenberg context texts succeeded; "
            "aborting before starting GPU jobs"
        )
    return ok


@app.local_entrypoint()
def stage0(what: str = "all", limit: int = 0):
    """Run stage 0 jobs. ``what``: all, or a comma list of
    parity, context, throughput, known_truth, leak, knowledge_bench, web."""
    import budget
    import firewall
    from controls import known_truth, leak_probes
    from forecaster.client import ModalBackend, TalkieClient

    if what == "all":
        jobs = {"parity", "context", "throughput", "known_truth", "leak", "web"}
    else:
        jobs = {j.strip() for j in what.split(",") if j.strip()}
        unknown = jobs - VALID_JOBS
        if unknown:
            raise ValueError(f"unknown stage0 job(s) {sorted(unknown)}; valid: {sorted(VALID_JOBS)}")

    server_box = budget.Container(GPU, SERVER_CPU, SERVER_MEM_GIB)

    # Fetch Gutenberg texts once on CPU before booting any GPU container.
    texts: list[dict] = []
    if jobs & {"parity", "context", "throughput"}:
        texts = _require_context_texts()
        _save("context_texts.json", [{k: v for k, v in t.items() if k != "text"} for t in texts])

    if "parity" in jobs:
        prompts = parity_prompts()
        prompts += [t["text"][20000:20000 + 9000] for t in texts[:3]]  # ~2k-token passages
        with budget.gpu_job("parity", budget.Container(GPU, SERVER_CPU, PARITY_MEM_GIB), 5400):
            _save("parity.json", parity.remote(prompts))

    base = TalkieServer(model="talkie-1930-13b-base")
    if jobs & {"context", "throughput", "known_truth", "leak", "knowledge_bench"}:
        n_base_steps = (
            1  # health
            + (2 if "known_truth" in jobs else 0)
            + (1 if "leak" in jobs else 0)
            + (1 if "knowledge_bench" in jobs else 0)
            + (1 if "throughput" in jobs else 0)
            + (1 if "context" in jobs else 0)
        )
        # Worst case: every sequential RPC runs to METHOD_TIMEOUT. Reserving less
        # would let the guard start a job whose worst case crosses the cap.
        base_timeout = METHOD_TIMEOUT * n_base_steps
        with budget.gpu_job("stage0-base", server_box, timeout_s=base_timeout,
                            scaledown_s=SCALEDOWN_WINDOW):
            _save("health_base.json", base.health.remote())
            confirmatory = TalkieClient(ModalBackend({"talkie-1930-13b-base": base}))
            probe_client = TalkieClient(ModalBackend({"talkie-1930-13b-base": base}),
                                        allow_leak_probes=True)
            if "known_truth" in jobs:
                _save("known_truth_base.json", known_truth.run(confirmatory, limit=limit or None))
                # Same questions without the <|endoftext|> document start (official behaviour).
                bare = TalkieClient(ModalBackend({"talkie-1930-13b-base": base}, prepend_eot=False))
                _save("known_truth_base_noeot.json", known_truth.run(bare, limit=limit or None))
            if "leak" in jobs:
                _save("leak_base.json", leak_probes.choice_and_year(probe_client))
            if "knowledge_bench" in jobs:
                _save("knowledge_bench_base.json", leak_probes.knowledge_bench(probe_client))
            if "throughput" in jobs:
                _save("throughput.json", base.throughput.remote([t["text"] for t in texts]))
            if "context" in jobs:
                _save("context.json", base.context.remote([t["text"] for t in texts]))

    if "web" in jobs:
        srv = TalkieServer(model="talkie-web-13b-base")
        with budget.gpu_job("stage0-talkie-web-13b-base", server_box,
                            timeout_s=METHOD_TIMEOUT, scaledown_s=SCALEDOWN_WINDOW):
            confirmatory = TalkieClient(ModalBackend({"talkie-web-13b-base": srv}),
                                        model="talkie-web-13b-base")
            probe_client = TalkieClient(ModalBackend({"talkie-web-13b-base": srv}),
                                        model="talkie-web-13b-base", allow_leak_probes=True)
            _save("known_truth_web.json", known_truth.run(confirmatory, limit=limit or None))
            _save("leak_web.json", leak_probes.choice_and_year(probe_client))

    if "leak" in jobs:
        srv = TalkieServer(model="talkie-1930-13b-it")
        with budget.gpu_job("stage0-talkie-1930-13b-it", server_box,
                            timeout_s=METHOD_TIMEOUT, scaledown_s=SCALEDOWN_WINDOW):
            raw_probe = TalkieClient(ModalBackend({"talkie-1930-13b-it": srv}),
                                     model="talkie-1930-13b-it", allow_leak_probes=True)
            chat_probe = TalkieClient(
                ModalBackend({"talkie-1930-13b-it": srv}, prepend_eot=False,
                             chat_user_prompt=firewall.admit(
                                 IT_CHAT_PROMPT, kind="probe", origin="it-chat-probe")),
                model="talkie-1930-13b-it", allow_leak_probes=True,
            )
            _save("leak_it.json", leak_probes.choice_and_year(raw_probe))
            _save("leak_it_chat.json", leak_probes.choice_and_year(chat_probe))

    print(json.dumps(budget.summary(), indent=1))
