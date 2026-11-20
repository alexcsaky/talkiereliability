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

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("talkie-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0", "transformers==5.8.0", "accelerate", "safetensors",
        "tiktoken==0.12.0", "huggingface_hub[hf_transfer]", "fastapi[standard]",
        "pydantic>=2", "pyyaml", "numpy",
        f"talkie @ git+https://github.com/talkie-lm/talkie@{OFFICIAL_CODE_COMMIT}",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TALKIE_FIREWALL_LOG": "/tmp/firewall_log.jsonl"})
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


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@app.function(image=image, volumes={"/weights": weights}, cpu=2, memory=4096,
              timeout=3 * 3600)
def download_weights(ports: list[str], officials: list[str]) -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download

    from serve.loaders import official_path, port_dir, sha256_file

    report = {}
    for name in ports:
        spec = PORTS[name]
        d = port_dir(name)
        snapshot_download(spec.repo, revision=spec.revision, local_dir=str(d),
                          allow_patterns=["*.safetensors", "*.json", "vocab.txt"])
        report[name] = {"dir": str(d), "files": sorted(p.name for p in d.iterdir())}
        weights.commit()
    for name in officials:
        spec = OFFICIAL[name]
        path = official_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        for fn in (spec.checkpoint, "vocab.txt"):
            hf_hub_download(spec.repo, fn, revision=spec.revision, local_dir=str(path.parent))
        digest = sha256_file(path)
        report[f"official:{name}"] = {"path": str(path), "sha256": digest,
                                      "sha256_ok": digest == spec.checkpoint_sha256}
        weights.commit()
    return report


# ---------------------------------------------------------------------------
# Scoring server
# ---------------------------------------------------------------------------


@app.cls(image=image, gpu=GPU, cpu=SERVER_CPU, memory=SERVER_MEM_GIB * 1024,
         volumes={"/weights": weights}, timeout=3600, scaledown_window=120)
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
    def score_batch(self, payload: list[dict], prepend_eot: bool = True) -> list[list[dict]]:
        self.engine.prepend_eot = prepend_eot
        try:
            res = self.engine.score_requests([(p["prefix"], p["continuations"]) for p in payload])
        finally:
            self.engine.prepend_eot = True
        return [[c.to_dict() for c in r] for r in res]

    @modal.method()
    def generate(self, **kw) -> dict:
        from serve.generation import generate
        return generate(self.hf_model, self.enc, style=PORTS[self.model].style, **kw)

    @modal.method()
    def throughput(self, text_ids: list[int]) -> list[dict]:
        from serve.throughput import measure
        return measure(self.engine, text_ids)

    @modal.method()
    def context(self, texts: list[str]) -> dict:
        import firewall
        from serve import context_test

        gate = lambda t: firewall.admit(t, kind="probe", origin="context-retrieval",  # noqa: E731
                                        allow_future_years=True, log=False).text
        return {"loss_vs_context": context_test.loss_vs_context(self.engine, texts),
                "loss_by_position": context_test.loss_by_position(self.engine, texts),
                "retrieval": context_test.retrieval(self.engine, texts, gate=gate)}

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        from serve.api import create_app
        return create_app({self.model: self.engine},
                          health=lambda: {"model": self.model, "revision": PORTS[self.model].revision},
                          generate=lambda model, **kw: self.generate.local(**kw))


@app.function(image=image, gpu=GPU, cpu=SERVER_CPU, memory=PARITY_MEM_GIB * 1024,
              volumes={"/weights": weights}, timeout=5400)
def parity(prompts: list[str], name: str = "talkie-1930-13b-base") -> dict:
    from serve.loaders import load_official, load_port
    from serve.parity_check import run_parity
    return run_parity(lambda: load_official(name), lambda: load_port(name), prompts)


@app.function(image=image, cpu=1, memory=2048, timeout=600)
def fetch_context_texts() -> list[dict]:
    """Fetch and admit the Gutenberg texts (Modal has open internet)."""
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
                                 pub_date=str(year), log=False)
            out.append({"ebook": ebook, "title": title, "ok": True, "text": adm.text,
                        "flags": list(adm.flags)})
        except firewall.FirewallRejection as e:
            out.append({"ebook": ebook, "title": title, "ok": False, "reasons": e.reasons})
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
    c = budget.Container(gpu=None, cpu_cores=2, memory_gib=4)
    with budget.gpu_job("download-weights", c, timeout_s=3 * 3600):
        rep = download_weights.remote([p for p in ports.split(",") if p],
                                      [o for o in officials.split(",") if o])
    _save("download.json", rep)


def parity_prompts() -> list[str]:
    """A few dozen admitted prompts: question frames plus longer passages."""
    import firewall
    from controls.known_truth import load
    from forecaster.elicit import build_requests

    qs, ts = load()
    reqs, _ = build_requests(qs[::8][:28], ts[:1])
    prompts = [r.prefix.text + r.yes[0].text for r in reqs]
    planted = REPO / "tests" / "data" / "planted_passages.jsonl"
    for line in planted.read_text().splitlines():
        p = json.loads(line)
        if p["expect"] == "pass":
            prompts.append(firewall.admit(p["text"], kind="passage", origin=p["id"],
                                          pub_date=p["pub_date"]).text)
    return prompts


@app.local_entrypoint()
def stage0(what: str = "all", limit: int = 0):
    """Run stage 0 jobs. ``what``: all, or a comma list of
    parity, context, throughput, known_truth, leak, knowledge_bench, web."""
    import budget
    from controls import known_truth, leak_probes
    from forecaster.client import ModalBackend, TalkieClient

    jobs = set(what.split(",")) if what != "all" else {
        "parity", "context", "throughput", "known_truth", "leak", "web"}
    server_box = budget.Container(GPU, SERVER_CPU, SERVER_MEM_GIB)

    if "parity" in jobs:
        prompts = parity_prompts()
        texts = [t for t in fetch_context_texts.remote() if t["ok"]]
        prompts += [t["text"][20000:20000 + 9000] for t in texts[:3]]  # ~2k-token passages
        with budget.gpu_job("parity", budget.Container(GPU, SERVER_CPU, PARITY_MEM_GIB), 5400):
            _save("parity.json", parity.remote(prompts))

    base = TalkieServer(model="talkie-1930-13b-base")
    if jobs & {"context", "throughput", "known_truth", "leak", "knowledge_bench"}:
        with budget.gpu_job("stage0-base", server_box, timeout_s=3600):
            _save("health_base.json", base.health.remote())
            client = TalkieClient(ModalBackend({"talkie-1930-13b-base": base}))
            if "known_truth" in jobs:
                _save("known_truth_base.json", known_truth.run(client, limit=limit or None))
                # Same questions without the <|endoftext|> document start (official behaviour).
                bare = TalkieClient(ModalBackend({"talkie-1930-13b-base": base}, prepend_eot=False))
                _save("known_truth_base_noeot.json", known_truth.run(bare, limit=limit or None))
            if "leak" in jobs:
                _save("leak_base.json", leak_probes.choice_and_year(client))
            if "knowledge_bench" in jobs:
                _save("knowledge_bench_base.json", leak_probes.knowledge_bench(client))
            if jobs & {"context", "throughput"}:
                texts = [t for t in fetch_context_texts.remote() if t["ok"]]
                _save("context_texts.json", [{k: v for k, v in t.items() if k != "text"}
                                              for t in texts])
                if "throughput" in jobs:
                    from serve.tokenizer import build_tokenizer, encode_text
                    vocab = REPO / "data" / "models" / "vocab.txt"
                    enc = build_tokenizer(vocab)
                    ids = encode_text(enc, "\n\n".join(t["text"] for t in texts))
                    _save("throughput.json", {
                        "rows": base.throughput.remote(ids),
                        "tokens_per_1000_chars_gutenberg": 1000 * len(ids) / sum(
                            len(t["text"]) for t in texts)})
                if "context" in jobs:
                    _save("context.json", base.context.remote([t["text"] for t in texts]))

    if "web" in jobs or "leak" in jobs:
        # Ceiling model (closed book) and the chat model's leak profile.
        for model, files in (("talkie-web-13b-base", ["known_truth_web.json", "leak_web.json"]),
                             ("talkie-1930-13b-it", ["leak_it.json"])):
            if model == "talkie-web-13b-base" and "web" not in jobs:
                continue
            srv = TalkieServer(model=model)
            with budget.gpu_job(f"stage0-{model}", server_box, timeout_s=1800):
                client = TalkieClient(ModalBackend({model: srv}), model=model)
                if "known_truth_web.json" in files:
                    _save(files[0], known_truth.run(client, limit=limit or None))
                _save(files[-1], leak_probes.choice_and_year(client))

    print(json.dumps(budget.summary(), indent=1))
