"""Load Talkie weights inside the GPU container (never in the coding session).

* ``load_port`` loads the bf16 safetensors of the transformers port with our
  vendored, reviewed model class, so no hub code runs.
* ``load_official`` loads the official FP32 checkpoint with the official model
  class, for the parity check only. It uses ``torch.load(weights_only=True,
  mmap=True)``: no pickled code can run, and the 53 GB file is streamed rather
  than read into memory. Weights are cast to bf16 as the official
  ``load_checkpoint`` does (it casts with ``.to(torch.bfloat16)``; copying an
  fp32 tensor into a bf16 parameter rounds the same way).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from serve.models import OFFICIAL, PORTS
from serve.tokenizer import build_tokenizer

WEIGHTS_ROOT = Path("/weights")


def port_dir(name: str) -> Path:
    spec = PORTS[name]
    return WEIGHTS_ROOT / "ports" / spec.repo.replace("/", "__") / spec.revision


def official_path(name: str) -> Path:
    spec = OFFICIAL[name]
    return WEIGHTS_ROOT / "official" / spec.repo.replace("/", "__") / spec.revision / spec.checkpoint


def load_port(name: str, device: str = "cuda"):
    from serve.vendor.talkie_port import TalkieForCausalLM

    path = port_dir(name)
    model = TalkieForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map={"": device}, use_safetensors=True)
    model.eval()
    enc = build_tokenizer(path / "vocab.txt", PORTS[name].style)
    return model, enc


def load_official(name: str, device: str = "cuda"):
    from talkie.model import GPTConfig, TalkieModel

    path = official_path(name)
    sd = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    for key in ("model_state_dict", "model"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    cfg = GPTConfig(vocab_size=sd["embed.weight"].shape[0])
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = TalkieModel(cfg, torch.device(device))
    finally:
        torch.set_default_dtype(prev)
    model.load_state_dict(sd, strict=True)
    del sd
    model.device = torch.device(device)
    model.eval()
    vocab = path.parent / "vocab.txt"
    return model, build_tokenizer(vocab, OFFICIAL[name].style)


def sha256_file(path: Path, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Compute the git blob SHA-1 reported by the Hugging Face Hub API."""
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()
