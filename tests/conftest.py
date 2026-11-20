"""Shared fixtures: tiny random Talkie models (official class and port) on CPU."""

from pathlib import Path

import pytest
import torch

VOCAB = Path(__file__).resolve().parents[1] / "data" / "models" / "vocab.txt"

N_LAYER, N_HEAD, HEAD_DIM = 2, 2, 16
N_EMBD = N_HEAD * HEAD_DIM


def official_to_port(sd: dict) -> dict:
    out = {k: v for k, v in sd.items() if k not in ("lm_head", "lm_head_gain.w_g")}
    out["lm_head.weight"] = sd["lm_head"] * sd["lm_head_gain.w_g"]
    return out


@pytest.fixture(scope="session")
def models():
    talkie_model = pytest.importorskip("talkie.model")
    from serve.vendor.talkie_port import TalkieConfig, TalkieForCausalLM

    torch.manual_seed(0)
    cfg = talkie_model.GPTConfig(vocab_size=65536, n_layer=N_LAYER, n_head=N_HEAD,
                                 n_embd=N_EMBD, head_dim=HEAD_DIM)
    off = talkie_model.TalkieModel(cfg, torch.device("cpu"), max_seq_len=4096)
    with torch.no_grad():
        for p in off.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    off.eval()
    pcfg = TalkieConfig(vocab_size=65536, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                        head_dim=HEAD_DIM, max_position_embeddings=4096)
    port = TalkieForCausalLM(pcfg)
    missing, unexpected = port.load_state_dict(official_to_port(off.state_dict()), strict=False)
    assert not unexpected and all(k in ("cos", "sin") for k in missing)
    port.eval()
    return off, port


@pytest.fixture(scope="session")
def enc():
    if not VOCAB.exists():
        pytest.skip("run data/fetch_vocab.sh")
    from serve.tokenizer import build_tokenizer

    return build_tokenizer(VOCAB, "base")


