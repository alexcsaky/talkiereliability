from __future__ import annotations

from collections.abc import Mapping

from transformers import PretrainedConfig


class TalkieConfig(PretrainedConfig):
    model_type = "talkie"

    def __init__(
        self,
        vocab_size: int = 65536,
        n_layer: int = 40,
        n_head: int = 40,
        n_embd: int = 5120,
        head_dim: int = 128,
        max_position_embeddings: int = 4096,
        rope_base: int = 1_000_000,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        logit_scale: float = 1.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] = 65535,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        if rope_scaling is None:
            rope_scaling = rope_parameters
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = self._normalize_rope_scaling(rope_scaling)
        self.rope_parameters = self.rope_scaling
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.rope_scaling = self._normalize_rope_scaling(rope_scaling)
        self.rope_parameters = self.rope_scaling
        self.logit_scale = logit_scale
        self.use_cache = use_cache

        # Common Transformers aliases used by generation/cache helpers.
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head

    @staticmethod
    def _normalize_rope_scaling(rope_scaling: dict | None) -> dict | None:
        if rope_scaling is None:
            return None
        if not isinstance(rope_scaling, Mapping):
            raise TypeError("rope_scaling must be a dictionary")

        scaling = dict(rope_scaling)
        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type is None:
            raise ValueError("rope_scaling must include 'rope_type' or 'type'")

        rope_type = str(rope_type).lower()
        if rope_type == "ntk":
            rope_type = "dynamic"
        supported = {"default", "linear", "dynamic", "yarn"}
        if rope_type not in supported:
            raise ValueError(
                f"unsupported rope_scaling type {rope_type!r}; expected one of {sorted(supported)}"
            )

        if rope_type == "default":
            return None

        factor = float(scaling.get("factor", 1.0))
        if factor < 1.0:
            raise ValueError("rope_scaling factor must be >= 1.0")

        scaling["rope_type"] = rope_type
        scaling.pop("type", None)
        scaling["factor"] = factor
        if "original_max_position_embeddings" in scaling:
            scaling["original_max_position_embeddings"] = int(
                scaling["original_max_position_embeddings"]
            )
        if "beta_fast" in scaling:
            scaling["beta_fast"] = float(scaling["beta_fast"])
        if "beta_slow" in scaling:
            scaling["beta_slow"] = float(scaling["beta_slow"])
        if "attention_factor" in scaling and scaling["attention_factor"] is not None:
            scaling["attention_factor"] = float(scaling["attention_factor"])
        return scaling
