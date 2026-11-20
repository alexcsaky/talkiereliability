from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers import GenerationMixin
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:  # pragma: no cover - compatibility with older Transformers.
    ALL_ATTENTION_FUNCTIONS = None

from .configuration_talkie import TalkieConfig


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    del kwargs
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal and attention_mask is None,
    )
    return output.transpose(1, 2).contiguous(), None


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones([n_head]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.w_g.type_as(w)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


class CausalSelfAttention(nn.Module):
    is_causal = True

    def __init__(self, config: TalkieConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        n_state = config.n_embd

        self.attn_query = nn.Linear(n_state, n_state, bias=False)
        self.attn_key = nn.Linear(n_state, n_state, bias=False)
        self.attn_value = nn.Linear(n_state, n_state, bias=False)
        self.attn_resid = nn.Linear(n_state, n_state, bias=False)
        self.head_gain = HeadGain(config.n_head)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.attn_query(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.attn_key(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.attn_value(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        key_states = k.transpose(1, 2)
        value_states = v.transpose(1, 2)
        if kwargs.get("past_key_values") is not None:
            key_states, value_states = kwargs["past_key_values"].update(
                key_states, value_states, self.layer_idx
            )

        if ALL_ATTENTION_FUNCTIONS is None:
            attention_interface = eager_attention_forward
        elif hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
        else:  # pragma: no cover - compatibility with older Transformers.
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(
                self.config._attn_implementation, eager_attention_forward
            )
        is_causal = attention_mask is None and key_states.shape[-2] == q.shape[1]
        y, _ = attention_interface(
            self,
            q.transpose(1, 2),
            key_states,
            value_states,
            attention_mask,
            is_causal=is_causal,
            **kwargs,
        )
        y = y.contiguous().view_as(x)
        return self.attn_resid(y)


class MLP(nn.Module):
    def __init__(self, config: TalkieConfig):
        super().__init__()
        n_state = config.n_embd
        n_mlp = int(round(((8 / 3) * n_state) / 128) * 128)

        self.mlp_gate = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, n_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.mlp_gate(x)) * self.mlp_linear(x)
        return self.mlp_resid(x)


class Block(nn.Module):
    def __init__(self, config: TalkieConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.attn_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.mlp = MLP(config)
        self.mlp_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def forward(
        self,
        e_x: torch.Tensor,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = x + self.attn_gain(
            self.attn(F.rms_norm(x, (x.shape[-1],)), cos_sin, attention_mask, **kwargs)
        )
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        return x


class TalkiePreTrainedModel(PreTrainedModel):
    config_class = TalkieConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _no_split_modules = ["Block"]
    _tied_weights_keys = None

    def _init_weights(self, module: nn.Module) -> None:
        return


class TalkieModel(TalkiePreTrainedModel, GenerationMixin):
    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.gradient_checkpointing = False

        cos, sin = self._precompute_rotary_embeddings(config.max_position_embeddings)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._rotary_initialized = cos.device.type != "meta"
        self.post_init()

    def _apply(self, fn, recurse: bool = True):
        for name in ("cos", "sin"):
            buffer = self._buffers.get(name)
            if buffer is not None and getattr(buffer, "is_meta", False):
                self._buffers[name] = torch.empty(0, dtype=buffer.dtype)
                self._rotary_initialized = False
        return super()._apply(fn, recurse=recurse)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int | None = None,
        base: int | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device if hasattr(self, "embed") else "cpu"
        head_dim = head_dim if head_dim is not None else self.config.head_dim
        base = base if base is not None else self.config.rope_base
        inv_freq, attention_factor = self._rotary_inv_freq(seq_len, head_dim, float(base), device)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        if attention_factor != 1.0:
            cos = cos * attention_factor
            sin = sin * attention_factor
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _rotary_inv_freq(
        self,
        seq_len: int,
        head_dim: int,
        base: float,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, float]:
        scaling = self.config.rope_scaling
        rope_type = scaling.get("rope_type") if scaling else None
        if rope_type in (None, "default"):
            return self._default_rotary_inv_freq(head_dim, base, device), 1.0
        if rope_type == "linear":
            inv_freq = self._default_rotary_inv_freq(head_dim, base, device)
            return inv_freq / float(scaling["factor"]), 1.0
        if rope_type == "dynamic":
            return self._dynamic_rotary_inv_freq(seq_len, head_dim, base, device, scaling), 1.0
        if rope_type == "yarn":
            return self._yarn_rotary_inv_freq(head_dim, base, device, scaling)
        raise ValueError(f"unsupported rope_scaling type {rope_type!r}")

    @staticmethod
    def _default_rotary_inv_freq(
        head_dim: int, base: float, device: torch.device | str
    ) -> torch.Tensor:
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (base ** (channel_range / head_dim))

    def _original_max_position_embeddings(self, scaling: dict | None) -> int:
        if scaling and "original_max_position_embeddings" in scaling:
            return int(scaling["original_max_position_embeddings"])
        return int(self.config.max_position_embeddings)

    def _dynamic_rotary_inv_freq(
        self,
        seq_len: int,
        head_dim: int,
        base: float,
        device: torch.device | str,
        scaling: dict,
    ) -> torch.Tensor:
        original_max_position_embeddings = self._original_max_position_embeddings(scaling)
        scaled_seq_len = max(seq_len, original_max_position_embeddings)
        factor = float(scaling["factor"])
        base = base * (
            (factor * scaled_seq_len / original_max_position_embeddings) - (factor - 1.0)
        ) ** (head_dim / (head_dim - 2.0))
        return self._default_rotary_inv_freq(head_dim, base, device)

    def _yarn_rotary_inv_freq(
        self,
        head_dim: int,
        base: float,
        device: torch.device | str,
        scaling: dict,
    ) -> tuple[torch.Tensor, float]:
        factor = float(scaling["factor"])
        original_max_position_embeddings = self._original_max_position_embeddings(scaling)
        beta_fast = float(scaling.get("beta_fast", 32.0))
        beta_slow = float(scaling.get("beta_slow", 1.0))
        attention_factor = scaling.get("attention_factor")
        if attention_factor is None:
            attention_factor = 1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        pos_freqs = base ** (channel_range / head_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = self._yarn_correction_range(
            beta_fast,
            beta_slow,
            head_dim,
            base,
            original_max_position_embeddings,
            truncate=bool(scaling.get("truncate", True)),
        )
        ramp = self._yarn_linear_ramp(low, high, head_dim // 2, device)
        extrapolation_factor = 1.0 - ramp
        inv_freq = (
            inv_freq_interpolation * (1.0 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor
        )
        return inv_freq, float(attention_factor)

    @staticmethod
    def _yarn_correction_range(
        low_rot: float,
        high_rot: float,
        head_dim: int,
        base: float,
        original_max_position_embeddings: int,
        truncate: bool,
    ) -> tuple[float, float]:
        def correction_dim(num_rotations: float) -> float:
            return (
                head_dim
                * math.log(original_max_position_embeddings / (num_rotations * 2.0 * math.pi))
                / (2.0 * math.log(base))
            )

        low = correction_dim(low_rot)
        high = correction_dim(high_rot)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0.0), min(high, float(head_dim - 1))

    @staticmethod
    def _yarn_linear_ramp(
        low: float,
        high: float,
        dim: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
        return torch.clamp(ramp, 0.0, 1.0)

    def _ensure_rotary_embeddings(self, seq_len: int) -> None:
        device = self.embed.weight.device
        needs_init = (
            not self._rotary_initialized
            or self.cos.device != device
            or self.sin.device != device
            or self.cos.shape[1] < seq_len
        )
        if needs_init:
            max_seq_len = max(seq_len, self.config.max_position_embeddings)
            cos, sin = self._precompute_rotary_embeddings(max_seq_len)
            self.cos = cos.to(device=device)
            self.sin = sin.to(device=device)
            self._rotary_initialized = True

    def reset_rotary_embeddings(self) -> None:
        self._rotary_initialized = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed = value

    def _position_ids(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.LongTensor:
        batch_size, seq_len = input_ids.shape
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            return position_ids.to(device=input_ids.device, dtype=torch.long)
        if cache_position is not None:
            if cache_position.dim() == 1:
                cache_position = cache_position.unsqueeze(0)
            if cache_position.shape[0] == 1 and batch_size != 1:
                cache_position = cache_position.expand(batch_size, -1)
            return cache_position.to(device=input_ids.device, dtype=torch.long)
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long) + past_seen
        return position_ids.unsqueeze(0).expand(batch_size, -1)

    def _attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if attention_mask is not None and attention_mask.dim() >= 4:
            return attention_mask
        batch_size, query_length = input_ids.shape
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0

        if attention_mask is not None and attention_mask.dim() != 2:
            return attention_mask
        if attention_mask is None and past_seen == 0:
            return None

        key_length = past_seen + query_length
        if attention_mask is not None:
            if attention_mask.shape[-1] == query_length and past_seen:
                prefix = torch.ones(
                    attention_mask.shape[0],
                    past_seen,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([prefix, attention_mask], dim=-1)
            key_length = attention_mask.shape[-1]

        key_positions = torch.arange(key_length, device=input_ids.device, dtype=torch.long)
        future_mask = key_positions.view(1, 1, 1, key_length) > position_ids.view(
            batch_size, 1, query_length, 1
        )
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(device=input_ids.device) == 0
            mask = future_mask | padding_mask
        else:
            mask = future_mask

        min_value = torch.finfo(dtype).min
        causal_mask = torch.zeros(
            batch_size, 1, query_length, key_length, dtype=dtype, device=input_ids.device
        )
        return causal_mask.masked_fill(mask, min_value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple[torch.Tensor, ...]:
        cache_position = kwargs.pop("cache_position", None)
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds is required")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("provide only one of input_ids or inputs_embeds")
        if input_ids is None:
            input_ids = torch.empty(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        position_ids = self._position_ids(input_ids, position_ids, cache_position, past_key_values)
        # Keep graph capture free of CUDA tensor -> Python scalar syncs. The
        # configured context length is the static serving/training contract.
        self._ensure_rotary_embeddings(int(self.config.max_position_embeddings))

        cos = self.cos[0, position_ids, :, :]
        sin = self.sin[0, position_ids, :, :]
        cos_sin = cos, sin
        x = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        x = F.rms_norm(x, (x.shape[-1],))
        attention_mask = self._attention_mask(attention_mask, input_ids, position_ids, past_key_values, x.dtype)
        e_x = x
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                def custom_forward(
                    e_x: torch.Tensor,
                    x: torch.Tensor,
                    cos: torch.Tensor,
                    sin: torch.Tensor,
                    attention_mask: torch.Tensor | None,
                    block: Block = block,
                ) -> torch.Tensor:
                    return block(e_x, x, (cos, sin), attention_mask=attention_mask)

                x = self._gradient_checkpointing_func(
                    custom_forward,
                    e_x,
                    x,
                    cos,
                    sin,
                    attention_mask,
                )
            else:
                x = block(
                    e_x,
                    x,
                    cos_sin,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values if use_cache else None,
                    **kwargs,
                )
        x = F.rms_norm(x, (x.shape[-1],))
        past_key_values = past_key_values if use_cache else None
        use_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if use_return_dict:
            return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values)
        output = (x,)
        return output + ((past_key_values,) if past_key_values is not None else ())


class TalkieForCausalLM(TalkieModel):
    _tied_weights_keys = None

    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def _chunked_lm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)
        total_tokens = hidden_states.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden_states.shape[1], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[1])
            logits = self.lm_head(hidden_states[:, start:end, :]).float()
            if self.config.logit_scale != 1.0:
                logits = logits * self.config.logit_scale
            chunk_labels = labels[:, start:end].contiguous()
            total_loss = total_loss + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_tokens = total_tokens + (chunk_labels != -100).sum(dtype=torch.float32)
        return total_loss / total_tokens.clamp_min(1.0)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        position_ids: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        loss_chunk_size: int = 0,
        return_logits: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds is required")
        cache_position = kwargs.pop("cache_position", None)
        outputs = super().forward(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        loss = None
        logits = None
        if labels is not None and loss_chunk_size > 0:
            loss = self._chunked_lm_loss(
                hidden_states[:, :-1, :],
                labels[:, 1:],
                loss_chunk_size,
            )
        if return_logits:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :]).float()
            if self.config.logit_scale != 1.0:
                logits = logits * self.config.logit_scale

        if labels is not None and loss is None:
            if logits is None:
                raise ValueError("return_logits must be true when loss_chunk_size is not used")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        use_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if use_return_dict:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
            )
        output = (logits,)
        if outputs.past_key_values is not None:
            output += (outputs.past_key_values,)
        return ((loss,) + output) if loss is not None else output


__all__ = ["TalkieConfig", "TalkieForCausalLM", "TalkieModel"]
