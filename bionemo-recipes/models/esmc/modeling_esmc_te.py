# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TransformerEngine-optimized ESMC (EvolutionaryScale ESM-Cambrian) model.

This module provides HuggingFace-compatible ESMC model classes using NVIDIA's TransformerEngine
for optimized attention and MLP computation. The model is an encoder-only protein language model
with bidirectional attention, RoPE, SwiGLU activation, and QK LayerNorm.

Reference: EvolutionaryScale's ESMC-300M (esm PyPI package).
"""

from typing import ClassVar, Literal, Optional, Unpack

import torch
import transformer_engine.pytorch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.utils.generic import TransformersKwargs


AUTO_MAP = {
    "AutoConfig": "modeling_esmc_te.NVEsmcConfig",
    "AutoModel": "modeling_esmc_te.NVEsmcModel",
    "AutoModelForMaskedLM": "modeling_esmc_te.NVEsmcForMaskedLM",
}


class NVEsmcConfig(PretrainedConfig):
    """Configuration for the NVEsmc TransformerEngine model."""

    model_type: str = "nv_esmc"

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 960,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 15,
        intermediate_size: int = 2560,
        layer_norm_eps: float = 1e-5,
        position_embedding_type: str = "rotary",
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        # TE-specific options
        attn_input_format: Literal["bshd", "thd"] = "bshd",
        self_attn_mask_type: str = "padding",
        fuse_qkv_params: bool = True,
        qkv_weight_interleaved: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize NVEsmcConfig.

        Args:
            vocab_size: Vocabulary size (padded to 64 from real vocab of 33).
            hidden_size: Dimension of hidden representations.
            num_hidden_layers: Number of transformer layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: FFN intermediate dimension (SwiGLU corrected).
            layer_norm_eps: Layer normalization epsilon.
            position_embedding_type: Type of position embedding (only "rotary" supported).
            initializer_range: Standard deviation for weight initialization.
            pad_token_id: Padding token ID.
            attn_input_format: Attention input format for TE ("bshd" or "thd").
            self_attn_mask_type: Attention mask type ("padding" for bidirectional).
            fuse_qkv_params: Whether to fuse QKV parameters in TE.
            qkv_weight_interleaved: Whether QKV weights are interleaved.
            tie_word_embeddings: Whether to tie input/output embeddings.
            **kwargs: Additional config options.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.initializer_range = initializer_range
        self.attn_input_format = attn_input_format
        self.self_attn_mask_type = self_attn_mask_type
        self.fuse_qkv_params = fuse_qkv_params
        self.qkv_weight_interleaved = qkv_weight_interleaved


class NVEsmcPreTrainedModel(PreTrainedModel):
    """Base class for NVEsmc models."""

    config_class = NVEsmcConfig
    base_model_prefix = "esmc"
    _no_split_modules = ("TransformerLayer",)
    _tied_weights_keys: ClassVar[dict[str, str]] = {}

    def init_empty_weights(self):
        """Move model from meta device to CUDA and initialize weights."""
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.esmc.embed_tokens.to_empty(device="cuda")
        self.esmc.embed_tokens.apply(self._init_weights)

        # Meta-device init breaks weight tying, so re-tie.
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize weights for standard pytorch modules.

        TE modules handle their own initialization through `init_method` and `reset_parameters`.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            return

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def state_dict(self, *args, **kwargs):
        """Override to filter out TE's _extra_state keys."""
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}


class NVEsmcModel(NVEsmcPreTrainedModel):
    """ESMC encoder model with TransformerEngine layers."""

    def __init__(self, config: NVEsmcConfig):
        """Initialize the NVEsmc model."""
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id, dtype=config.dtype
        )

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        self.layers = nn.ModuleList(
            [
                transformer_engine.pytorch.TransformerLayer(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    bias=False,
                    layernorm_epsilon=config.layer_norm_eps,
                    hidden_dropout=0,
                    attention_dropout=0,
                    fuse_qkv_params=config.fuse_qkv_params,
                    qkv_weight_interleaved=config.qkv_weight_interleaved,
                    normalization="LayerNorm",
                    activation="swiglu",
                    attn_input_format=config.attn_input_format,
                    self_attn_mask_type=config.self_attn_mask_type,
                    num_gqa_groups=config.num_attention_heads,
                    layer_number=layer_idx + 1,
                    params_dtype=config.dtype,
                    device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                    init_method=_init_method,
                    output_layer_init_method=_init_method,
                    qk_norm_type="LayerNorm",
                    qk_norm_before_rope=True,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # TE's _create_qk_norm_modules doesn't respect params_dtype, so QK norm
        # weights default to float32. Cast them to match the model dtype to avoid
        # Q/K vs V dtype mismatch during FP8 attention.
        if config.dtype is not None:
            for layer in self.layers:
                for norm in (layer.self_attention.q_norm, layer.self_attention.k_norm):
                    if norm is not None:
                        norm.to(dtype=config.dtype)

        # Final LayerNorm (no bias, matching reference model)
        self.norm = transformer_engine.pytorch.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            params_dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        """Forward pass for the NVEsmc encoder model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments (THD params, output_hidden_states, etc.).

        Returns:
            BaseModelOutput with last_hidden_state and optional hidden_states.
        """
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Handle THD format conversion
        has_thd_input = [x in kwargs for x in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        if should_pack_inputs:
            assert attention_mask is not None, "Attention mask is required when packing BSHD inputs."
            batch_size = hidden_states.size(0)
            padded_seq_len = input_ids.size(1)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)

        if self.config.attn_input_format == "bshd" and attention_mask is not None and attention_mask.dim() == 2:
            # Convert 2D HF mask (1=attend, 0=pad) to 4D TE mask (True=masked, False=attend)
            attention_mask = attention_mask[:, None, None, :] == 0

        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_emb(max_seq_len=4096)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            hidden_states = layer(
                hidden_states,
                attention_mask=None if self.config.attn_input_format == "thd" else attention_mask,
                rotary_pos_emb=te_rope_emb,
                cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                cu_seqlens_q_padded=kwargs.get("cu_seq_lens_q_padded", None),
                cu_seqlens_kv_padded=kwargs.get("cu_seq_lens_k_padded", None),
                max_seqlen_q=kwargs.get("max_length_q", None),
                max_seqlen_kv=kwargs.get("max_length_k", None),
                pad_between_seqs=kwargs.get("pad_between_seqs", None),
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            hidden_states = _pad_input(hidden_states, indices, batch_size, padded_seq_len)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if all_hidden_states else None,
        )


class NVEsmcForMaskedLM(NVEsmcPreTrainedModel):
    """ESMC model with masked language modeling head."""

    _tied_weights_keys: ClassVar[dict[str, str]] = {}
    _do_not_quantize: ClassVar[list[str]] = ["sequence_head.dense", "sequence_head.decoder"]

    def __init__(self, config: NVEsmcConfig):
        """Initialize NVEsmcForMaskedLM."""
        super().__init__(config)
        self.esmc = NVEsmcModel(config)
        self.sequence_head = NVEsmcLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        """Get the output embeddings."""
        return self.sequence_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings."""
        self.sequence_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        """Forward pass with masked language modeling loss.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            inputs_embeds: Pre-computed input embeddings.
            labels: Labels for masked token prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            MaskedLMOutput with loss, logits, and optional hidden states.
        """
        outputs = self.esmc(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state

        with transformer_engine.pytorch.autocast(enabled=False):
            prediction_scores = self.sequence_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.to(prediction_scores.device).view(-1),
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )


class NVEsmcLMHead(nn.Module):
    """ESMC language modeling head: Linear -> GELU -> LayerNorm -> Linear.

    This matches the EvolutionaryScale `RegressionHead(d_model, output_dim)` architecture.
    """

    def __init__(self, config: NVEsmcConfig):
        """Initialize NVEsmcLMHead."""
        super().__init__()
        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.dense = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.hidden_size,
                bias=True,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
            )

            self.decoder = transformer_engine.pytorch.LayerNormLinear(
                config.hidden_size,
                config.vocab_size,
                bias=True,
                eps=config.layer_norm_eps,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
            )

    def forward(self, features):
        """Forward pass: Dense -> GELU -> LayerNormLinear."""
        with transformer_engine.pytorch.autocast(enabled=False):
            x = self.dense(features)
            x = torch.nn.functional.gelu(x)
            x = self.decoder(x)
        return x


# ===================== Utility Functions for THD Packing =====================

torch._dynamo.config.capture_scalar_outputs = True


@torch.compile
def _pad_input(hidden_states, indices, batch, seqlen):
    """Convert a THD tensor to BSHD format."""
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


@torch.compile
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Convert a BSHD tensor to THD format."""
    batch_size = hidden_states.size(0)
    seq_length = hidden_states.size(1)

    if attention_mask.shape[1] != seq_length:
        return (
            hidden_states.squeeze(1),
            torch.arange(batch_size, dtype=torch.int64, device=hidden_states.device),
            torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device),
            1,
            1,
        )

    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        hidden_states.reshape(-1, *hidden_states.shape[2:])[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )
