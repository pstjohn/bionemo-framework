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

from typing import Unpack

import torch
import torch.nn as nn
import transformer_engine.pytorch
import transformers
from transformers import LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils.generic import TransformersKwargs


class NVLlamaConfig(LlamaConfig): ...


class NVLlamaPreTrainedModel(PreTrainedModel):
    config: NVLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["TransformerLayer"]
    _skip_keys_device_placement = ["past_key_values"]


class NVLlamaModel(NVLlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)
        self.layers = nn.ModuleList(
            [
                transformer_engine.pytorch.TransformerLayer(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    bias=False,
                    layernorm_epsilon=config.rms_norm_eps,
                    fuse_qkv_params=False,
                    qkv_weight_interleaved=False,
                    hidden_dropout=0,
                    attention_dropout=0,
                    normalization="RMSNorm",
                    activation="swiglu",
                    attn_input_format="bshd",
                    num_gqa_groups=config.num_key_value_heads,
                    layer_number=layer_idx + 1,
                    params_dtype=config.dtype,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = transformer_engine.pytorch.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype)
        # self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        # self.rotary_emb.inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](config, device=None)
        self.rotary_emb = NVLlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        all_hidden_states = []
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = transformers.cache_utils.DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = transformers.masking_utils.create_causal_mask(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids).transpose(0, 1).unsqueeze(1)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                self_attn_mask_type="causal",
                rotary_pos_emb=position_embeddings,
                # position_ids=position_ids,
                # past_key_values=past_key_values,
                # cache_position=cache_position,
                # **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class NVLlamaForCausalLM(NVLlamaPreTrainedModel, transformers.GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = NVLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = transformer_engine.pytorch.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            params_dtype=config.dtype,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NVLlamaForSequenceClassification(
    transformers.modeling_layers.GenericForSequenceClassification, NVLlamaPreTrainedModel
): ...


class NVLlamaForQuestionAnswering(transformers.modeling_layers.GenericForQuestionAnswering, NVLlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class NVLlamaForTokenClassification(
    transformers.modeling_layers.GenericForTokenClassification, NVLlamaPreTrainedModel
): ...


class NVLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

        return emb.to(dtype=x.dtype)
