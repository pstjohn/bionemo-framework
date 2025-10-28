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

import torch
import torch.nn as nn
from nemo.lightning import io

from modeling_llama_te import NVLlamaConfig, NVLlamaModel


mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.q_proj.weight": "model.layers.*.self_attention.layernorm_qkv.query_weight",
    "model.layers.*.self_attn.k_proj.weight": "model.layers.*.self_attention.layernorm_qkv.key_weight",
    "model.layers.*.self_attn.v_proj.weight": "model.layers.*.self_attention.layernorm_qkv.value_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.self_attention.layernorm_mlp.layer_norm_weight",
    "model.layers.*.mlp.gate_proj.weight": "model.layers.*.self_attention.layernorm_mlp.fc1_weight",
    "model.layers.*.mlp.up_proj.weight": "model.layers.*.self_attention.layernorm_mlp.fc1_weight",
    "model.layers.*.mlp.down_proj.weight": "model.layers.*.self_attention.layernorm_mlp.fc2_weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def convert_llama_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Hugging Face model to a Transformer Engine model.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to NVLlamaConfig.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    # TODO (peter): this is super similar method to the AMPLIFY one, maybe we can abstract or keep simlar naming? models/amplify/src/amplify/state_dict_convert.py:convert_amplify_hf_to_te
    te_config = NVLlamaConfig(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVLlamaModel(te_config)

    output_model = io.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [_pack_qkv_weight, _pack_qkv_bias],
    )

    output_model.tie_weights()

    return output_model


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="esm.encoder.layers.*.self_attention.layernorm_qkv.weight",
)
def _pack_qkv_weight(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    num_heads = ctx.target.config.num_attention_heads
    # transpose weights
    # [sequence length, batch size, num_splits_model_parallel * attention head size * #attention heads]
    # --> [sequence length, batch size, attention head size * num_splits_model_parallel * #attention heads]
    concat_weights = concat_weights.view(3, num_heads, -1, query.size()[-1])
    concat_weights = concat_weights.transpose(0, 1).contiguous()
    concat_weights = concat_weights.view(*input_shape)
    return concat_weights


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="esm.encoder.layers.*.self_attention.layernorm_qkv.bias",
)
def _pack_qkv_bias(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_biases = torch.cat((query, key, value), dim=0)
    input_shape = concat_biases.size()
    num_heads = ctx.target.config.num_attention_heads
    # transpose biases
    # [num_splits_model_parallel * attention head size * #attention heads]
    # --> [attention head size * num_splits_model_parallel * #attention heads]
    concat_biases = concat_biases.view(3, num_heads, -1)
    concat_biases = concat_biases.transpose(0, 1).contiguous()
    concat_biases = concat_biases.view(*input_shape)
    return concat_biases


@io.state_transform(
    source_key="esm.encoder.layers.*.self_attention.layernorm_qkv.weight",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
)
def _unpack_qkv_weight(ctx: io.TransformCTX, qkv_weight):
    """Unpack fused QKV weights into separate [hidden_size, input_dim] tensors for query/key/value."""
    num_heads = ctx.source.config.num_attention_heads
    total_rows, input_dim = qkv_weight.size()  # size: [num_heads * 3 *head_dim, input_dim]
    assert total_rows % (3 * num_heads) == 0, (
        f"QKV weight rows {total_rows} not divisible by 3*num_heads {3 * num_heads}"
    )
    head_dim = total_rows // (3 * num_heads)

    qkv_weight = (
        qkv_weight.view(num_heads, 3, head_dim, input_dim).transpose(0, 1).contiguous()
    )  # size: [3, num_heads, head_dim, input_dim]
    query, key, value = qkv_weight[0], qkv_weight[1], qkv_weight[2]  # size: [num_heads, head_dim, input_dim]

    query = query.reshape(-1, input_dim)  # size: [num_heads * head_dim, input_dim]
    key = key.reshape(-1, input_dim)  # size: [num_heads * head_dim, input_dim]
    value = value.reshape(-1, input_dim)  # size: [num_heads * head_dim, input_dim]

    return query, key, value


@io.state_transform(
    source_key="esm.encoder.layers.*.self_attention.layernorm_qkv.bias",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
)
def _unpack_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    """Unpack fused QKV biases into separate [hidden_size] tensors for query/key/value."""
    num_heads = ctx.source.config.num_attention_heads
    total_size = qkv_bias.size(0)  # size: [num_heads * 3 * head_dim]
    assert total_size % (3 * num_heads) == 0, (
        f"QKV bias size {total_size} not divisible by 3*num_heads {3 * num_heads}"
    )
    head_dim = total_size // (3 * num_heads)

    qkv_bias = qkv_bias.view(num_heads, 3, head_dim).transpose(0, 1).contiguous()  # size: [3, num_heads, head_dim]
    query, key, value = qkv_bias[0], qkv_bias[1], qkv_bias[2]  # size: [num_heads, head_dim]

    query = query.reshape(-1)  # size: [num_heads * head_dim]
    key = key.reshape(-1)  # size: [num_heads * head_dim]
    value = value.reshape(-1)  # size: [num_heads * head_dim]

    return query, key, value
