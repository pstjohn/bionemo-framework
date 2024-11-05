# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from typing import Callable, Optional, Sequence, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from torch import Tensor


__all__: Sequence[str] = ("ESM2DotProductAttention", "ESM2TEDotProductAttention")


class ESM2TEDotProductAttention(TEDotProductAttention):
    """ESM2-Specific transformer engine core attention.

    Override the softmax_scale to 1.0 to match the ESM2 implementation while keeping the rest from the original TEDotProductAttention.
    """

    def __init__(
        self,
        *args,
        softmax_scale: float = 1.0,
        **kwargs,
    ):
        """Initialize ESM2TEDotProductAttention."""
        if not is_te_min_version("1.10.0"):
            raise NotImplementedError(
                "We can reuse __init__ only if TE version is greater than 1.10.0. Otherwise softmax_scale will be ignored. See https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/extensions/transformer_engine.py#L620"
            )

        super().__init__(*args, softmax_scale=softmax_scale, **kwargs)


class ESM2DotProductAttention(DotProductAttention):
    """ESM2-Specific core attention.

    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
    ) -> None:
        """Initializes the Attention class.

        Args:
            config: The configuration object for the transformer.
            layer_number: The layer number of the attention module.
            attn_mask_type: The type of attention mask to be used.
            attention_type: The type of attention mechanism.
            attention_dropout: The dropout rate for attention weights. Defaults to None.
        """
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: Optional[AttnMaskType] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Forward pass of the ESM2DotProductAttention module.

        Args:
            query: The query tensor of shape [sq, b, np, hn].
            key: The key tensor of shape [sk, b, ng, hn].
            value: The value tensor of shape [sk, b, ng, hn].
            attention_mask: The attention mask tensor of shape [b, np, sq, sk].
            attn_mask_type: The attention mask type, currently unused. Defaults to None.
            packed_seq_params: The packed sequence parameters. These are used for context parallelism so will be needed
                to be implemented if we want to support this. Defaults to None.

        Returns:
            Tensor: The context tensor of shape [sq, b, hp].
        """
        if packed_seq_params is not None:
            raise ValueError(
                "Packed sequence is not supported by DotProductAttention. " "Please use TEDotProductAttention instead."
            )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if (np_ng := self.num_attention_heads_per_partition // self.num_query_groups_per_partition) > 1:
            key = key.repeat_interleave(np_ng, dim=2)
            value = value.repeat_interleave(np_ng, dim=2)

        # [b, np, sq, sk]
        b, np, sq, sk = query.size(1), query.size(2), query.size(0), key.size(0)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(sq, b * np, -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(sk, b * np, -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (b * np, sq, sk),
            query.dtype,
            "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor) if self.config.normalize_attention_scores else 1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(b, np, sq, sk)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        # ESM2 Customization
        if self.config.use_esm_attention:
            # NOTE: the slicing here is to make the attention_mask the same shape as the extended
            # attention mask in ESM2. The multiplication by -3.4028e+38 (float32 min_val) is
            # similarly motivated by ESM2's masking approach, which forces softmax of attention scores
            # for masked entries to be close to 0. This number is replaced with min_val of the precision
            # using min_val instead of -inf is stable in an special case where all sequence is masked
            min_val = torch.finfo(attention_scores.dtype).min

            attention_probs: Tensor = self.esm2_scale_mask_softmax(
                attention_scores.masked_fill(attention_mask[:, :, 0:1, :].to(bool), min_val)
            )
        # END ESM2 Customization
        else:
            attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        b, np, sq, hn = value.size(1), value.size(2), query.size(0), value.size(3)

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), b * np, -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(b * np, sq, -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(b, np, sq, hn)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context = context.view(sq, b, self.hidden_size_per_partition)

        return context

    def esm2_scale_mask_softmax(
        self,
        input: Tensor,
        mask: Optional[Tensor] = None,
        scale: Optional[Union[float, int]] = None,
        mask_func: Optional[Callable] = None,
    ) -> Tensor:
        """Scale Mask Softmax function.

        Args:
            input: Tensor of shape (Batch, NP, SK, SQ). The input may or may not have already
                had a mask applied to it.
            mask: If a mask is to be applied, it will go here.
            scale: A scale factor that will be applied before the softmax.
            mask_func: An optional function to apply to the mask. If None, it is assumed that
                the input already had the mask applied to it.

        Returns:
            probs: Tensor of normalized probabilities after the softmax has been applied,
                of shape (Batch, NP, SK, SQ).
        """
        if self.attn_mask_type.name != "padding":
            raise ValueError(
                f"self.attn_mask_type: {self.attn_mask_type} is not 'padding'. "
                "Only 'padding' type is supported currently."
            )

        original_dtype = input.dtype  # Store original dtype
        if (
            original_dtype == torch.float16 or original_dtype == torch.bfloat16
        ) and self.config.attention_softmax_in_fp32:
            input = input.float()  # Convert to float32 for softmax

        if scale is not None:
            input = input * scale  # Apply scaling

        if mask is not None and mask_func is not None:
            input = mask_func(input, mask)  # Apply mask function if provided

        probs = torch.nn.functional.softmax(input, dim=-1)  # Apply softmax

        if self.config.attention_softmax_in_fp32 and original_dtype in (torch.float16, torch.bfloat16):
            probs = probs.to(original_dtype)  # Convert back to original dtype if necessary

        return probs
