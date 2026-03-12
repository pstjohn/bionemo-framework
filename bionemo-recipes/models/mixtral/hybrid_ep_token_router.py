# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""DeepEP-backed token dispatchers for expert-parallel MoE communication.

Two dispatcher classes are provided, both implementing the ``TokenDispatcher``
protocol from ``modeling_mixtral_te``:

* ``HybridEPTokenDispatcher`` - Uses ``deep_ep.HybridEPBuffer`` with fused
  permute+dispatch CUDA kernels.  Works today (no NVSHMEM needed).
* ``DeepEPBufferTokenDispatcher`` - Uses ``deep_ep.Buffer`` with lower-latency
  intranode NVLink kernels.  Requires NVSHMEM to be enabled at build time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from modeling_mixtral_te import DispatchOutput


# ---------------------------------------------------------------------------
# HybridEPBuffer backend (no NVSHMEM dependency)
# ---------------------------------------------------------------------------


@dataclass
class _HybridEPHandle:
    """Opaque handle for HybridEPTokenDispatcher, storing state between dispatch and combine."""

    deep_ep_handle: tuple
    dispatched_probs: torch.Tensor


class HybridEPTokenDispatcher:
    """TokenDispatcher using DeepEP ``HybridEPBuffer`` for expert-parallel communication.

    Uses fused permute+dispatch CUDA kernels with auto-tuned SM allocation.
    Best for large batches and multi-node EP.

    Args:
        num_experts: Total number of experts (global).
        num_local_experts: Number of experts on this rank.
        hidden_size: Hidden dimension size.
        ep_size: Expert parallel world size.
        max_tokens_per_rank: Maximum number of tokens per rank for buffer allocation.
    """

    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        ep_size: int,
        max_tokens_per_rank: int = 512,
    ):
        """Initialize the HybridEPTokenDispatcher."""
        del ep_size  # Derived from the process group in set_ep_group()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self._buffer: Any = None

    def set_ep_group(self, ep_group: dist.ProcessGroup) -> None:
        """Create the ``HybridEPBuffer`` for the given expert-parallel process group."""
        from deep_ep import HybridEPBuffer

        self._buffer = HybridEPBuffer(
            group=ep_group,
            hidden_dim=self.hidden_size,
            max_num_of_tokens_per_rank=self.max_tokens_per_rank,
            num_local_experts=self.num_local_experts,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> DispatchOutput:
        """Dispatch tokens to their assigned experts via fused permute+dispatch.

        Args:
            hidden_states: Flattened input tensor of shape ``[N, H]``.
            selected_experts: Expert assignments, shape ``[N, top_k]``, int.
            routing_weights: Normalized routing probabilities, shape ``[N, top_k]``, float32.

        Returns:
            DispatchOutput with expert-sorted tokens, per-expert counts, and an opaque handle.
        """
        assert self._buffer is not None, "EP group must be set via set_ep_group() before dispatch"

        topk_idx = selected_experts.to(torch.int64)

        dispatched_token, dispatched_probs, _dispatched_scaling_factor, tokens_per_expert, handle = (
            self._buffer.dispatch_with_permute(
                hidden=hidden_states,
                topk_idx=topk_idx,
                topk_weights=routing_weights,
                num_of_experts=self.num_experts,
                num_of_experts_per_rank=self.num_local_experts,
            )
        )

        # tokens_per_expert is a CPU tensor (stream already synced when non_blocking=False)
        tokens_per_expert_list = tokens_per_expert.tolist()

        return DispatchOutput(
            expert_input=dispatched_token,
            tokens_per_expert=tokens_per_expert_list,
            handle=_HybridEPHandle(deep_ep_handle=handle, dispatched_probs=dispatched_probs),
        )

    def combine(self, expert_output: torch.Tensor, handle: _HybridEPHandle) -> torch.Tensor:
        """Combine expert outputs back to the original token order.

        Args:
            expert_output: Expert output tensor of shape ``[total_recv_tokens, H]``.
            handle: Handle from ``dispatch()`` containing state for the reverse operation.

        Returns:
            Combined output tensor of shape ``[N, H]`` with routing weights applied.
        """
        # combine_with_unpermute performs addition WITHOUT applying routing weights,
        # so we pre-multiply expert outputs by the dispatched routing probabilities.
        weighted_output = expert_output * handle.dispatched_probs.unsqueeze(1).to(expert_output.dtype)
        combined_token, _combined_probs = self._buffer.combine_with_unpermute(
            hidden=weighted_output,
            handle=handle.deep_ep_handle,
        )
        return combined_token


# ---------------------------------------------------------------------------
# Buffer backend (requires NVSHMEM)
# ---------------------------------------------------------------------------


def _local_permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand and sort tokens by local expert using a multihot routing map.

    Tokens assigned to multiple local experts are duplicated. The output is
    sorted by expert (all tokens for expert 0, then expert 1, etc.).

    Args:
        tokens: Unique received tokens, shape ``[N_unique, H]``.
        routing_map: Boolean map, shape ``[N_unique, num_local_experts]``.
        probs: Routing weights in multihot layout, shape ``[N_unique, num_local_experts]``, float32.

    Returns:
        ``(expanded_tokens, permuted_probs, sorted_indices)`` where
        ``sorted_indices`` maps expanded rows back to unique-token rows.
    """
    num_local_experts = routing_map.shape[1]
    num_unique = tokens.shape[0]
    token_ids = torch.arange(num_unique, device=tokens.device).unsqueeze(0).expand(num_local_experts, -1)
    sorted_indices = token_ids.masked_select(routing_map.T.contiguous())
    permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
    return tokens[sorted_indices], permuted_probs, sorted_indices


def _local_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_unique: int,
) -> torch.Tensor:
    """Reverse ``_local_permute``: scatter-add expanded tokens back to unique positions.

    Args:
        permuted_tokens: Expert outputs (already weighted), shape ``[N_expanded, H]``.
        sorted_indices: Mapping from ``_local_permute``, shape ``[N_expanded]``.
        num_unique: Number of unique received tokens.

    Returns:
        Contracted tensor of shape ``[N_unique, H]``.
    """
    hidden = permuted_tokens.shape[1]
    output = torch.zeros(num_unique, hidden, device=permuted_tokens.device, dtype=permuted_tokens.dtype)
    output.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output


@dataclass
class _BufferHandle:
    """Opaque handle for DeepEPBufferTokenDispatcher, storing state between dispatch and combine."""

    deep_ep_handle: tuple
    recv_topk_weights: torch.Tensor
    permuted_probs: torch.Tensor
    sorted_indices: torch.Tensor
    num_unique: int


class DeepEPBufferTokenDispatcher:
    """TokenDispatcher using DeepEP ``Buffer`` for expert-parallel communication.

    Uses lower-latency intranode NVLink kernels. Requires NVSHMEM to be enabled
    at build time.

    ``Buffer.dispatch`` returns *unique* received tokens — one copy per token
    regardless of how many local experts it is routed to.  A local
    permute/unpermute step (``_local_permute`` / ``_local_unpermute``) expands
    tokens for multi-expert assignments and contracts them back after the
    expert FFN, following the pattern used in NeMo Automodel.

    Args:
        num_experts: Total number of experts (global).
        num_local_experts: Number of experts on this rank.
        hidden_size: Hidden dimension size.
        ep_size: Expert parallel world size.
    """

    def __init__(self, num_experts: int, num_local_experts: int, hidden_size: int, ep_size: int):
        """Initialize the DeepEPBufferTokenDispatcher."""
        del ep_size  # Derived from the process group in set_ep_group()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self._buffer: Any = None
        self._ep_rank: int = 0

    def set_ep_group(self, ep_group: dist.ProcessGroup) -> None:
        """Create the ``Buffer`` for the given expert-parallel process group.

        Computes buffer sizes via ``Config.get_nvl_buffer_size_hint`` using the
        dispatch and combine configs for the given EP world size.
        """
        from deep_ep import Buffer

        self._ep_rank = dist.get_rank(ep_group)
        group_size = ep_group.size()
        # hidden_size is in elements; Buffer uses bf16 so 2 bytes per element
        hidden_bytes = self.hidden_size * 2

        num_nvl_bytes = 0
        for config in (Buffer.get_dispatch_config(group_size), Buffer.get_combine_config(group_size)):
            num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group_size), num_nvl_bytes)

        self._buffer = Buffer(group=ep_group, num_nvl_bytes=num_nvl_bytes)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> DispatchOutput:
        """Dispatch tokens to their assigned experts via Buffer intranode kernels.

        Args:
            hidden_states: Flattened input tensor of shape ``[N, H]``.
            selected_experts: Expert assignments, shape ``[N, top_k]``, int.
            routing_weights: Normalized routing probabilities, shape ``[N, top_k]``, float32.

        Returns:
            DispatchOutput with expert-sorted tokens, per-expert counts, and an opaque handle.
        """
        assert self._buffer is not None, "EP group must be set via set_ep_group() before dispatch"

        topk_idx = selected_experts.to(torch.int64)

        # Phase 1: compute layout
        num_tokens_per_rank, _num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _event = (
            self._buffer.get_dispatch_layout(topk_idx, self.num_experts)
        )

        # Phase 2: inter-rank dispatch (returns unique tokens)
        recv_x, recv_topk_idx, recv_topk_weights, _tokens_per_expert_list, handle, _event = self._buffer.dispatch(
            hidden_states,
            topk_idx=topk_idx,
            topk_weights=routing_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        # Phase 3: local permute — expand unique tokens for multi-local-expert assignments.
        # recv_topk_idx uses LOCAL expert indices (0-based per rank), with -1 for non-local experts.
        # Convert to multihot routing_map [N_unique, num_local_experts] and probs_map.
        num_unique = recv_x.shape[0]
        routing_map = torch.zeros(num_unique, self.num_local_experts, device=recv_x.device, dtype=torch.bool)
        probs_map = torch.zeros(num_unique, self.num_local_experts, device=recv_x.device, dtype=torch.float32)
        for k in range(recv_topk_idx.shape[1]):
            col = recv_topk_idx[:, k]
            valid = (col >= 0) & (col < self.num_local_experts)
            if valid.any():
                rows = torch.arange(num_unique, device=recv_x.device)[valid]
                routing_map[rows, col[valid]] = True
                probs_map[rows, col[valid]] = recv_topk_weights[valid, k]

        expanded_tokens, permuted_probs, sorted_indices = _local_permute(recv_x, routing_map, probs_map)
        tokens_per_expert = routing_map.sum(dim=0).int().tolist()

        return DispatchOutput(
            expert_input=expanded_tokens,
            tokens_per_expert=tokens_per_expert,
            handle=_BufferHandle(
                deep_ep_handle=handle,
                recv_topk_weights=recv_topk_weights,
                permuted_probs=permuted_probs,
                sorted_indices=sorted_indices,
                num_unique=num_unique,
            ),
        )

    def combine(self, expert_output: torch.Tensor, handle: _BufferHandle) -> torch.Tensor:
        """Combine expert outputs back to the original token order.

        Args:
            expert_output: Expert output tensor of shape ``[total_recv_tokens, H]``.
            handle: Handle from ``dispatch()`` containing state for the reverse operation.

        Returns:
            Combined output tensor of shape ``[N, H]`` with routing weights applied.
        """
        # Pre-multiply expert outputs by per-token routing probabilities.
        weighted = expert_output * handle.permuted_probs.unsqueeze(1).to(expert_output.dtype)

        # Local unpermute: scatter-add weighted outputs back to unique-token positions.
        contracted = _local_unpermute(weighted, handle.sorted_indices, handle.num_unique)

        # Inter-rank combine (addition without weights).
        recv_x, _recv_topk_weights, _event = self._buffer.combine(
            contracted,
            handle.deep_ep_handle,
            topk_weights=handle.recv_topk_weights,
        )
        return recv_x
