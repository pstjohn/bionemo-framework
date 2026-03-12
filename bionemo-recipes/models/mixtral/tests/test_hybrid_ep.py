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

"""Tests for DeepEP-backed token dispatchers in the Mixtral MoE model.

Verifies that both DeepEP dispatchers (HybridEPBuffer and Buffer backends) produce
the same logits and loss as the AllToAllTokenDispatcher when running with EP=2.
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def _deep_ep_available() -> bool:
    """Check if the deep_ep package and hybrid_ep_cpp extension are importable."""
    try:
        import deep_ep  # noqa: F401
        import hybrid_ep_cpp  # noqa: F401

        return True
    except ImportError:
        return False


def _cuda_peer_access_available() -> bool:
    """Check if CUDA peer access (NVLink IPC) is supported between GPU 0 and GPU 1."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    return torch.cuda.can_device_access_peer(0, 1)


requires_deep_ep = pytest.mark.skipif(not _deep_ep_available(), reason="deep_ep or hybrid_ep_cpp not available")

requires_peer_access = pytest.mark.skipif(
    not _cuda_peer_access_available(),
    reason="CUDA peer access (NVLink IPC) not supported between GPUs",
)


def _create_small_mixtral_config(**overrides) -> NVMixtralConfig:
    """Create a small Mixtral config suitable for testing."""
    defaults = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
        "attn_input_format": "bshd",
        "self_attn_mask_type": "causal",
        "router_jitter_noise": 0.0,
    }
    defaults.update(overrides)
    return NVMixtralConfig(**defaults)


def _get_dummy_batch(vocab_size: int, seq_len: int = 32, batch_size: int = 2, device: str = "cuda"):
    """Create a simple dummy batch for testing."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass(frozen=True)
class DistributedConfig:
    """Distributed environment configuration."""

    rank: int = field(default_factory=lambda: int(os.environ.setdefault("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.environ.setdefault("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.setdefault("WORLD_SIZE", "1")))
    _master_addr: str = field(default_factory=lambda: os.environ.setdefault("MASTER_ADDR", "localhost"))
    _master_port: str = field(default_factory=lambda: os.environ.setdefault("MASTER_PORT", "12355"))

    def is_main_process(self) -> bool:
        """Return True if this is the global rank 0 process."""
        return self.rank == 0


def _shard_expert_weights(full_state_dict: dict, ep_rank: int, ep_size: int, num_experts: int) -> dict:
    """Shard stacked expert weights from a full (EP=1) state dict for a given EP rank.

    Expert weight keys are ``...experts_gate_up_weight`` and ``...experts_down_weight``
    with shape ``[num_experts, ...]``. For EP, each rank keeps only its local slice.
    """
    experts_per_rank = num_experts // ep_size
    start_expert = ep_rank * experts_per_rank
    end_expert = start_expert + experts_per_rank

    new_state_dict = {}
    for key, value in full_state_dict.items():
        if key.endswith("experts_gate_up_weight") or key.endswith("experts_down_weight"):
            new_state_dict[key] = value[start_expert:end_expert]
        else:
            new_state_dict[key] = value

    return new_state_dict


# ---------------------------------------------------------------------------
# Pytest entry points — launch torchrun subprocesses
# ---------------------------------------------------------------------------


def _run_torchrun(backend_name: str, port: int):
    """Run the equivalence test worker via torchrun with 2 GPUs."""
    model_dir = str(Path(__file__).resolve().parent.parent)
    script = str(Path(__file__).resolve())
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{port}",
        script,
        backend_name,
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=model_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"DeepEP {backend_name} equivalence test failed with exit code {result.returncode}")


@requires_multi_gpu
@requires_deep_ep
@requires_peer_access
def test_hybrid_ep_matches_alltoall(unused_tcp_port):
    """Test that HybridEPBuffer dispatcher matches AllToAll dispatcher at EP=2."""
    _run_torchrun("hybrid_ep", unused_tcp_port)


@requires_multi_gpu
@requires_deep_ep
def test_buffer_matches_alltoall(unused_tcp_port):
    """Test that Buffer dispatcher matches AllToAll dispatcher at EP=2.

    Skipped inside the worker if NVSHMEM is not available (Buffer constructor fails).
    """
    _run_torchrun("buffer", unused_tcp_port)


# ---------------------------------------------------------------------------
# Distributed worker executed via torchrun
# ---------------------------------------------------------------------------


def _run_equivalence_test(backend: str):
    """Main worker function for the DeepEP equivalence test.

    1. Init distributed with 2 GPUs.
    2. Create EP=1 model for reference weights.
    3. Create EP=2 model with AllToAll dispatcher → reference logits/loss.
    4. Create EP=2 model with selected DeepEP dispatcher → test logits/loss.
    5. Compare results.
    """
    from torch.distributed.tensor.device_mesh import DeviceMesh

    from hybrid_ep_token_router import DeepEPBufferTokenDispatcher, HybridEPTokenDispatcher

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    ep_rank = dist_config.rank
    ep_size = dist_config.world_size

    # --- Phase 1: Create EP=1 model for reference weights ---
    config_ep1 = _create_small_mixtral_config(expert_parallel_size=1)
    torch.manual_seed(0)
    model_ep1 = NVMixtralForCausalLM(config_ep1).to(dtype=torch.bfloat16, device=device)
    full_state_dict = {k: v.clone().cpu() for k, v in model_ep1.state_dict().items()}
    del model_ep1
    torch.cuda.empty_cache()

    batch = _get_dummy_batch(config_ep1.vocab_size, seq_len=32, batch_size=2, device=device)
    num_experts = config_ep1.num_local_experts

    # --- Phase 2: EP=2 + AllToAll dispatcher → reference logits/loss ---
    config_ep2 = _create_small_mixtral_config(expert_parallel_size=ep_size)
    torch.manual_seed(0)
    model_alltoall = NVMixtralForCausalLM(config_ep2).to(dtype=torch.bfloat16, device=device)

    sharded_state = _shard_expert_weights(full_state_dict, ep_rank, ep_size, num_experts)
    model_alltoall.load_state_dict(sharded_state, strict=False)
    model_alltoall.eval()

    ep_mesh = DeviceMesh("cuda", list(range(ep_size)))
    ep_group = ep_mesh.get_group()
    model_alltoall.model.set_ep_groups(ep_group, ep_mesh)

    with torch.no_grad():
        outputs_ref = model_alltoall(**batch)
    logits_ref = outputs_ref.logits.detach().clone().cpu()
    loss_ref = outputs_ref.loss.detach().clone().cpu()

    del model_alltoall, outputs_ref
    torch.cuda.empty_cache()

    # --- Phase 3: EP=2 + DeepEP dispatcher → test logits/loss ---
    num_local_experts = num_experts // ep_size
    hidden_size = config_ep2.hidden_size

    if backend == "hybrid_ep":
        make_dispatcher = lambda: HybridEPTokenDispatcher(  # noqa: E731
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            ep_size=ep_size,
        )
    elif backend == "buffer":
        # Check if Buffer can be created (requires NVSHMEM)
        try:
            from deep_ep import Buffer

            Buffer(group=ep_group, num_nvl_bytes=0)
        except Exception as e:
            if dist_config.is_main_process():
                print(f"SKIP: Buffer backend not available (NVSHMEM disabled): {e}")
            torch.distributed.destroy_process_group()
            return

        make_dispatcher = lambda: DeepEPBufferTokenDispatcher(  # noqa: E731
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            ep_size=ep_size,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    torch.manual_seed(0)
    model_deepep = NVMixtralForCausalLM(config_ep2).to(dtype=torch.bfloat16, device=device)
    model_deepep.load_state_dict(sharded_state, strict=False)
    model_deepep.eval()

    # Set dispatchers BEFORE setting EP groups (set_ep_groups calls dispatcher.set_ep_group())
    model_deepep.model.set_dispatchers(make_dispatcher)
    model_deepep.model.set_ep_groups(ep_group, ep_mesh)

    with torch.no_grad():
        outputs_test = model_deepep(**batch)
    logits_test = outputs_test.logits.detach().cpu()
    loss_test = outputs_test.loss.detach().cpu()

    # --- Phase 4: Compare on rank 0 ---
    if dist_config.is_main_process():
        torch.testing.assert_close(
            logits_test,
            logits_ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"DeepEP {backend} logits do not match AllToAll logits",
        )

        torch.testing.assert_close(
            loss_test,
            loss_ref,
            atol=1e-3,
            rtol=1e-3,
            msg=f"DeepEP {backend} loss does not match AllToAll loss",
        )

        print(f"DeepEP {backend} equivalence test PASSED: logits and loss match AllToAll")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    backend_name = sys.argv[1]
    _run_equivalence_test(backend_name)
