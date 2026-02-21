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

"""Tests for DataCollatorWithFlattening and TokenPackingDataset for causal LM (Llama3)."""

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from collator import (
    DataCollatorWithFlattening,
    TokenPackingDataset,
    _split_sample_by_num_tokens,
)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the Llama3 nucleotide tokenizer for testing."""
    tokenizer_path = Path(__file__).parent.parent / "nucleotide_fast_tokenizer"
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_data_collator_with_flattening_causal_lm(tokenizer):
    """Test DataCollatorWithFlattening for causal LM with separator_id=-100."""
    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collator = DataCollatorWithFlattening(collator=base_collator, separator_id=-100)

    features = [
        {"input_ids": [1, 10, 20, 30, 2]},  # 5 tokens
        {"input_ids": [1, 40, 50, 60, 70, 2]},  # 6 tokens
        {"input_ids": [1, 80, 90, 2]},  # 4 tokens
    ]

    total_tokens = sum(len(f["input_ids"]) for f in features)
    batch = collator(features, return_tensors="pt")

    # Check shapes
    assert batch["input_ids"].shape == (1, total_tokens)
    assert batch["labels"].shape == (1, total_tokens)

    # Check cu_seq_lens
    expected_cu_seqlens = torch.tensor([0, 5, 11, 15], dtype=torch.int32)
    torch.testing.assert_close(batch["cu_seq_lens_q"], expected_cu_seqlens)
    torch.testing.assert_close(batch["cu_seq_lens_k"], expected_cu_seqlens)

    # Check separator_id=-100 at sequence boundaries in labels
    assert batch["labels"][0, 5].item() == -100
    assert batch["labels"][0, 11].item() == -100


def test_data_collator_pad_to_multiple_of(tokenizer):
    """Test DataCollatorWithFlattening with pad_to_multiple_of."""
    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collator = DataCollatorWithFlattening(collator=base_collator, separator_id=-100, pad_to_multiple_of=8)

    features = [
        {"input_ids": [1, 10, 20, 30, 2]},  # 5 tokens
        {"input_ids": [1, 40, 50, 2]},  # 4 tokens
    ]

    batch = collator(features, return_tensors="pt")

    # Total is 9 tokens, padded to 16
    assert batch["input_ids"].numel() % 8 == 0
    assert batch["input_ids"].numel() == 16


def test_token_packing_dataset_basic():
    """Test basic TokenPackingDataset functionality."""

    class MockDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.arange(40)}
            yield {"input_ids": torch.arange(40)}
            yield {"input_ids": torch.arange(30)}

    dataset = MockDataset()
    packing = TokenPackingDataset(dataset, max_tokens_per_batch=100, drop_last=False)
    batches = list(packing)

    assert len(batches) == 2
    assert sum(len(s["input_ids"]) for s in batches[0]) == 80
    assert sum(len(s["input_ids"]) for s in batches[1]) == 30


def test_split_sample_by_num_tokens_basic():
    """Test _split_sample_by_num_tokens with basic input."""
    sample = {"input_ids": [1, 10, 20, 30, 40, 50, 2]}
    first, remaining = _split_sample_by_num_tokens(sample, 3)

    assert first["input_ids"] == [1, 10, 20]
    assert remaining["input_ids"] == [30, 40, 50, 2]


def test_split_sample_by_num_tokens_errors():
    """Test _split_sample_by_num_tokens raises errors for invalid inputs."""
    sample = {"input_ids": [1, 10, 20, 2]}

    with pytest.raises(ValueError, match="must be less than sample length"):
        _split_sample_by_num_tokens(sample, 4)

    with pytest.raises(ValueError, match="must be positive"):
        _split_sample_by_num_tokens(sample, 0)
