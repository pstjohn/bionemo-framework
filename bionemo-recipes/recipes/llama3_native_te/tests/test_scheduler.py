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

"""Tests for get_cosine_annealing_schedule_with_warmup."""

import pytest
import torch

from scheduler import get_cosine_annealing_schedule_with_warmup


@pytest.fixture
def optimizer():
    """Create a dummy optimizer for scheduler testing."""
    model = torch.nn.Linear(2, 2)
    return torch.optim.SGD(model.parameters(), lr=1.0)


def test_step_zero(optimizer):
    """Step 0 should have lr=0."""
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=100, num_decay_steps=1000)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)


def test_end_of_warmup(optimizer):
    """End of warmup should have lr=1.0."""
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, num_warmup_steps=100, num_decay_steps=1000)
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0, abs=1e-6)


def test_mid_decay_default_min_lr(optimizer):
    """Mid-decay with min_lr_ratio=0 should be ~0.5."""
    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_decay_steps=1000, min_lr_ratio=0.0
    )
    mid_step = 100 + 500  # warmup + half of decay
    for _ in range(mid_step):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.5, abs=1e-2)


def test_end_of_decay(optimizer):
    """At end of decay, lr should be min_lr_ratio."""
    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_decay_steps=1000, min_lr_ratio=0.0
    )
    for _ in range(1100):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-6)


def test_past_decay(optimizer):
    """Past the decay period, lr should stay at min_lr_ratio."""
    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_decay_steps=1000, min_lr_ratio=0.0
    )
    for _ in range(1200):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-6)


def test_with_min_lr_ratio(optimizer):
    """With min_lr_ratio=0.1, lr should not go below 0.1."""
    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_decay_steps=1000, min_lr_ratio=0.1
    )
    # Step to end of decay
    for _ in range(1100):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1, abs=1e-6)

    # Step past decay
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1, abs=1e-6)


def test_monotonically_decreasing_after_warmup(optimizer):
    """LR should be monotonically decreasing after warmup."""
    num_warmup = 100
    num_decay = 1000
    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_decay_steps=num_decay
    )
    for _ in range(num_warmup):
        optimizer.step()
        scheduler.step()

    prev_lr = scheduler.get_last_lr()[0]
    for _ in range(num_decay):
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr <= prev_lr + 1e-9
        prev_lr = current_lr
