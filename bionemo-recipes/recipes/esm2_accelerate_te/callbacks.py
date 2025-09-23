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

import logging
import time
from collections import deque

import numpy as np
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


logger = logging.getLogger(__name__)


class StopAfterNStepsCallback(TrainerCallback):
    """Callback to interrupt training after a specified number of steps.

    This allows us to use a learning rate scheduler consistent with the full training run while
    stopping after a pre-determined number of steps.
    """

    def __init__(self, max_steps: int):
        """Initialize the callback."""
        self.max_steps = max_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Interrupt training after a specified number of steps."""
        if state.global_step >= self.max_steps:
            control.should_training_stop = True


class StepTimingCallback(TrainerCallback):
    """Callback to log the time taken for each step."""

    def __init__(self, max_step_times: int = 100):
        """Initialize the callback.

        Args:
            max_step_times: The maximum number of step times to store for the final mean step time calculation.
        """
        self.step_times = deque(maxlen=max_step_times)
        self.tokens_seen = deque(maxlen=max_step_times)

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        self.step_times.append(time.perf_counter())
        self.tokens_seen.append(state.num_input_tokens_seen)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged."""
        if len(self.step_times) > 1 and logs is not None:
            logs["train/step_time"] = self.step_times[-1] - self.step_times[-2]
            logs["train/tokens_per_step"] = self.tokens_seen[-1] - self.tokens_seen[-2]
            logs["train/tokens_per_second"] = logs["train/tokens_seen"] / logs["train/step_time"]

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if len(self.step_times) > 1 and state.is_world_process_zero:
            logger.info(
                "Mean step time (last %d steps): %f seconds", len(self.step_times), np.diff(self.step_times).mean()
            )
            logger.info(
                "Mean tokens per second (last %d steps): %.3g",
                len(self.tokens_seen),
                (np.diff(self.tokens_seen) / np.diff(self.step_times)).mean(),
            )
