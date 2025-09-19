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
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Class to track distributed ranks and handle basic distributed training setup.

    Attributes:
        rank: The rank of the process.
        local_rank: The local rank of the process.
        world_size: The total number of processes.
    """

    rank: int = field(default_factory=lambda: dist.get_rank() if dist.is_initialized() else 0)
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: dist.get_world_size() if dist.is_initialized() else 1)

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0

    def __post_init__(self):
        """Post-initialization setup for distributed training.

        If the distributed environment is not initialized, we set the environment variables for single process
        fallback (direct python execution) for debugging.
        """
        # Single process fallback (direct python execution) for debugging.
        logger.warning("Running in single-process mode. Use torchrun for distributed training.")
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # Initialize the distributed process group.
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Set the device id.
        torch.cuda.set_device(self.local_rank)

        # Log the distributed configuration.
        logger.info("Initialized distributed training: %s", self)

    def __del__(self):
        """Clean up the distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
