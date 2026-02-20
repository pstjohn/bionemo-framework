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

import gc
import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from transformer_engine.common.recipe import Format

from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp, save_final_model_ddp, should_save_checkpoint
from collator import ContextParallelDataLoaderWrapper, DataCollatorForContextParallel
from dataset import create_bshd_dataloader, create_mock_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity_ddp_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Llama3 with TE layers using DDP + Context Parallelism for benchmarking.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Create a 2D device mesh for DDP + CP.
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size // args.cp_size, args.cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    logger.info(f"Created device mesh: {device_mesh}")

    # Create an FP8 recipe -- this is only used if FP8 is enabled in the config.
    fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
        fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
    )

    # Create an empty Llama3 model with a causal language model head.
    config = NVLlamaConfig.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **args.config_kwargs)

    with transformer_engine.pytorch.quantized_model_init(
        recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
    ):
        model = NVLlamaForCausalLM(config)

    logger.info("Initialized Model:\n%s", model)

    model = model.to(device=device)

    # Attach the CP group to the model layers before wrapping with DDP.
    for layer in model.model.layers:
        layer.set_context_parallel_group(
            device_mesh["cp"].get_group(),
            torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
            torch.cuda.Stream(),
        )

    # Wrap with DDP for data parallelism across the dp mesh dimension.
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        device_mesh=device_mesh["dp"],
    )

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        model = torch.compile(model)

    # Create the context-aware dataloader. We only create the dataloader on cp rank 0 and wrap it in a
    # ContextParallelDataLoaderWrapper that will shard and distribute the data across the context parallelism group.
    if args.dataset.get("pad_sequences_to_be_divisible_by", None) is None:
        logger.info("pad_sequences_to_be_divisible_by is not provided, using cp_mesh.size() * 2")
        OmegaConf.update(args, "dataset.pad_sequences_to_be_divisible_by", device_mesh["cp"].size() * 2)

    if device_mesh["cp"].get_local_rank() == 0:
        if args.use_mock_dataset:
            train_dataloader, dataset_or_sampler = create_mock_dataloader(
                dist_config, vocab_size=config.vocab_size, **args.dataset
            )
        elif args.use_sequence_packing:
            train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
        else:
            train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

        train_dataloader.collate_fn = DataCollatorForContextParallel(
            collator=train_dataloader.collate_fn,
            device_mesh=device_mesh,
            qkv_format=args.config_kwargs.attn_input_format,
            is_causal_lm=True,
        )
    else:
        train_dataloader = None
        dataset_or_sampler = None

    train_dataloader = ContextParallelDataLoaderWrapper(train_dataloader, device_mesh["cp"])

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_ddp_cp" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_ddp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
        )
    else:
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    gc.collect()
    torch.cuda.empty_cache()

    # Training loop
    step = start_step
    micro_step = 0  # Gradient accumulation step counter
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1
            # Use no_sync to prevent gradient synchronization until the last microbatch
            with model.no_sync() if micro_step % args.grad_acc_steps != 0 else nullcontext():
                # Forward pass with mixed precision.
                with transformer_engine.pytorch.autocast(enabled=args.fp8_config.enabled, recipe=fp8_recipe):
                    outputs = model(**batch)

                # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
                loss = outputs.loss / args.grad_acc_steps
                loss.backward()

                # Log microbatch step data for accumulation metrics
                perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

            # Gradient accumulation - only step optimizer after accumulating gradients
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                # Compute and clip gradient norms.
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

                # Step optimizer.
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                perf_logger.log_step(
                    step=step,
                    grad_norm=total_norm,
                    lr=optimizer.param_groups[0]["lr"],
                )

                if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                    save_checkpoint_ddp(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ckpt_path=ckpt_path,
                        step=step,
                        epoch=epoch,
                        dist_config=dist_config,
                        dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
                        max_checkpoints=args.checkpoint.max_checkpoints,
                    )

                step += 1
                if step >= args.num_train_steps:
                    break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        if dataset_or_sampler is not None:
            dataset_or_sampler.set_epoch(epoch)

    # Save final model to a .safetensors file.
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_ddp(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Clean up distributed training. Must close the dataloader wrapper before destroying the process group
    # to avoid hangs from in-flight prefetch threads.
    train_dataloader.close()
    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
