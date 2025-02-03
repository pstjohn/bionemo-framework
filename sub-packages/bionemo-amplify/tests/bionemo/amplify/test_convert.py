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


import gc
from pathlib import Path

import torch
from megatron.core.transformer.module import Float16Module
from nemo.lightning import io
from transformers import AutoModel

from bionemo.amplify.convert import HFAMPLIFYImporter  # noqa: F401
from bionemo.amplify.model import AMPLIFYConfig
from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.testing.compare import assert_cosine_similarity, get_input_tensors
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.testing import megatron_parallel_state_utils


def assert_amplify_equivalence(
    ckpt_path: str,
    model_tag: str,
    precision: PrecisionTypes = "fp32",
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    tokenizer = BioNeMoAMPLIFYTokenizer()

    input_ids, attention_mask = get_input_tensors(tokenizer)
    hf_logits, hf_hidden_state, hf_attn_inputs, hf_attn_outputs = load_and_evaluate_hf_amplify(
        model_tag, precision, input_ids, attention_mask
    )
    gc.collect()
    torch.cuda.empty_cache()
    nemo_logits, nemo_hidden_state, nemo_attn_inputs, nemo_attn_outputs = load_and_evaluate_nemo_amplify(
        tokenizer,
        ckpt_path,
        precision,
        input_ids,
        attention_mask,
    )

    # Rather than directly comparing the logit or hidden state tensors, we compare their cosine similarity. These
    # should be essentially 1 if the outputs are equivalent, but is less sensitive to small numerical differences.
    # We don't care about the padding tokens, so we only compare the non-padding tokens.
    assert_cosine_similarity(nemo_attn_inputs[0].transpose(0, 1), hf_attn_inputs[0], attention_mask, msg="Attn inputs")
    assert_cosine_similarity(
        nemo_attn_outputs[0].transpose(0, 1), hf_attn_outputs[0], attention_mask, msg="Attn inputs"
    )

    assert_cosine_similarity(nemo_hidden_state, hf_hidden_state, attention_mask, rtol, atol)
    assert_cosine_similarity(nemo_logits, hf_logits, attention_mask, rtol, atol)


def load_and_evaluate_hf_amplify(
    model_tag: str, precision: PrecisionTypes, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Load a HuggingFace model and evaluate it on the given inputs.

    Args:
        model_tag: The HuggingFace model tag for the model to compare against.
        precision: The precision type to use for the comparison.
        input_ids: The input IDs tensor to evaluate.
        attention_mask: The attention mask tensor to evaluate.

    Returns:
        A tuple of the logits and hidden states tensors calculated by the HuggingFace model, respectively.
    """
    hf_model = AutoModel.from_pretrained(
        model_tag,
        torch_dtype=get_autocast_dtype(precision),
        trust_remote_code=True,
    )

    def hook_fn(module, inputs, outputs):
        hook_fn.inputs = inputs
        hook_fn.outputs = outputs

    hook_fn.inputs = None
    hook_fn.outputs = None

    hf_model.transformer_encoder[0].register_forward_hook(hook_fn)
    # hf_model.transformer_encoder[0].ffn.register_forward_hook(hook_fn)

    hf_model = hf_model.to("cuda").eval()
    hf_output_all = hf_model(input_ids, attention_mask.float(), output_hidden_states=True)
    hf_hidden_state = hf_output_all.hidden_states[-1]
    return hf_output_all.logits, hf_hidden_state, hook_fn.inputs, hook_fn.outputs


def load_and_evaluate_nemo_amplify(
    tokenizer: BioNeMoAMPLIFYTokenizer,
    ckpt_path: Path | str,
    precision: PrecisionTypes,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Load a AMPLIFY NeMo2 model checkpoint and evaluate it on the input tensors.

    It would be great to make this more ergonomic, i.e., how to create a model from a checkpoint and evaluate it.

    Args:
        tokenizer: Not sure why we need to pass a tokenizer to `configure_model`.
        ckpt_path: Path to the newly created NeMo2 converted checkpoint.
        precision: Precision type to use for the model.
        input_ids: Input tokens
        attention_mask: Input attention mask

    Returns:
        The logits and hidden states from the model.
    """

    dtype = get_autocast_dtype(precision)
    nemo_config = AMPLIFYConfig(
        initial_ckpt_path=str(ckpt_path),
        include_embeddings=True,
        include_hiddens=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
    )

    nemo_model = nemo_config.configure_model(tokenizer).to("cuda").eval()

    if dtype is torch.float16 or dtype is torch.bfloat16:
        nemo_model = Float16Module(nemo_config, nemo_model)

    def hook_fn(module, inputs, outputs):
        hook_fn.inputs = inputs
        hook_fn.outputs = outputs

    hook_fn.inputs = None
    hook_fn.outputs = None

    nemo_model.encoder.layers[0].self_attention.register_forward_hook(hook_fn)
    # nemo_model.encoder.layers[0].mlp.register_forward_hook(hook_fn)

    nemo_output = nemo_model(input_ids, attention_mask)
    nemo_logits = nemo_output["token_logits"].transpose(0, 1).contiguous()[..., : tokenizer.vocab_size]
    nemo_hidden_state = nemo_output["hidden_states"]
    return nemo_logits, nemo_hidden_state, hook_fn.inputs, hook_fn.outputs


def test_convert_amplify_120M_smoke(tmp_path):
    model_tag = "chandar-lab/AMPLIFY_120M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")


def test_convert_amplify_120M(tmp_path):
    model_tag = "chandar-lab/AMPLIFY_120M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_amplify_equivalence(tmp_path / "nemo_checkpoint", model_tag)


def test_convert_amplify_350M(tmp_path):
    model_tag = "chandar-lab/AMPLIFY_350M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_amplify_equivalence(tmp_path / "nemo_checkpoint", model_tag)
