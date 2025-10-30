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

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline

from convert import convert_llama_hf_to_te
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


def test_llama_model_forward_pass():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    model = NVLlamaForCausalLM(config)

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    assert outputs.logits is not None
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1


def test_llama_model_golden_values():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config.num_hidden_layers = 3
    model_hf = LlamaForCausalLM(config)
    model_te = convert_llama_hf_to_te(model_hf)

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_hf.to("cuda")
    model_te.to("cuda")

    with torch.no_grad():
        breakpoint()
        outputs_te = model_te.model(**inputs)
        outputs_hf = model_hf.model(**inputs)

    torch.testing.assert_close(outputs_te.last_hidden_state, outputs_hf.last_hidden_state)


def test_llama_model_golden_values_2():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config.num_hidden_layers = 3
    model_hf = LlamaForCausalLM(config)
    model_te = convert_llama_hf_to_te(model_hf)

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_hf.to("cuda")
    model_te.to("cuda")

    with torch.no_grad():
        embeds_te = model_te.model.embed_tokens(inputs["input_ids"])
        embeds_hf = model_hf.model.embed_tokens(inputs["input_ids"])

    torch.testing.assert_close(embeds_te, embeds_hf)

    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs["input_ids"].shape[1], device=inputs["input_ids"].device
    )
    position_ids = cache_position.unsqueeze(0)

    # Create some necessary tensors for the hidden layer forward passes
    with torch.no_grad():
        causal_mask = transformers.masking_utils.create_causal_mask(
            config=model_te.config,
            input_embeds=embeds_te,
            attention_mask=inputs["attention_mask"],
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            position_embeddings_te = model_te.model.rotary_emb(max_seq_len=embeds_te.shape[1]).to(
                embeds_te.device, dtype=embeds_te.dtype, non_blocking=True
            )
            position_embeddings_hf = model_hf.model.rotary_emb(embeds_hf, position_ids)

    hidden_states_hf = model_hf.model.layers[0](
        embeds_hf,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        cache_position=cache_position,
        position_embeddings=position_embeddings_hf,
    )

    hidden_states_te = model_te.model.layers[0](
        embeds_te,
        attention_mask=causal_mask,
        rotary_pos_emb=position_embeddings_te,
    )

    breakpoint()


# def test_llama_model_forward_pass():
#     tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
#     config = LlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
#     model = LlamaForCausalLM(config)

#     inputs = tokenizer("Hello, how are you?", return_tensors="pt")
#     inputs = {k: v.to("cuda") for k, v in inputs.items()}
#     model.to("cuda")
#     with torch.no_grad():
#         model(**inputs)


def test_llama_model_generate():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    model = NVLlamaForCausalLM(config)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

    prompt = "Hello, how are you?"
    generator(prompt, max_new_tokens=16)


# def test_llama_model_generate_golden_values():
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#     model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

#     generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

#     prompt = "Licensed under the Apache License, Version 2.0"
#     outputs = generator(prompt, max_new_tokens=16)
#     assert "you may not use this file except" in outputs[0]["generated_text"]


def test_llama_model_forward_golden_values(te_model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    inputs = tokenizer("Unless required by applicable law", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    model_hf = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf.to("cuda")

    with torch.no_grad():
        outputs_hf = model_hf(**inputs)

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    model_te = NVLlamaForCausalLM.from_pretrained(te_model_checkpoint)
    model_te.to("cuda")
    with torch.no_grad():
        outputs_te = model_te(**inputs)

    del model_te
    gc.collect()
    torch.cuda.empty_cache()

    breakpoint()

    torch.testing.assert_close(outputs_te.logits, outputs_hf.logits)
