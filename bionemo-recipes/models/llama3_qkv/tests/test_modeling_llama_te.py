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
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline

from convert import convert_llama_hf_to_te
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


def test_llama_model_forward_pass():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    model = NVLlamaForCausalLM(config)

    inputs = tokenizer("Licensed under the Apache License, Version 2.0", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    assert outputs.logits is not None
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1


def test_llama_model_golden_values():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16)

    model_te = convert_llama_hf_to_te(model_hf)

    # model_hf.model.layers = model_hf.model.layers[:1]
    # model_te.model.layers = model_te.model.layers[:1]

    inputs = tokenizer(
        """Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.""",
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_hf.to("cuda")
    with torch.no_grad():
        outputs_hf = model_hf(**inputs, labels=inputs["input_ids"], output_hidden_states=True)

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    model_te.to("cuda")
    with torch.no_grad():
        outputs_te = model_te(**inputs, labels=inputs["input_ids"], output_hidden_states=True)

    torch.testing.assert_close(outputs_te.loss, outputs_hf.loss)


def test_llama_model_golden_values_2():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_hf

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    model_te = convert_llama_hf_to_te(model_hf)

    model_hf.model.layers = model_hf.model.layers[:1]
    model_te.model.layers = model_te.model.layers[:1]

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_hf.to("cuda")
    model_te.to("cuda")

    with torch.no_grad():
        outputs_hf = model_hf.model(**inputs)
        outputs_te = model_te.model(**inputs)

    with torch.no_grad():
        inputs_embeds = model_te.model.embed_tokens(inputs["input_ids"])

    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs["input_ids"].shape[1],
        device=inputs["input_ids"].device,
    )
    position_ids = cache_position.unsqueeze(0)

    hidden_states = inputs_embeds
    position_embeddings_hf = model_hf.model.rotary_emb(hidden_states, position_ids)
    position_embeddings_te = model_te.model.rotary_emb(hidden_states, position_ids).transpose(0, 1).unsqueeze(1)

    torch.testing.assert_close(
        position_embeddings_hf[0][0], position_embeddings_te.cos()[:, 0, 0, :], atol=1e-8, rtol=1e-8
    )
    torch.testing.assert_close(
        position_embeddings_hf[1][0], position_embeddings_te.sin()[:, 0, 0, :], atol=1e-8, rtol=1e-8
    )

    # HF decoder layer forward pass
    breakpoint()
    hf_layer = model_hf.model.layers[0]
    residual = hidden_states
    hidden_states = hf_layer.input_layernorm(hidden_states)

    # LlamaAttention.forward
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, hf_layer.self_attn.head_dim)
    query_states = hf_layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = hf_layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = hf_layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings_hf
    query_states, key_states = apply_rotary_pos_emb_hf(query_states, key_states, cos, sin)
    attention_interface = ALL_ATTENTION_FUNCTIONS[model_hf.config._attn_implementation]
    attn_output, _ = attention_interface(
        hf_layer.self_attn,
        query_states,
        key_states,
        value_states,
        None,
        dropout=0.0,
        scaling=hf_layer.self_attn.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    hidden_states = hf_layer.self_attn.o_proj(attn_output)

    # LlamaDecoderLayer.forward
    hf_self_attention_outputs = hidden_states.clone()  # qkv, attn, and output proj.
    hidden_states = residual + hidden_states
    residual = hidden_states
    hf_mlp_inputs = hidden_states.clone()
    hidden_states = hf_layer.post_attention_layernorm(hidden_states)
    hidden_states = hf_layer.mlp(hidden_states)
    hf_mlp_outputs = hidden_states.clone()
    hidden_states = residual + hidden_states  # L310
    hf_pre_norm_hidden_states = hidden_states.clone()

    hidden_states = model_hf.model.norm(hidden_states)  # L405
    torch.testing.assert_close(outputs_hf.last_hidden_state, hidden_states, atol=1e-8, rtol=1e-8)

    # TE decoder layer forward pass
    breakpoint()
    te_layer = model_te.model.layers[0]
    hidden_states = inputs_embeds
    attention_output, _ = te_layer.self_attention(
        inputs_embeds,
        attention_mask=None,
        attn_mask_type="causal",
        rotary_pos_emb=position_embeddings_te,
    )
    torch.testing.assert_close(hf_self_attention_outputs, attention_output, atol=3e-6, rtol=1e-6)  # 2.7e-06

    hidden_states = attention_output + inputs_embeds

    mlp_outputs, _ = te_layer.layernorm_mlp(
        hidden_states,
        is_first_microbatch=False,
    )
    # with the same inputs, this yields ~3.5e-6 abs error from the HF version.
    torch.testing.assert_close(hf_mlp_outputs, mlp_outputs, atol=2.3e-5, rtol=1e-6)  # 2.3e-05

    hidden_states = mlp_outputs + hidden_states
    torch.testing.assert_close(hf_pre_norm_hidden_states, hidden_states, atol=2.1e-5, rtol=1e-6)  # 2.1e-05

    te_pre_norm_hidden_states = hidden_states.clone()
    hidden_states = model_te.model.norm(hidden_states)  # L405
    torch.testing.assert_close(outputs_te.last_hidden_state, hidden_states, atol=1e-8, rtol=1e-8)

    torch.testing.assert_close(outputs_te.last_hidden_state, outputs_hf.last_hidden_state, atol=1e-3, rtol=1e-3)


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


def test_layernorm_linear_equivalence():
    from transformer_engine.pytorch.module.layernorm_linear import LayerNormLinear
    from transformer_engine.pytorch.utils import SplitAlongDim

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_hf = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_te = convert_llama_hf_to_te(model_hf)

    model_hf.model.layers = model_hf.model.layers[:1]
    model_te.model.layers = model_te.model.layers[:1]

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model_hf.to("cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        inputs_embeds = model_hf.model.embed_tokens(inputs["input_ids"])

    hf_layer = model_hf.model.layers[0]
    hidden_states = hf_layer.input_layernorm(inputs_embeds)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, hf_layer.self_attn.head_dim)
    query_states = hf_layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    torch.testing.assert_close(query_states[0].transpose(0, 1), torch.load("hf_pre_rot.pt"), atol=1e-8, rtol=1e-8)

    # Just create a single identical LayerNormLinear
    te_layernorm_linear = LayerNormLinear(
        hf_layer.self_attn.q_proj.in_features,
        hf_layer.self_attn.q_proj.out_features,
        eps=1e-5,
        normalization="RMSNorm",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    te_layernorm_linear.load_state_dict(
        {
            "layer_norm_weight": hf_layer.input_layernorm.state_dict()["weight"],
            "weight": hf_layer.self_attn.q_proj.state_dict()["weight"],
            "_extra_state": te_layernorm_linear.state_dict()["_extra_state"],
        }
    )
    te_layernorm_linear.to("cuda")
    with torch.no_grad():
        te_hidden_states = te_layernorm_linear(inputs_embeds).view(hidden_shape).transpose(1, 2)
    torch.testing.assert_close(query_states, te_hidden_states, atol=1e-6, rtol=1e-6)  # 1.2e-07

    # Now try the layernorm_qkv from the TE model
    model_te.to("cuda")
    te_self_attention = model_te.model.layers[0].self_attention
    mixed_x_layer = te_self_attention.layernorm_qkv(
        inputs_embeds,
        is_first_microbatch=False,
        fp8_output=False,
    )

    # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, (np/ng + 2), ng, hn]
    num_queries_per_key_value = (
        te_self_attention.num_attention_heads_per_partition // te_self_attention.num_gqa_groups_per_partition
    )
    new_tensor_shape = mixed_x_layer.size()[:-1] + (
        (num_queries_per_key_value + 2),
        te_self_attention.num_gqa_groups_per_partition,
        te_self_attention.hidden_size_per_attention_head,
    )
    # split along third last dimension
    split_dim = -3

    mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

    query_layer, key_layer, value_layer = SplitAlongDim.apply(
        mixed_x_layer, split_dim, (num_queries_per_key_value, 1, 1)
    )
    query_layer, key_layer, value_layer = (
        x.reshape(x.size(0), x.size(1), -1, te_self_attention.hidden_size_per_attention_head)
        for x in (query_layer, key_layer, value_layer)
    )

    torch.testing.assert_close(te_hidden_states.transpose(1, 2), query_layer)

    breakpoint()
