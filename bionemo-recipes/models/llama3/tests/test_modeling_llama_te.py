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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM


def test_llama_model_forward_pass():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    model = NVLlamaForCausalLM(config)

    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model.to("cuda")
    with torch.no_grad():
        model(**inputs)


def test_llama_model_generate():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    config = NVLlamaConfig.from_pretrained("nvidia/Llama-3.1-8B-Instruct-FP8")
    model = NVLlamaForCausalLM(config)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

    prompt = "Hello, how are you?"
    generator(prompt, max_new_tokens=16)


def test_llama_model_generate_golden_values():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

    prompt = "Licensed under the Apache License, Version 2.0"
    outputs = generator(prompt, max_new_tokens=16)
    assert "you may not use this file except" in outputs[0]["generated_text"]
