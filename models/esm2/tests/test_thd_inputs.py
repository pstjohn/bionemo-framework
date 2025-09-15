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

import pytest
import torch
from transformers import AutoModelForMaskedLM, DataCollatorForTokenClassification, DataCollatorWithFlattening

from esm.convert import convert_esm_hf_to_te
from esm.modeling_esm_te import NVEsmForMaskedLM


@pytest.fixture
def te_model_checkpoint(tmp_path):
    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"


def test_thd_from_collator_output(te_model_checkpoint, input_data_thd):
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_thd.to("cuda")
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model_thd(**input_data_thd, output_hidden_states=True)

    assert outputs.loss < 3.0


def test_thd_values_match(te_model_checkpoint, tokenizer, input_data, input_data_thd):
    # Manually masked input tokens so that both BSHD and THD models have the same mask pattern
    sequences = [
        {
            "input_ids": [0, 32, 10, 15, 20, 1],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [-100, 5, -100, -100, -100, -100],
        },
        {
            "input_ids": [0, 25, 32, 15, 1],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, -100, 8, -100, -100],
        },
        {
            "input_ids": [0, 10, 12, 32, 18, 20, 22, 1],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, 14, -100, -100, -100, -100],
        },
    ]

    bhsd_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    thd_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

    manual_input_data_bhsd = bhsd_collator(sequences)
    manual_input_data_thd = thd_collator(sequences)

    model_bshd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, dtype=torch.bfloat16)
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd", dtype=torch.bfloat16)
    model_bshd.to("cuda")
    model_thd.to("cuda")

    manual_input_data_bhsd = {k: v.to("cuda") for k, v in manual_input_data_bhsd.items()}
    manual_input_data_thd = {
        k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in manual_input_data_thd.items()
    }

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        bshd_outputs = model_bshd(**manual_input_data_bhsd, output_hidden_states=True)
        thd_outputs = model_thd(**manual_input_data_thd, output_hidden_states=True)

    print("bshd_outputs.loss", bshd_outputs.loss)
    print("thd_outputs.loss", thd_outputs.loss)
    torch.testing.assert_close(bshd_outputs.loss, thd_outputs.loss, atol=1e-2, rtol=1e-2)
