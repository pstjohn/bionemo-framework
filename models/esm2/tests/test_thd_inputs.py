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
from transformers import AutoModelForMaskedLM

from esm.convert import convert_esm_hf_to_te
from esm.modeling_esm_te import NVEsmForMaskedLM


@pytest.fixture
def te_model_checkpoint(tmp_path):
    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"


def test_thd_values_match(te_model_checkpoint, input_data, input_data_thd):
    model_bshd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint)
    model_thd = NVEsmForMaskedLM.from_pretrained(te_model_checkpoint, attn_input_format="thd")
    model_bshd.to("cuda")
    model_thd.to("cuda")

    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    input_data_thd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd.items()}

    with torch.no_grad():
        bshd_outputs = model_bshd(**input_data, output_hidden_states=True)
        thd_outputs = model_thd(**input_data_thd, output_hidden_states=True)

    torch.testing.assert_close(bshd_outputs.loss, thd_outputs.loss, atol=1e-3, rtol=1e-3)
