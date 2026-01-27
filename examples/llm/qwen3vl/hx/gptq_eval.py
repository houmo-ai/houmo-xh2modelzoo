# Copyright 2025 HOUMO AI
#
# File: gptq_eval.py
# Description:
#   Example script: llm/qwen3vl/hx/gptq_eval.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.glm4_moe import Glm4MoeForCausalLM
from transformers import AutoTokenizer, Glm4MoeLiteForCausalLM, Glm4MoeLiteModel

if __name__ == "__main__":
    model_name_or_path = "Qwen/Qwen-3-VL-HF"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    inputs = tokenizer(
        "用户: 请简要介绍一下人工智能的发展历程。\n助手:",
        return_tensors="pt",
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.95,
        repetition_penalty=1.2,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))