# Copyright 2025 HOUMO AI
#
# File: mineru2.5.py
# Description:
#   Example script: llm/mineru2.5/mineru2.5.py
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

from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient

# for transformers>=4.56.0
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/data02/datasets/mineru2.5",
    # dtype="auto", # use `torch_dtype` instead of `dtype` for transformers<4.56.0
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "/data02/datasets/mineru2.5",
    use_fast=True
)

client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor
)

image = Image.open("data/images/houmo_logo.jpg")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)