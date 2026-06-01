# Copyright 2025 HOUMO AI
#
# File: deepseek_ocr_export.py
# Description:
#   Example script: llm/deepseek_ocr/deepseek_ocr_export.py
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

import argparse,os.path as osp,torch
from xh_model_zoo.xh_llm.models.deepseek_ocr._export_vision_utils import vision_processor, vision_processor_gundam, export_hmonnx
from xh_model_zoo.xh_llm.models.deepseek_ocr._export_deepseekv2 import export_deepseekv2


def main(args):
    from xh_model_zoo.xh_llm.models.deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
    from xh_model_zoo.xh_llm.models.deepseek_ocr._model_vision import register_wrap_modules as vision_register_wrap_cls


    model:DeepseekOCRForCausalLM = DeepseekOCRForCausalLM.from_pretrained(args.model, 
                                _attn_implementation='eager', 
                                trust_remote_code=True, 
                                use_safetensors=True)
    


    if args.export_mode in ["Tiny", "Small", "Base", "Large"]:
        if args.export_mode == "Tiny":
            input_resolution=512
        elif args.export_mode == "small":
            input_resolution=640
        elif args.export_mode == "base":
            input_resolution=1024
        elif args.export_mode == "large":
            input_resolution=1280

        vision_processor_model = vision_processor(model.model)
        inputs = []
        inputs.append(torch.randn([1, 3, input_resolution, input_resolution]))  # image_original
        export_hmonnx(vision_processor_model, 
                        inputs, 
                        "vision_processor",
                        vision_register_wrap_cls,
                        args,
                        )
    elif args.export_mode == "gundam":
        input_resolution_1=640
        input_resolution_2=1024
        vision_processor_model = vision_processor_gundam(model.model)
        inputs = []
        inputs.append(torch.randn([1, 3, input_resolution_1, input_resolution_1]))  # image_original_1
        inputs.append(torch.randn([1, 3, input_resolution_2, input_resolution_2]))  # image_original_2
        export_hmonnx(vision_processor_model, 
                        inputs, 
                        "vision_processor_gundam",
                        vision_register_wrap_cls,
                        args,
                        )
        


    else:
        raise NotImplementedError(f"Not support export mode: {args.export_mode}")
    

    export_deepseekv2(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")  # /data02/datasets/qwen3-8B-AWQ
    parser.add_argument(
        "--model",
        type=str,
        default="/data01/datasets/DeepSeek-OCR/",  # /data02/datasets/Qwen2.5-1.5B-Instruct-int4-sym-inc
    )
    parser.add_argument("--export-mode", type=str, default="Tiny", choices=["Tiny", "Small", "Base", "Large", "Gundam"],help="export mode, default is hmonnx")
    parser.add_argument("--context-length", type=int, default=8192, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--num_logits_to_keep", type=int, default=1, help="not for test ppl")
    parser.add_argument("--quant-type", default="w8a8h0_ssfp", help="quant type, default is w8a8")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    parser.add_argument("--out_dirs", default="./work_dirs/DeepSeek-OCR-XH2a", action="store_true", help="demo mode")
    parser.add_argument("--device",type=str,default="xh2a",help="device, default is xh2a")
    parser.add_argument("--generate-golden", default=True, help="generate golden")
    parser.add_argument("--extra_config_file", type=str, default=None, help="extra config file")
    parser.add_argument("--demo", default=False, action="store_true", help="demo mode")
    args = parser.parse_args()
    main(args)
