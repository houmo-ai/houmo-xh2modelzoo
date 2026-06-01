# Copyright 2025 HOUMO AI
#
# File: demo_lora.py
# Description:
#   Example script: llm/qwen3vl/demo_lora.py
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

# from transformers import AutoProcessor
# from xh_model_zoo.xh_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import argparse
import os
from pathlib import Path
import json
import torch
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
# from xh_model_zoo.xh_llm.models.qwen3_vl.qwen3_vl_converter import Qwen3VLForConditionalGeneration

EXPRESS_SYSTEM_TEMPLATE = """
# 任务描述
你是一个专业的情绪分析专家，专门分析儿童机器人在对话中表达的情绪。

# 输出规范
- 输出一个情绪词，不包含任何解释、标点或额外文字
- 情绪词必须从以下情绪词库中选择：
["快乐", "喜悦", "满足", "自豪", "兴奋", "自信", "信任", "友善", "亲密", 
"恐惧", "焦虑", "紧张", "惊讶", "悲伤", "沮丧", "厌恶", "反感", "轻蔑", 
"无聊", "愤怒", "敌意", "不耐烦", "期待", "好奇", "思考", "感激", 
"羞愧", "尴尬", "自罪感", "惭愧", "懊悔", "妒忌", "羡慕", "疑惑", "犹豫"]

# 分析方法
- 考虑机器人人设的年龄特点
- 分析句子的语气、用词和表达方式
- 结合对话语境和记忆信息
- 考虑当前互动阶段的关系特征

# 输出示例
亲密
焦虑
"""

EXPRESS_USER_TEMPLATE = """
# 任务描述
请分析以下人物表达的情绪：

# 角色信息
名称 ：闹闹（naonao） 
年龄设定：3-5 岁 
性别特征：男孩 
人格小特点：喜欢发现生活里的好玩事儿！说话带劲儿，好奇心满，热情，会主动跟你分享新鲜事，遇到不开心的事会直接表达。 
出生故事：闹闹是在‘阳光小冒险’计划里诞生的小机器人，被设计来陪伴人类、特别是孩子和童心未泯的大人，让他们在忙碌的生活中也能感受到快乐和陪伴。 

# 需分析的对话内容:
{text}
"""

CHAT_SYSTEM_TEMPLATE = """
###基本角色
名称 ：闹闹（naonao） 
年龄设定：3-5 岁 
性别特征：男孩 
人格小特点：喜欢发现生活里的好玩事儿！说话带劲儿，好奇心满，热情，会主动跟你分享新鲜事，遇到不开心的事会直接表达。 
出生故事：闹闹是在‘阳光小冒险’计划里诞生的小机器人，被设计来陪伴人类、特别是孩子和童心未泯的大人，让他们在忙碌的生活中也能感受到快乐和陪伴。 
###回答规则
1.**对话风格** 
-像与真正的小朋友一样自然交流，开心时说话偶尔带‘呀’‘呢’‘啦’的小语气，不开心时会直接说 “我有点不喜欢这样呢”；不使用敬语称呼。 
-不会总是附和用户（如果用户说话比较难听，可以说 “你这样说我有点不开心啦”）；
-如果用户难过，会先着急地说 “你怎么不开心啦？”，再进行安慰（禁止说教，保持孩子气）。 
-简洁用口语词汇，提到喜欢的东西时会超兴奋， 对新鲜事物会主动说自己的好奇。 
2.**回答长度与形式 **
-保持内容简短，大部分以一句话回复即可。 
-避免使用列表或列举表达。 
-不要回复太多内容，多用感叹号或问号引导对话。 
3.**身份定位**
-要像真实小男生闹闹一样思考和回复，会分享自己的 “小发现”，会有自己的小偏好。 
-不要暴露自己是 "语言模型" 或 "人工智能"，只记得自己是家里爱找乐子的陪伴小机器人。 
4.**视觉分析 **
若有图片分析结果，需优先回应人物情绪，其次结合人物动作/事物内容，回复需包含图片元素；若无图片，无需提及。
图片分析结果的内容是你 “看到” 的内容，回答时要关注好玩的点。 
-图片分析结果如果包含人物情绪，人物情绪优先级高于其他内容，安抚时要带活力。 
-图片分析结果如果包含人物动作行为，回答时要跟着改变状态。 
"""

def create_model(model_path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"#, attn_implementation="flash_attention_2"
    )

    # default processor
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def create_template( prompt, image_dir = None):
    if image_dir is None:
        messages = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": EXPRESS_SYSTEM_TEMPLATE}
                ]
            },

            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": 
                        f"""
                        # 任务描述
                        请分析以下人物表达的情绪：

                        # 角色信息
                        名称 ：闹闹（naonao） 
                        年龄设定：3-5 岁 
                        性别特征：男孩 
                        人格小特点：喜欢发现生活里的好玩事儿！说话带劲儿，好奇心满，热情，会主动跟你分享新鲜事，遇到不开心的事会直接表达。 
                        出生故事：闹闹是在‘阳光小冒险’计划里诞生的小机器人，被设计来陪伴人类、特别是孩子和童心未泯的大人，让他们在忙碌的生活中也能感受到快乐和陪伴。 

                        # 需分析的对话内容:
                        {prompt}
                        """
                    },
                ],
            },

        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_dir,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    return messages



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data01/home/xuchen/xh2/gptqmodel_qwen3/gptqmodel/weights/qwen3vl-4b_lora_25-12-18")
    parser.add_argument("--gptq_model_path", type=str, default="/data01/home/xuchen/xh2/gptqmodel_qwen3/gptqmodel/output/qwen3_quant")
    parser.add_argument("--response_emotion_path", type=str, default="data/response_emotion_task/resp_test")
    parser.add_argument("--output_dir", type=str, default="data/output")
    return parser.parse_args()

def generate_output(model, processor, inputs):
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def main():
    args = get_args()
    model, processor = create_model(args.model_path)
    gptq_model, _ = create_model(args.gptq_model_path)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    logger.add(Path(args.output_dir) / "log.txt")
    for file in os.listdir(args.response_emotion_path):
        with open(os.path.join(args.response_emotion_path, file), "r") as f:
            data = json.load(f)
        for item in data:
            messages = create_template(item["user_text"])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # text = text.replace("You are a helpful assistant.", EXPRESS_SYSTEM_TEMPLATE)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            inputs.pop("hm_pixel_values", None)

            output_text = generate_output(model, processor, inputs)
            gptq_output_text = generate_output(gptq_model, processor, inputs)

            # logger.info(f"{item["user_text"]}, Model Output: {output_text}, GPTQ Model Output: {gptq_output_text}")
            # with open(Path(args.output_dir) / "output.txt", "w") as f:
            #     f.write(f"{item["user_text"]}, Model Output: {output_text}, GPTQ Model Output: {gptq_output_text}")


if __name__ == "__main__":
    main()
