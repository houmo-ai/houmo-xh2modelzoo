from evalscope.constants import ModelTask
from evalscope import TaskConfig, run_task
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch

# HiDream模型需要llama-3.1-8b-instruct模型作为Encoder
tokenizer_4 = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3.1-8B-Instruct")
text_encoder_4 = AutoModelForCausalLM.from_pretrained(
    "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)
# 配置评测参数
task_cfg = TaskConfig(
    model='HiDream-ai/HiDream-I1-Dev',  # 指定modelscope上的 model id
    model_task=ModelTask.IMAGE_GENERATION,  # 需要为 IMAGE_GENERATION
    # 配置模型参数，具体支持的参数参考对应的Pipeline
    model_args={
        'pipeline_cls': 'HiDreamImagePipeline',  # 指定使用 HiDreamImagePipeline
        'torch_dtype': 'torch.bfloat16',  # 使用 bfloat16 精度
        'tokenizer_4': tokenizer_4,  # 指定 tokenizer
        'text_encoder_4': text_encoder_4,  # 指定 text encoder
    },
    # 配置评测数据集
    datasets=[
        'evalmuse',
    ],
    # 配置模型生成参数，具体支持的参数参考对应的Pipeline
    generation_config={
        'height': 1024,  # 生成图片的高度
        'width': 1024,  # 生成图片的宽度
        'num_inference_steps': 28,  # 对于HiDream-Dev，建议生成步数为28
        'guidance_scale': 0.0,  # 对于HiDream-Dev，建议使用0.0
    },
    # 是否需要生成分析报告
    analysis_report=True,
)

# 运行评测任务
run_task(task_cfg=task_cfg)